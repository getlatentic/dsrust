"""A dspy.ChatAdapter whose rendering is this crate's Rust, for running upstream's own tests.

Upstream's suite constructs `dspy.ChatAdapter()` and asserts on the messages it returns.
Subclassing it and overriding `format` and `parse` means their tests run unmodified while the
bytes under assertion come from Rust.

There is deliberately no deferral to the Python implementation. A shim that quietly handed an
unsupported case back to dspy would report a pass for code this crate has not written, which
is the one thing a conformance suite must never do. When Rust cannot render a case this
raises, the test goes red, and `conftest.py` carries the red ones as a declared to-do list.
"""

from __future__ import annotations

import json
import typing

import dspy
import pydantic
from dspy.adapters.base import Adapter
from dspy.adapters.json_adapter import JSONAdapter
from dspy.adapters.utils import format_field_value, get_annotation_name, parse_value
from dspy.utils.exceptions import AdapterParseError
from litellm import ContextWindowExceededError

import dsrs_bridge

# The Rust FieldKind each Python annotation maps to. Anything absent is not yet modelled.
KINDS: dict[typing.Any, str] = {str: "str", int: "int", float: "float", bool: "bool"}


class Unsupported(Exception):
    """The case needs a Rust feature that does not exist yet. Never caught here on purpose."""


def kind_of(annotation: typing.Any) -> str:
    """A scalar names itself; anything else carries the name dspy would print.

    Sending `json:<annotation>` rather than a bare `json` is what lets the numbered line read
    `(PetOwner)` the way dspy renders it, instead of collapsing every non-scalar to one word.
    """
    try:
        return KINDS[annotation]
    except (KeyError, TypeError):
        pass
    if isinstance(annotation, type) and issubclass(annotation, pydantic.BaseModel):
        return f"json:{get_annotation_name(annotation)}"
    raise Unsupported(f"no Rust FieldKind for annotation {annotation!r}")


def describe(fields: dict) -> list[tuple]:
    described = []
    for name, info in fields.items():
        desc = info.json_schema_extra.get("desc") or ""
        if desc == f"${{{name}}}":  # dspy's placeholder for "no description given"
            desc = ""
        described.append((name, kind_of(info.annotation), desc))
    return described


def described_outputs(signature) -> list[tuple]:
    return [(name, kind, desc, None) for name, kind, desc in describe(signature.output_fields)]


class RustChatAdapter(dspy.ChatAdapter):
    """Renders and parses through Rust, or raises. It never falls through to Python."""

    def __call__(self, lm, lm_kwargs, signature, demos, inputs):
        """dspy's ChatAdapter re-asks through the JSON adapter when a reply fails to parse.

        Inheriting dspy's `__call__` would let its Python flag make that decision, and the
        conformance run would pass without this crate's policy ever being consulted. Asking
        Rust keeps the decision where the implementation is; Python still owns the model call
        itself, because that is litellm's job on this side of the bridge.
        """
        try:
            return Adapter.__call__(self, lm, lm_kwargs, signature, demos, inputs)
        except Unsupported:
            # dspy's fallback exists for a reply the marker parser cannot read, not for a
            # case this crate has not written. Letting it swallow `Unsupported` would run the
            # exchange on Python's JSONAdapter and report a pass for absent Rust — the one
            # thing this harness must never do.
            raise
        except Exception as error:
            fallback = dsrs_bridge.has_json_fallback("chat", self.use_json_adapter_fallback)
            if isinstance(error, ContextWindowExceededError) or not fallback:
                raise
            return JSONAdapter()(lm, lm_kwargs, signature, demos, inputs)

    def format(self, signature, demos, inputs) -> list[dict[str, typing.Any]]:
        rendered_demos = [
            [
                (name, format_field_value(field_info=field, value=demo[name]))
                for name, field in {**signature.input_fields, **signature.output_fields}.items()
                if name in demo
            ]
            for demo in demos
        ]
        values = [
            (name, format_field_value(field_info=signature.input_fields[name], value=value))
            for name, value in inputs.items()
            if name in signature.input_fields
        ]
        system, turns = dsrs_bridge.format_messages(
            "chat",
            signature.instructions,
            describe(signature.input_fields),
            described_outputs(signature),
            values,
            rendered_demos,
        )
        return [{"role": "system", "content": system}] + [
            {"role": role, "content": content} for role, content in turns
        ]

    def parse(self, signature, completion):
        # Rust reports a parse failure as an error; dspy's contract is that a ChatAdapter
        # raises AdapterParseError, and callers — including its own fallback path — branch on
        # that type. Translating at the boundary is this shim's job, the same as field kinds.
        try:
            rendered = dsrs_bridge.parse_reply(
                "chat",
                signature.instructions,
                describe(signature.input_fields),
                described_outputs(signature),
                completion,
            )
        except ValueError as error:
            raise AdapterParseError(
                adapter_name="ChatAdapter",
                signature=signature,
                lm_response=completion,
                message=str(error),
            ) from error
        parsed = json.loads(rendered)
        # Rust hands back strings; dspy's callers expect the signature's declared types.
        return {
            name: parse_value(value, signature.output_fields[name].annotation)
            for name, value in parsed.items()
            if name in signature.output_fields
        }
