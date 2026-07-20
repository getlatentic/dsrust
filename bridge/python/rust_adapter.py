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
from dspy.adapters.utils import format_field_value, parse_value
from dspy.utils.exceptions import AdapterParseError

import dsrs_bridge

# The Rust FieldKind each Python annotation maps to. Anything absent is not yet modelled.
KINDS: dict[typing.Any, str] = {str: "str", int: "int", float: "float", bool: "bool"}


class Unsupported(Exception):
    """The case needs a Rust feature that does not exist yet. Never caught here on purpose."""


def kind_of(annotation: typing.Any) -> str:
    try:
        return KINDS[annotation]
    except (KeyError, TypeError):
        raise Unsupported(f"no Rust FieldKind for annotation {annotation!r}") from None


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

    def format(self, signature, demos, inputs) -> list[dict[str, typing.Any]]:
        if demos:
            raise Unsupported("demos are not implemented in Rust yet")
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
