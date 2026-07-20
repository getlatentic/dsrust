"""A dspy.ChatAdapter whose rendering is this crate's Rust, for running upstream's own tests.

Upstream's suite constructs `dspy.ChatAdapter()` and asserts on the messages it returns.
Subclassing it and overriding only `format` means their tests run unmodified while the bytes
under assertion come from Rust. Anything not yet implemented in Rust — demos, images, tool
calls — falls through to the Python implementation, so an unsupported case is a Python pass
rather than a false Rust failure. `RUST_ADAPTER_STRICT=1` turns those into errors instead,
which is how you find out what is left to build.
"""

from __future__ import annotations

import json
import os
import typing

import dspy
from dspy.adapters.utils import format_field_value

import dsrs_bridge

STRICT = os.environ.get("RUST_ADAPTER_STRICT") == "1"

# The Rust FieldKind each Python annotation maps to. Anything absent is not yet modelled.
KINDS: dict[typing.Any, str] = {str: "str", int: "int", float: "float", bool: "bool"}


class Unsupported(Exception):
    """The case needs a Rust feature that does not exist yet."""


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


class RustChatAdapter(dspy.ChatAdapter):
    """Renders through Rust; defers to Python for anything Rust cannot express yet."""

    def format(self, signature, demos, inputs) -> list[dict[str, typing.Any]]:
        try:
            if demos:
                raise Unsupported("demos are not implemented in Rust yet")
            described_inputs = describe(signature.input_fields)
            described_outputs = [
                (name, kind, desc, None) for name, kind, desc in describe(signature.output_fields)
            ]
            values = [
                (name, format_field_value(field_info=signature.input_fields[name], value=value))
                for name, value in inputs.items()
                if name in signature.input_fields
            ]
            system, user = dsrs_bridge.format_messages(
                "chat",
                signature.instructions,
                described_inputs,
                described_outputs,
                values,
            )
        except Unsupported:
            if STRICT:
                raise
            return super().format(signature, demos, inputs)
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def parse(self, signature, completion):
        try:
            described_inputs = describe(signature.input_fields)
            described_outputs = [
                (name, kind, desc, None) for name, kind, desc in describe(signature.output_fields)
            ]
            parsed = json.loads(
                dsrs_bridge.parse_reply(
                    "chat", signature.instructions, described_inputs, described_outputs, completion
                )
            )
        except (Unsupported, ValueError):
            if STRICT:
                raise
            return super().parse(signature, completion)
        # Rust hands back strings; dspy's callers expect the signature's declared types.
        return {
            name: dspy.adapters.utils.parse_value(value, signature.output_fields[name].annotation)
            for name, value in parsed.items()
            if name in signature.output_fields
        }
