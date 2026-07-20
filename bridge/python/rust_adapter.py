"""A dspy.ChatAdapter whose rendering is this crate's Rust, for running upstream's own tests.

Upstream's suite constructs `dspy.ChatAdapter()` and asserts on the messages it returns.
Subclassing it and overriding `format` and `parse` means their tests run unmodified while the
bytes under assertion come from Rust.

The re-ask through `JSONAdapter` survives here because dspy has it: a ChatAdapter that cannot
read a reply retries through the JSON one, and upstream tests that behaviour directly. What
must never reach that path is a case Rust has not implemented, because rendering it on Python
would report a pass for code this crate has not written — the one thing a conformance suite
must never do. `Unsupported` is therefore a `BaseException`, out of reach of every
`except Exception` on both the sync and async paths, so an unimplemented case goes red and
`conftest.py` carries the red ones as a declared to-do list.
"""

from __future__ import annotations

import json
import typing

import dspy
import pydantic
from dspy.adapters.base import Adapter
from dspy.adapters.json_adapter import JSONAdapter
from dspy.adapters.types.base_type import Type
from dspy.adapters.types.code import Code
from dspy.adapters.utils import (
    _annotation_is_subclass,
    _get_json_schema,
    format_field_value,
    get_annotation_name,
    parse_value,
    serialize_for_json,
)
from dspy.utils.exceptions import AdapterParseError
from litellm import ContextWindowExceededError

import dsrs_bridge

# The Rust FieldKind each Python annotation maps to. Anything absent is not yet modelled.
KINDS: dict[typing.Any, str] = {str: "str", int: "int", float: "float", bool: "bool"}


class Unsupported(BaseException):
    """The case needs a Rust feature that does not exist yet.

    Deriving from `BaseException` puts this outside `except Exception`, which is what dspy's
    JSON-adapter fallback catches on both paths. Inheriting from `Exception` would leave the
    exclusion resting on a matching `except Unsupported: raise` in every handler that could
    ever see one, and the async path already inherits a handler this shim does not write.
    """


def kind_of(annotation: typing.Any) -> str:
    """A scalar names itself; anything else carries the name dspy would print.

    Sending `json:<annotation>` rather than a bare `json` is what lets the numbered line read
    `(PetOwner)` the way dspy renders it, instead of collapsing every non-scalar to one word.
    """
    if typing.get_origin(annotation) is typing.Literal:
        # The closed set travelling beside this becomes the printed annotation, so what is
        # named here is the wire type under it: text, which is how every member of a Literal
        # reaches the marker path whatever its Python type.
        return "str"
    try:
        return KINDS[annotation]
    except (KeyError, TypeError):
        pass
    if _carries_as_json(annotation):
        return f"json:{get_annotation_name(annotation)}"
    raise Unsupported(f"no Rust FieldKind for annotation {annotation!r}")


def _carries_as_json(annotation: typing.Any) -> bool:
    """Whether the crate's `Json` kind can carry values of this annotation.

    A model does, since its values are objects. A container does exactly when everything it
    holds does — `list[Tool]` rides on `Tool` — because the container is JSON either way and
    what is inside it is what has to survive the crossing. Anything else says so rather than
    crossing as an annotation whose values were never checked: a kind that renders a value
    wrongly is worse than one that refuses it, because the prompt still looks plausible.
    """
    if isinstance(annotation, type) and issubclass(annotation, pydantic.BaseModel):
        return True
    args = typing.get_args(annotation)
    return bool(args) and all(
        arg is Ellipsis or arg is type(None) or _scalar_or_json(arg) for arg in args
    )


def _scalar_or_json(annotation: typing.Any) -> bool:
    """Whether one member of a container is itself carryable."""
    try:
        return annotation in KINDS or _carries_as_json(annotation)
    except TypeError:
        return _carries_as_json(annotation)


def closed_set_of(annotation: typing.Any) -> str | None:
    """A `Literal`'s members as JSON, or None where the annotation is not one.

    JSON is what the bridge carries a closed set over, and it spells the three member types
    Rust models. A `Literal` over anything else — an Enum, None, bytes — has no crossing yet,
    and must say so rather than lose members on the way across.
    """
    if typing.get_origin(annotation) is not typing.Literal:
        return None
    members = typing.get_args(annotation)
    if not all(isinstance(member, (str, int, bool)) for member in members):
        raise Unsupported(f"no Rust closed set for annotation {annotation!r}")
    return json.dumps(members)


def schema_of(kind: str, annotation: typing.Any) -> str | None:
    """A structured field's JSON schema, as dspy builds it, or None for a scalar.

    Reading a schema off a Python annotation is pydantic's job, so upstream's own extractor runs
    here — key order included, since it is part of the bytes. Whether the schema reaches the
    prompt stays the crate's decision: a type whose description already states its contract
    drops the note, and the crate is what knows that.

    Only a structured field is asked for one, which is where the crate consults it and where
    dspy computes it. An annotation that cannot produce one is a gap in this bridge rather than
    a field to render blank, so it says so instead of quietly dropping the schema — a missing
    note renders a prompt that looks right and is not.
    """
    if not kind.startswith("json:"):
        return None
    try:
        return json.dumps(_get_json_schema(annotation), ensure_ascii=False)
    except Exception as error:
        raise Unsupported(f"no JSON schema for annotation {annotation!r}: {error}") from error


def type_descriptions_of(annotation: typing.Any) -> str | None:
    """The custom types an annotation names, as JSON `[[name, prose], ...]`, or None.

    Which types an annotation mentions is Python reflection, so it happens here; how the pairs
    read in a prompt is the crate's business, so only the pairs cross.
    """
    described = [
        {
            "name": get_annotation_name(custom),
            "text": custom.description(),
            # dspy asks whether the annotation *is* a `dspy.Code`, not what it is called, so
            # this asks the same way: a subclass counts and a look-alike does not.
            "replaces_schema": _annotation_is_subclass(custom, Code),
        }
        for custom in Type.extract_custom_type_from_annotation(annotation)
        if custom.description()
    ]
    return json.dumps(described) if described else None


def describe(fields: dict) -> list[tuple]:
    described = []
    for name, info in fields.items():
        desc = info.json_schema_extra.get("desc") or ""
        if desc == f"${{{name}}}":  # dspy's placeholder for "no description given"
            desc = ""
        annotation = info.annotation
        described.append(
            (
                name,
                kind_of(annotation),
                desc,
                closed_set_of(annotation),
                type_descriptions_of(annotation),
            )
        )
    return described


def described_outputs(signature) -> list[tuple]:
    """Outputs carry the nested schema of a structured field ahead of the closed set."""
    return [
        (name, kind, desc, schema_of(kind, signature.output_fields[name].annotation), values, types)
        for name, kind, desc, values, types in describe(signature.output_fields)
    ]


class _RustBacked:
    """Rendering and parsing through Rust, for whichever wire format subclasses it.

    Both adapters cross the same bridge and differ only in the format they name, so the two
    calls live here once. `WIRE` picks the crate's adapter; `ADAPTER_NAME` is what dspy's own
    error type expects to be told.
    """

    WIRE: str
    ADAPTER_NAME: str
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
            (name, json.dumps(serialize_for_json(value), ensure_ascii=False))
            for name, value in inputs.items()
            if name in signature.input_fields
        ]
        system, turns = dsrs_bridge.format_messages(
            self.WIRE,
            signature.instructions,
            describe(signature.input_fields),
            described_outputs(signature),
            values,
            rendered_demos,
        )
        return [{"role": "system", "content": system}] + [
            {"role": role, "content": json.loads(content)} for role, content in turns
        ]

    def parse(self, signature, completion):
        # Rust reports a parse failure as an error; dspy's contract is that a ChatAdapter
        # raises AdapterParseError, and callers — including its own fallback path — branch on
        # that type. Translating at the boundary is this shim's job, the same as field kinds.
        try:
            rendered = dsrs_bridge.parse_reply(
                self.WIRE,
                signature.instructions,
                describe(signature.input_fields),
                described_outputs(signature),
                completion,
            )
        except ValueError as error:
            # Rust sends the declared fields it did find as a second argument when the reply
            # named the wrong ones; dspy reports those on the error rather than only the text.
            message, *partial = error.args
            # dspy states no message of its own for a field mismatch — the fields it recovered
            # are the report — so passing one would prepend text upstream never writes.
            detail = (
                {"parsed_result": json.loads(partial[0])}
                if partial
                else {"message": message}
            )
            raise AdapterParseError(
                adapter_name=self.ADAPTER_NAME,
                signature=signature,
                lm_response=completion,
                **detail,
            ) from error
        parsed = json.loads(rendered)
        # Rust hands back strings; dspy's callers expect the signature's declared types.
        return {
            name: parse_value(value, signature.output_fields[name].annotation)
            for name, value in parsed.items()
            if name in signature.output_fields
        }

class RustChatAdapter(_RustBacked, dspy.ChatAdapter):
    """The marker-based adapter, whose parse failures may re-ask through Python's JSON one."""

    WIRE = "chat"
    ADAPTER_NAME = "ChatAdapter"

    def _retry_adapter(self, error):
        """The adapter to re-ask through after `error`, or no adapter if it must propagate.

        dspy's ChatAdapter decides this from its own Python flag. Both of its entry points are
        overridden below so the decision comes from Rust instead, because a conformance run
        that never consults this crate's policy is not testing it. Python still owns the model
        call itself, since that is litellm's job on this side of the bridge.
        """
        if isinstance(error, ContextWindowExceededError):
            return None
        if not dsrs_bridge.has_json_fallback("chat", self.use_json_adapter_fallback):
            return None
        return JSONAdapter()

    def __call__(self, lm, lm_kwargs, signature, demos, inputs):
        try:
            return Adapter.__call__(self, lm, lm_kwargs, signature, demos, inputs)
        except Exception as error:
            retry = self._retry_adapter(error)
            if retry is None:
                raise
            return retry(lm, lm_kwargs, signature, demos, inputs)

    async def acall(self, lm, lm_kwargs, signature, demos, inputs):
        try:
            return await Adapter.acall(self, lm, lm_kwargs, signature, demos, inputs)
        except Exception as error:
            retry = self._retry_adapter(error)
            if retry is None:
                raise
            return await retry.acall(lm, lm_kwargs, signature, demos, inputs)


class RustJSONAdapter(_RustBacked, dspy.JSONAdapter):
    """The provider-native structured-output adapter, rendered and parsed by the crate."""

    WIRE = "json"
    ADAPTER_NAME = "JSONAdapter"
