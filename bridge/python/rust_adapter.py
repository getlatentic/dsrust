"""dspy adapters whose rendering is this crate's Rust, for running upstream's own tests.

Upstream's suite constructs `dspy.ChatAdapter()` and asserts on the messages it returns.
Subclassing it and overriding `format` and `parse` means their tests run unmodified while the
bytes under assertion come from Rust. `reflect.py` answers the questions about a signature
that only Python can answer; everything a model reads is written on the other side.

The re-ask through `JSONAdapter` survives here because dspy has it: a ChatAdapter that cannot
read a reply retries through the JSON one, and upstream tests that behaviour directly. What
must never reach that path is a case Rust has not implemented, because rendering it on Python
would report a pass for code this crate has not written — the one thing a conformance suite
must never do. `reflect.Unsupported` is therefore a `BaseException`, out of reach of every
`except Exception` on both the sync and async paths, so an unimplemented case goes red and
`conftest.py` carries the red ones as a declared to-do list.
"""

from __future__ import annotations

import json
import typing

import dspy
import pydantic
from dspy.adapters.base import Adapter
from dspy.adapters.baml_adapter import BAMLAdapter
from dspy.adapters.json_adapter import JSONAdapter
from dspy.adapters.xml_adapter import XMLAdapter
from dspy.adapters.utils import (
    format_field_value,
    parse_value,
    serialize_for_json,
)
from dspy.utils.exceptions import AdapterParseError
from litellm import ContextWindowExceededError

import dsrs_bridge
from reflect import describe, described_outputs

#: How many times the crate rendered or parsed. A test that passes without moving this never
#: exercised Rust, whatever its name says, so `conftest.py` refuses to count it as conformance.
CROSSINGS = 0


class _RustBacked:
    """Rendering and parsing through Rust, for whichever wire format subclasses it.

    Every adapter crosses the same bridge and differs only in the format it names, so the calls
    live here once. `WIRE` picks the crate's adapter; `ADAPTER_NAME` is what dspy's own error
    type expects to be told.
    """

    WIRE: str
    ADAPTER_NAME: str

    def crossing_value(self, value: typing.Any) -> str:
        """One input value as the JSON text it crosses in.

        Serializing a Python object is pydantic's job and stays here. How the text is laid out
        in a prompt is the crate's, so nothing about that is decided on this side.
        """
        return json.dumps(serialize_for_json(value), ensure_ascii=False)

    def format_system_message(self, signature) -> str:
        """dspy exposes this on its own, and a caller reading it should read the crate's."""
        global CROSSINGS
        CROSSINGS += 1
        return dsrs_bridge.format_system_message(
            self.WIRE,
            signature.instructions,
            describe(signature.input_fields),
            described_outputs(signature),
        )

    def format(self, signature, demos, inputs) -> list[dict[str, typing.Any]]:
        global CROSSINGS
        CROSSINGS += 1
        rendered_demos = [
            [
                (name, format_field_value(field_info=field, value=demo[name]))
                for name, field in {**signature.input_fields, **signature.output_fields}.items()
                if name in demo
            ]
            for demo in demos
        ]
        values = [
            (name, self.crossing_value(value))
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
        global CROSSINGS
        CROSSINGS += 1
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


class RustXMLAdapter(_RustBacked, XMLAdapter):
    """Tag-wrapped fields, rendered and parsed by the crate."""

    WIRE = "xml"
    ADAPTER_NAME = "XMLAdapter"


class RustBAMLAdapter(_RustBacked, BAMLAdapter):
    """Each output's type stated as a compact notation, rendered and parsed by the crate.

    Upstream builds this on its JSON adapter and inherits the parse outright, so a failure is
    reported under that adapter's name and its own tests assert as much.
    """

    WIRE = "baml"
    ADAPTER_NAME = "JSONAdapter"

    def crossing_value(self, value: typing.Any) -> str:
        """A pydantic instance crosses keyed by its aliases, which is the form BAML renders."""
        if isinstance(value, pydantic.BaseModel):
            return value.model_dump_json(by_alias=True)
        return super().crossing_value(value)

    def format_field_structure(self, signature) -> str:
        """Upstream's tests read this section on its own, so it has to be the crate's.

        It is the only section any adapter states alone, and the only one that can refuse a
        signature — the crate raises on a model that reaches itself, exactly where dspy does.
        """
        global CROSSINGS
        CROSSINGS += 1
        return dsrs_bridge.baml_field_structure(
            signature.instructions,
            describe(signature.input_fields),
            described_outputs(signature),
        )
