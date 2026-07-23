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
from dspy.adapters.two_step_adapter import TwoStepAdapter
from dspy.adapters.types.tool import ToolCalls
from dspy.adapters.xml_adapter import XMLAdapter
from dspy.adapters.utils import (
    format_field_value,
    parse_value,
    serialize_for_json,
)
from dspy.utils.exceptions import AdapterParseError, LMError
from litellm import ContextWindowExceededError

import dsrs_bridge
import crossings
from reflect import Unsupported, describe, described_outputs



def _serialized(value: typing.Any) -> typing.Any:
    """A value as JSON, keeping what dspy's own serializer drops but its renderer still reads.

    `ToolCalls.format` omits each call's id — the model is never shown it — and the model
    serializer is built on `format`, so a dumped value loses it. dspy still has the id, because it
    renders from the live object, and its native-tool replay needs it: a provider pairs a `tool`
    message to the call it answers by that id. The crossing therefore carries it, so the crate
    decides whether the results replay on the same information upstream decides on.
    """
    if isinstance(value, dspy.History):
        return {"messages": [{k: _serialized(v) for k, v in m.items()} for m in value.messages]}
    if isinstance(value, ToolCalls):
        data = {
            "tool_calls": [
                {"id": call.id, "name": call.name, "args": call.args} for call in value.tool_calls
            ]
        }
        if value.tool_call_results is not None:
            data["tool_call_results"] = serialize_for_json(value.tool_call_results)
        return data
    return serialize_for_json(value)


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
        return json.dumps(_serialized(value), ensure_ascii=False)

    def format_system_message(self, signature) -> str:
        """dspy exposes this on its own, and a caller reading it should read the crate's."""
        crossings.record_render()
        return dsrs_bridge.format_system_message(
            self.WIRE,
            signature.instructions,
            describe(signature.input_fields),
            described_outputs(signature),
        )

    def format_field_description(self, signature) -> str:
        """Upstream's tests read this section on its own, so it has to be the crate's."""
        crossings.record_render()
        return dsrs_bridge.field_description(
            signature.instructions,
            describe(signature.input_fields),
            described_outputs(signature),
        )

    def _custom_history(self, signature, inputs):
        """A subclass's own conversation history, or None where it did not override the hook.

        dspy lets a caller replace just the history half of a request, and a subclass that does so
        expects its messages used verbatim. The hook is Python's, so honouring it is this side's
        job; deleting the history input is what upstream's own hook does, and it leaves the crate
        rendering everything else — demos, the live request, the system message — as before.
        """
        history_field = self._get_history_field_name(signature)
        if not history_field:
            return None
        if type(self).format_conversation_history is Adapter.format_conversation_history:
            return None
        return self.format_conversation_history(
            signature.delete(history_field), history_field, inputs
        )

    def format(self, signature, demos, inputs) -> list[dict[str, typing.Any]]:
        crossings.record_render()
        inputs = dict(inputs)
        custom_history = self._custom_history(signature, inputs)
        rendered_demos = [
            [
                (name, format_field_value(field_info=field, value=demo[name]))
                for name, field in {**signature.input_fields, **signature.output_fields}.items()
                if name in demo
            ]
            for demo in demos
        ]
        # The third element is what dspy branches on when it lays a value out:
        # `isinstance(value, BaseModel)`. It cannot survive as JSON — a dumped model and a
        # hand-written mapping are the same text — so it crosses beside the value.
        values = [
            (name, self.crossing_value(value), isinstance(value, pydantic.BaseModel))
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
            # dspy's base Adapter carries this, and it decides how a tool-calling exchange
            # replays; the crate's adapter is what acts on it.
            getattr(self, "use_native_function_calling", False),
        )
        # Each turn crosses as the whole message the crate rendered, since a tool result and an
        # assistant turn carrying tool calls have keys beside `content`.
        messages = [{"role": "system", "content": system}] + [json.loads(turn) for turn in turns]
        if custom_history is not None:
            # dspy orders these demos, history, then the live request, and the crate's last turn
            # is that request.
            messages[-1:-1] = custom_history
        return messages

    def parse(self, signature, completion):
        crossings.record_render()
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
        # Both the decision and what it propagates are the crate's: dspy carries the
        # native-function-calling settings into the fallback so a re-ask asks the provider the
        # same way the first attempt did, and the crate's ChatAdapter is what states that.
        settings = dsrs_bridge.json_fallback_settings(
            "chat",
            self.use_json_adapter_fallback,
            self.use_native_function_calling,
            self.parallel_tool_calls,
        )
        if settings is None:
            return None
        use_native_function_calling, parallel_tool_calls = settings
        return JSONAdapter(
            use_native_function_calling=use_native_function_calling,
            parallel_tool_calls=parallel_tool_calls,
        )

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
        crossings.record_render()
        return dsrs_bridge.baml_field_structure(
            signature.instructions,
            describe(signature.input_fields),
            described_outputs(signature),
        )

class RustTwoStepAdapter(_RustBacked, TwoStepAdapter):
    """Prose first, then a second model naming the fields — both rendered by the crate.

    The crate says what the first ask reads like and what the second one asks for; the second
    model call stays here, because Python holds the models on this side of the bridge. That is
    the same split the crate makes internally, where its `Predict` runs the extraction and the
    adapter only describes it.
    """

    WIRE = "two_step"
    ADAPTER_NAME = "TwoStepAdapter"

    def _extractor_signature(self, signature):
        """`text` in, the original outputs out, asked for in the crate's words.

        The fields keep their Python annotations, which only this side can build; the
        instruction is the crate's, so it comes across rather than being written twice.
        """
        fields = {
            "text": (str, dspy.InputField()),
            **{name: (field.annotation, field) for name, field in signature.output_fields.items()},
        }
        instructions = dsrs_bridge.extractor_instructions(described_outputs(signature))
        return dspy.signatures.signature.make_signature(fields, instructions)

    def parse(self, signature, completion):
        try:
            extracted = RustChatAdapter()(
                lm=self.extraction_model,
                lm_kwargs={},
                signature=self._extractor_signature(signature),
                demos=[],
                inputs={"text": completion},
            )
            return extracted[0]
        except Unsupported:
            raise
        except LMError:
            # dspy 3.3 lets an LM failure through rather than reporting it as a parse failure.
            raise
        except Exception as error:
            raise AdapterParseError(
                adapter_name="TwoStepAdapter",
                signature=signature,
                lm_response=completion,
                message=f"Failed to parse response from the original completion: {error}",
            ) from error
