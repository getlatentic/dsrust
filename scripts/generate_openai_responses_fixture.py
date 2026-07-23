"""Record what dspy 3.3's `to_openai_responses_request` emits for a typed request.

The Responses API is OpenAI's second wire, used for reasoning models: a flat `input` list of items
rather than `messages`, `input_text`/`input_image` content, `function_call`/`function_call_output`
items for a tool exchange, `max_output_tokens`, and `reasoning: {effort, summary}`. This pins the
body dspy builds (`dspy/clients/openai_format.py`) so the Rust Responses request builder can assert
byte equality in `tests/lm_api_conformance.rs`.

No `response_format` case, same as the chat wire fixture: dspy carries the whole `text.format`
envelope while this crate stores the bare schema and builds the envelope in the provider.

    .dspy-venv-3.3/bin/python scripts/generate_openai_responses_fixture.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import dspy
import dspy.core.types as t
from dspy.clients.openai_format import responses_to_lm_response, to_openai_responses_request

PINNED = "3.3.0b1"
OUT = pathlib.Path(__file__).parent.parent / "tests" / "conformance" / "lm_api" / "openai_responses.json"


def text(value: str) -> t.LMTextPart:
    return t.LMTextPart(text=value)


def case(name: str, model: str, messages: list, tools=None, config=None, **cfg) -> dict:
    request = t.LMRequest(
        model=model,
        messages=messages,
        tools=tools or [],
        config=config if config is not None else t.LMConfig.from_kwargs(**cfg),
    )
    return {"name": name, "lm_request": request.model_dump(mode="json"), "expected": to_openai_responses_request(request)}


WEATHER = t.LMToolSpec(name="get_weather", parameters={"type": "object", "properties": {"city": {"type": "string"}}})
CONVERSATION = [
    t.LMMessage(role="user", parts=[text("weather in Paris?")]),
    t.LMMessage(role="assistant", parts=[t.LMToolCallPart(id="call_1", name="get_weather", args={"city": "Paris"})]),
    t.LMMessage(role="tool", parts=[t.LMToolResultPart(call_id="call_1", name="get_weather", content=[text("sunny, 22C")])]),
]

CASES = [
    case("minimal", "openai/gpt-5",
         [t.LMMessage(role="system", parts=[text("be helpful")]), t.LMMessage(role="user", parts=[text("hi")])]),
    case("temperature_and_top_p", "openai/gpt-4o",
         [t.LMMessage(role="user", parts=[text("hi")])], temperature=0.7, top_p=0.9),
    case("max_output_tokens", "openai/gpt-4o", [t.LMMessage(role="user", parts=[text("hi")])], max_tokens=16000),
    case("stop_and_n_and_logprobs", "openai/gpt-4o",
         [t.LMMessage(role="user", parts=[text("hi")])], stop=["\n\n"], n=2, logprobs=5),
    case("extensions_passthrough", "openai/gpt-4o",
         [t.LMMessage(role="user", parts=[text("hi")])], seed=42, user="acct-1"),
    case("reasoning", "openai/o3", [t.LMMessage(role="user", parts=[text("hi")])],
         config=t.LMConfig(reasoning=t.LMReasoningConfig(effort="high", summary="concise", max_tokens=512))),
    case("prompt_cache_key", "openai/gpt-4o",
         [t.LMMessage(role="user", parts=[text("hi")])], config=t.LMConfig.from_kwargs(prompt_cache_key="k1")),
    case("image", "openai/gpt-5",
         [t.LMMessage(role="user", parts=[text("describe"), t.LMImagePart(url="https://example.com/a.jpg", detail="high")])]),
    case("tools", "openai/gpt-5", [t.LMMessage(role="user", parts=[text("weather?")])], tools=[WEATHER]),
    case("tool_choice_required", "openai/gpt-5", [t.LMMessage(role="user", parts=[text("weather?")])],
         tools=[WEATHER], config=t.LMConfig(tool_choice=t.LMToolChoice(mode="required"))),
    case("tool_choice_single_named", "openai/gpt-5", [t.LMMessage(role="user", parts=[text("weather?")])],
         tools=[WEATHER], config=t.LMConfig(tool_choice=t.LMToolChoice(mode="required", allowed=["get_weather"]))),
    case("tool_choice_parallel_off", "openai/gpt-5", [t.LMMessage(role="user", parts=[text("weather?")])],
         tools=[WEATHER], config=t.LMConfig(tool_choice=t.LMToolChoice(mode="auto", parallel=False))),
    case("tool_conversation", "openai/gpt-5", list(CONVERSATION), tools=[WEATHER]),
]

# The reply direction: a raw Responses object and the LMResponse dspy reads from it. One output
# holds every item as a part — reasoning summary as thinking, message content as text, a
# function_call as a tool call keeping its raw item, a refusal as its own part.
REQUEST = t.LMRequest(model="openai/gpt-5", messages=[t.LMMessage(role="user", parts=[text("hi")])])


def reply_case(name: str, output: list, *, model: str = "gpt-5", rid: str = "resp_1", usage: dict | None = None, **extra) -> dict:
    response = {"id": rid, "model": model, "output": output}
    if usage is not None:
        response["usage"] = usage
    response.update(extra)
    return {"name": name, "response": response, "lm_response": responses_to_lm_response(response, REQUEST).model_dump(mode="json")}


USAGE = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
MSG = lambda *content: {"type": "message", "role": "assistant", "content": list(content)}
OUT_TEXT = lambda s: {"type": "output_text", "text": s, "annotations": []}
FCALL = {"type": "function_call", "name": "get_weather", "arguments": '{"city": "Paris"}', "call_id": "call_1"}

REPLY_CASES = [
    reply_case("text_message", [MSG(OUT_TEXT("It is sunny."))], usage=USAGE),
    reply_case("reasoning_summary_and_message",
               [{"type": "reasoning", "summary": [{"type": "summary_text", "text": "Let me think."}]}, MSG(OUT_TEXT("Sunny."))], usage=USAGE),
    reply_case("reasoning_content",
               [{"type": "reasoning", "content": [{"type": "reasoning_text", "text": "Working it out."}]}, MSG(OUT_TEXT("42."))], usage=USAGE),
    reply_case("function_call", [FCALL], usage=USAGE),
    reply_case("text_and_function_call", [MSG(OUT_TEXT("Let me check.")), FCALL], usage=USAGE),
    reply_case("refusal", [MSG({"type": "refusal", "refusal": "I can't help with that."})], usage=USAGE),
    reply_case("citations_from_annotations", [MSG({"type": "output_text", "text": "Paris is the capital.",
              "annotations": [{"type": "url_citation", "url": "https://example.com/fr", "title": "France"}]})], usage=USAGE),
    reply_case("output_image_b64", [{"type": "image", "b64_json": "iVBORw0KGgo=", "media_type": "image/png"}], usage=USAGE),
    reply_case("output_image_url", [{"type": "image", "image_url": {"url": "https://example.com/gen.png"}}], usage=USAGE),
    reply_case("output_image_file_id", [{"type": "image_generation_call", "file_id": "file_img_1"}], usage=USAGE),
    reply_case("output_audio", [{"type": "output_audio", "data": "YQ==", "media_type": "audio/wav"}], usage=USAGE),
    reply_case("output_file_data_uri", [{"type": "file", "file_data": "data:application/pdf;base64,JVBERi0=", "filename": "o.pdf"}], usage=USAGE),
    reply_case("output_file_url", [{"type": "file", "file": {"url": "https://example.com/o.pdf", "media_type": "application/pdf"}}], usage=USAGE),
    reply_case("message_with_generated_image",
               [MSG(OUT_TEXT("here it is"), {"type": "output_image", "b64_json": "iVBORw0=", "media_type": "image/png"})], usage=USAGE),
    reply_case("cache_hit", [MSG(OUT_TEXT("cached"))], usage=USAGE, cache_hit=True),
    reply_case("no_usage", [MSG(OUT_TEXT("hi"))]),
    reply_case("model_fallback", [MSG(OUT_TEXT("hi"))], model=None, rid="resp_9"),
]


def main() -> None:
    if dspy.__version__ != PINNED:
        raise SystemExit(f"expected dspy {PINNED}, found {dspy.__version__}")
    fixture = {
        "source": f"dspy=={PINNED} clients/openai_format to_openai_responses_request + responses_to_lm_response",
        "dspy_version": PINNED,
        "request_cases": CASES,
        "reply_cases": REPLY_CASES,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
    print(f"  wrote {OUT.name}: {len(CASES)} request + {len(REPLY_CASES)} reply cases", file=sys.stderr)


if __name__ == "__main__":
    main()
