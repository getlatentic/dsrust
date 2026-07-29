"""Record what dspy 3.3's `to_openai_chat_request` emits for a typed request.

A faithful `typed_lm` LM converts an `LMRequest` to its provider's body; dspy 3.3 ships the
canonical OpenAI conversion in `clients/openai_format.py`. This dumps, per case, the typed request
(as pydantic serializes it, which the Rust `api::LmRequest` parses) and the OpenAI body dspy
builds from it. `tests/openai_wire.rs`-style checks in `src/lm/openai.rs` assert our body equals
dspy's, so a divergence in the config mapping (a defaulted `max_tokens`, a dropped `top_p`) is a
test failure rather than a silent difference on the wire.

    .dspy-venv-3.3/bin/python scripts/generate_openai_wire_fixture.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import dspy
import dspy.core.types as t
from dspy.clients.openai_format import to_openai_chat_request

PINNED = "3.3.0b1"
OUT = pathlib.Path(__file__).parent.parent / "crates" / "dsrust" / "tests" / "conformance" / "lm_api" / "openai_chat.json"


def text(value: str) -> t.LMTextPart:
    return t.LMTextPart(text=value)


def case(name: str, model: str, messages: list, tools=None, config=None, **cfg) -> dict:
    request = t.LMRequest(
        model=model,
        messages=messages,
        tools=tools or [],
        config=config if config is not None else t.LMConfig.from_kwargs(**cfg),
    )
    return {
        "name": name,
        "lm_request": request.model_dump(mode="json"),
        "expected": to_openai_chat_request(request),
    }


# Deliberately no `response_format` case: dspy carries the whole envelope in `config.response_format`
# while this crate stores the bare schema and builds the envelope in the provider, a representation
# difference the JSON-mode wire tests cover separately.
CASES = [
    case("minimal", "openai/gpt-4o-mini",
         [t.LMMessage(role="system", parts=[text("be helpful")]),
          t.LMMessage(role="user", parts=[text("hi")])]),
    case("temperature_and_n", "openai/gpt-4o-mini",
         [t.LMMessage(role="user", parts=[text("hi")])], temperature=0.7, n=2),
    case("max_tokens_set", "openai/gpt-4o-mini",
         [t.LMMessage(role="user", parts=[text("hi")])], max_tokens=256),
    case("reasoning_max_completion_tokens", "openai/o3-mini",
         [t.LMMessage(role="user", parts=[text("hi")])], max_tokens=16000),
    case("top_p_and_stop", "openai/gpt-4o-mini",
         [t.LMMessage(role="user", parts=[text("hi")])], top_p=0.9, stop=["\n\n"]),
    case("logprobs", "openai/gpt-4o-mini",
         [t.LMMessage(role="user", parts=[text("hi")])], logprobs=True),
    case("extensions_passthrough", "openai/gpt-4o-mini",
         [t.LMMessage(role="user", parts=[text("hi")])], seed=42, user="acct-1"),
    case("multimodal_image", "openai/gpt-4o-mini",
         [t.LMMessage(role="user", parts=[text("describe"),
                                          t.LMImagePart(url="https://example.com/a.jpg")])]),
    case("multimodal_audio", "openai/gpt-4o-mini",
         [t.LMMessage(role="user", parts=[text("transcribe"), t.LMAudioPart(data="YQ==", media_type="audio/wav")])]),
    case("multimodal_video", "openai/gpt-4o-mini",
         [t.LMMessage(role="user", parts=[text("describe"), t.LMVideoPart(url="https://example.com/a.mp4")])]),
    case("multimodal_binary_file_id", "openai/gpt-4o-mini",
         [t.LMMessage(role="user", parts=[text("summarize"), t.LMBinaryPart(file_id="file_1", filename="a.pdf")])]),
    case("multimodal_binary_data", "openai/gpt-4o-mini",
         [t.LMMessage(role="user", parts=[text("read"), t.LMBinaryPart(data="JVBERi0=", media_type="application/pdf", filename="doc.pdf")])]),
    case("multimodal_document", "openai/gpt-4o-mini",
         [t.LMMessage(role="user", parts=[text("analyze"), t.LMDocumentPart(source={"type": "text", "data": "the contract"}, title="Contract")])]),
    case("tools", "openai/gpt-4o",
         [t.LMMessage(role="user", parts=[text("weather in Paris?")])],
         tools=[t.LMToolSpec(name="get_weather", description="look up the weather",
                             parameters={"type": "object", "properties": {"city": {"type": "string"}}})]),
    case("tool_choice_required_single", "openai/gpt-4o",
         [t.LMMessage(role="user", parts=[text("weather?")])],
         tools=[t.LMToolSpec(name="get_weather", parameters={"type": "object"})],
         config=t.LMConfig(tool_choice=t.LMToolChoice(mode="required", allowed=["get_weather"]))),
    case("tool_choice_auto_parallel", "openai/gpt-4o",
         [t.LMMessage(role="user", parts=[text("weather?")])],
         tools=[t.LMToolSpec(name="get_weather", parameters={"type": "object"})],
         config=t.LMConfig(tool_choice=t.LMToolChoice(mode="auto", parallel=False))),
    case("reasoning_effort", "openai/o3-mini",
         [t.LMMessage(role="user", parts=[text("hi")])],
         config=t.LMConfig.from_kwargs(reasoning_effort="high")),
    case("prompt_cache_key", "openai/gpt-4o-mini",
         [t.LMMessage(role="user", parts=[text("hi")])],
         config=t.LMConfig.from_kwargs(prompt_cache_key="k1")),
    # A multi-turn tool conversation: the assistant's tool call splits into `tool_calls` beside a
    # null content, and the tool result is its own `{content, tool_call_id, name}` message.
    case("tool_conversation", "openai/gpt-4o",
         [t.LMMessage(role="user", parts=[text("weather in Paris?")]),
          t.LMMessage(role="assistant", parts=[t.LMToolCallPart(id="call_1", name="get_weather", args={"city": "Paris"})]),
          t.LMMessage(role="tool", parts=[t.LMToolResultPart(call_id="call_1", name="get_weather", content=[text("sunny, 22C")])])],
         tools=[t.LMToolSpec(name="get_weather", parameters={"type": "object", "properties": {"city": {"type": "string"}}})]),
]


def main() -> None:
    if dspy.__version__ != PINNED:
        raise SystemExit(f"expected dspy {PINNED}, found {dspy.__version__}")
    fixture = {
        "source": f"dspy=={PINNED} clients/openai_format.to_openai_chat_request",
        "dspy_version": PINNED,
        "cases": CASES,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
    print(f"  wrote {OUT.name}: {len(CASES)} cases", file=sys.stderr)


if __name__ == "__main__":
    main()
