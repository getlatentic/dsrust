"""Record the exact request body litellm puts on the wire for Anthropic and ollama.

dspy 3.3 ships native wire code only for OpenAI (`clients/openai_format.py`); every other provider
is routed through litellm, so litellm's body *is* what dspy sends to Anthropic and ollama. This
captures that body the only way that cannot drift from litellm's real behaviour: it mocks litellm's
HTTP layer and reads what `litellm.completion` was about to POST.

Each case follows dspy's own path for a typed request to a litellm provider — `LMRequest` through
`to_openai_chat_request` (the verified typed->messages converter, the same first hop the OpenAI wire
fixture uses) into `litellm.completion` — and records the typed request (as the Rust `api::LmRequest`
parses it) beside litellm's captured body. The Anthropic and ollama request builders assert byte
equality against it in `tests/lm_api_conformance.rs`, so a divergence — a bare string where litellm
sends a text block, a defaulted `max_tokens`, a `format` string where litellm sends the schema — is a
test failure rather than a difference discovered on a real call.

    .dspy-venv-3.3/bin/python scripts/generate_litellm_wire_fixture.py
"""

from __future__ import annotations

import json
import pathlib
import sys
from unittest import mock

import dspy
import dspy.core.types as t
import litellm
from dspy.clients.openai_format import to_openai_chat_request
from litellm.llms.custom_httpx.http_handler import HTTPHandler

PINNED = "3.3.0b1"
OUT = pathlib.Path(__file__).parent.parent / "tests" / "conformance" / "lm_api" / "litellm_chat.json"

# The litellm route for each dsrs provider. dsrs mirrors litellm's own split: `ollama` here is
# the `/api/chat` route (litellm's `ollama_chat`), and `ollama_generate` is the `/api/generate`
# route (litellm's bare `ollama`).
LITELLM_ROUTE = {
    "anthropic": "anthropic",
    "ollama": "ollama_chat",
    "ollama_generate": "ollama",
}
# What each provider needs to get past litellm's credential/host checks and reach the transform.
REACH = {
    "anthropic": {"api_key": "sk-ant-x"},
    "ollama": {"api_base": "http://localhost:11434"},
    "ollama_generate": {"api_base": "http://localhost:11434"},
}

_captured: dict = {}


def _capture_post(self, *args, **kwargs):  # noqa: ANN001
    body = kwargs.get("json")
    if body is None and kwargs.get("data") is not None:
        try:
            body = json.loads(kwargs["data"])
        except (TypeError, ValueError):
            body = kwargs["data"]
    _captured["body"] = body
    raise RuntimeError("__captured__")


def _litellm_body(openai_call: dict, reach: dict) -> dict:
    _captured.clear()
    with mock.patch.object(HTTPHandler, "post", _capture_post):
        try:
            litellm.completion(**openai_call, **reach)
        except Exception as error:
            # litellm wraps the sentinel raised from the mocked POST in its own provider exception,
            # so the class is not RuntimeError by the time it surfaces — the marker in the message is.
            if "__captured__" not in str(error):
                raise
    return _captured["body"]


def text(value: str) -> t.LMTextPart:
    return t.LMTextPart(text=value)


def case(name: str, provider: str, model_id: str, messages: list, tools=None, config=None, **cfg) -> dict:
    request = t.LMRequest(
        model=f"{LITELLM_ROUTE[provider]}/{model_id}",
        messages=messages,
        tools=tools or [],
        config=config if config is not None else t.LMConfig.from_kwargs(**cfg),
    )
    body = _litellm_body(to_openai_chat_request(request), REACH[provider])
    return {
        "name": name,
        "provider": provider,
        "lm_request": request.model_dump(mode="json"),
        "expected": body,
    }


SYS = t.LMMessage(role="system", parts=[text("be helpful")])
ASK = t.LMMessage(role="user", parts=[text("weather in Paris?")])
HI = [t.LMMessage(role="user", parts=[text("hi")])]
WEATHER = t.LMToolSpec(
    name="get_weather",
    description="look up the weather",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}},
)
SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}

# A multi-turn tool conversation: user asks, assistant calls the tool, the tool answers.
CONVERSATION = [
    t.LMMessage(role="user", parts=[text("weather in Paris?")]),
    t.LMMessage(role="assistant", parts=[t.LMToolCallPart(id="call_1", name="get_weather", args={"city": "Paris"})]),
    t.LMMessage(role="tool", parts=[t.LMToolResultPart(call_id="call_1", name="get_weather", content=[text("sunny, 22C")])]),
]


def structured_case(name: str, provider: str, model_id: str) -> dict:
    """A json-mode case. dspy's `response_format` is the whole `{"type": "json_schema", ...}`
    envelope; this crate stores the bare schema and builds each provider's form itself. litellm is
    driven with the envelope (its real `json_tool_call` / `format` output is the expectation), and
    the dumped request's `response_format` is rewritten to the bare schema the Rust type parses — so
    the case stays litellm's own output, not a hand-written one, while matching this crate's shape."""
    envelope = {"type": "json_schema", "json_schema": {"name": "resp", "schema": SCHEMA}}
    request = t.LMRequest(
        model=f"{LITELLM_ROUTE[provider]}/{model_id}",
        messages=list(HI),
        config=t.LMConfig(response_format=envelope),
    )
    body = _litellm_body(to_openai_chat_request(request), REACH[provider])
    dumped = request.model_dump(mode="json")
    dumped["config"]["response_format"] = SCHEMA
    return {"name": name, "provider": provider, "lm_request": dumped, "expected": body}


def anthropic_cases() -> list:
    p = "anthropic"
    m = "claude-3-5-sonnet-20241022"
    return [
        case("minimal", p, m, [SYS, ASK]),
        case("temperature", p, m, HI, temperature=0.7),
        case("top_p_and_stop", p, m, HI, top_p=0.9, stop=["\n\n", "END"]),
        case("max_tokens_overrides_the_default", p, m, HI, max_tokens=256),
        case("image_url", p, m, [t.LMMessage(role="user", parts=[
            text("describe"), t.LMImagePart(url="https://example.com/a.jpg")])]),
        case("image_base64", p, m, [t.LMMessage(role="user", parts=[
            t.LMImagePart(url="data:image/png;base64,iVBORw0KGgo=")])]),
        case("document", p, m, [t.LMMessage(role="user", parts=[text("summarize"),
            t.LMDocumentPart(source={"type": "base64", "media_type": "application/pdf", "data": "JVBERi0="}, title="Contract")])]),
        case("tools", p, m, [ASK], tools=[WEATHER]),
        case("tool_choice_required", p, m, [ASK], tools=[WEATHER],
             config=t.LMConfig(tool_choice=t.LMToolChoice(mode="required"))),
        case("tool_choice_auto", p, m, [ASK], tools=[WEATHER],
             config=t.LMConfig(tool_choice=t.LMToolChoice(mode="auto"))),
        case("tool_choice_none", p, m, [ASK], tools=[WEATHER],
             config=t.LMConfig(tool_choice=t.LMToolChoice(mode="none"))),
        case("tool_choice_single_named", p, m, [ASK], tools=[WEATHER],
             config=t.LMConfig(tool_choice=t.LMToolChoice(mode="required", allowed=["get_weather"]))),
        case("tool_conversation", p, m, list(CONVERSATION), tools=[WEATHER]),
        structured_case("structured_output", p, m),
    ]


def ollama_cases() -> list:
    p = "ollama"
    m = "llama3.2"
    return [
        case("minimal", p, m, [SYS, ASK]),
        case("temperature", p, m, HI, temperature=0.7),
        case("stop_and_max_tokens", p, m, HI, temperature=0.1, stop=["\n\n"], max_tokens=64),
        case("image_base64", p, m, [t.LMMessage(role="user", parts=[
            text("describe"), t.LMImagePart(url="data:image/png;base64,iVBORw0KGgo=")])]),
        case("tools", p, m, [ASK], tools=[WEATHER]),
        case("tool_conversation", p, m, list(CONVERSATION), tools=[WEATHER]),
        structured_case("structured_output", p, m),
    ]


def ollama_generate_cases() -> list:
    """The `/api/generate` route: one flattened prompt rather than a message list. dsrs never
    sends native tools here — the route cannot carry them — so these are the conversations the
    adapter produces, and the assertion is that `ollama_pt`'s flattening is reproduced byte for
    byte. A tool conversation is included to pin the `Tool Calls:` append litellm does when an
    assistant turn carries calls."""
    p = "ollama_generate"
    m = "llama3.2"
    return [
        case("minimal", p, m, [SYS, ASK]),
        case("temperature", p, m, HI, temperature=0.7),
        case("stop_and_max_tokens", p, m, HI, temperature=0.1, stop=["\n\n"], max_tokens=64),
        case("multi_turn", p, m, [
            ASK,
            t.LMMessage(role="assistant", parts=[text("Let me check.")]),
            t.LMMessage(role="user", parts=[text("thanks")]),
        ]),
        case("image_base64", p, m, [t.LMMessage(role="user", parts=[
            text("describe"), t.LMImagePart(url="data:image/png;base64,iVBORw0KGgo=")])]),
        case("tool_conversation", p, m, list(CONVERSATION)),
        structured_case("structured_output", p, m),
    ]


def main() -> None:
    if dspy.__version__ != PINNED:
        raise SystemExit(f"expected dspy {PINNED}, found {dspy.__version__}")
    cases = anthropic_cases() + ollama_cases() + ollama_generate_cases()
    fixture = {
        "source": f"dspy=={PINNED} to_openai_chat_request -> litellm.completion (HTTP body captured)",
        "dspy_version": PINNED,
        "litellm_version": __import__("importlib.metadata", fromlist=["version"]).version("litellm"),
        "cases": cases,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
    print(f"  wrote {OUT.name}: {len(cases)} cases", file=sys.stderr)


if __name__ == "__main__":
    main()
