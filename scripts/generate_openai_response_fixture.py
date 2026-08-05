"""Record what dspy 3.3's `completion_to_lm_response` makes of a raw chat-completion response.

The mirror of `generate_openai_wire_fixture.py`: that pins the request this crate *sends* to
`to_openai_chat_request`; this pins the `LMResponse` it *reads back* to `completion_to_lm_response`
(`dspy/clients/openai_format.py`). Each case is a raw OpenAI-shaped response and the `LMResponse`
dspy builds from it — reasoning content as a thinking part first, tool calls carrying the raw call as
provider data, usage with every counter aliased, the response id and finish reason kept. The OpenAI
provider's `reply` asserts it parses to the same value in `tests/lm_api_conformance.rs`, so a dropped
reasoning part or a mis-aliased count is a test failure rather than a difference a caller discovers.

    .dspy-venv-3.3/bin/python scripts/generate_openai_response_fixture.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import dspy

from pins import require
import dspy.core.types as t
from dspy.clients.openai_format import completion_to_lm_response

# Read from the pin rather than written here: a generator that names its own
# version cannot follow a bump, and six of them refused to run at 3.3.0 for
# exactly that reason while claiming the pin had drifted.
PINNED = require("dspy")
OUT = pathlib.Path(__file__).parent.parent / "crates" / "dsrust" / "tests" / "conformance" / "lm_api" / "openai_response.json"

# The request only supplies the model fallback when a response omits its own; every case here sets one.
REQUEST = t.LMRequest(model="openai/gpt-4o", messages=[t.LMMessage(role="user", parts=[t.LMTextPart(text="hi")])])


def message(**fields) -> dict:
    return {"role": "assistant", **fields}


def choice(msg: dict, finish_reason: str = "stop", **extra) -> dict:
    return {"message": msg, "finish_reason": finish_reason, **extra}


def response(choices: list, *, model: str = "openai/gpt-4o", rid: str = "chatcmpl-1", usage: dict | None = None, **extra) -> dict:
    body: dict = {"id": rid, "model": model, "choices": choices}
    if usage is not None:
        body["usage"] = usage
    body.update(extra)
    return body


USAGE = {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28}
TOOL_CALL = {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}

CASES = {
    "plain_text": response([choice(message(content="Paris"))], usage=USAGE),
    "reasoning_and_text": response(
        [choice(message(content="The answer is 4.", reasoning_content="2+2 = 4"))],
        model="openai/o3-mini", usage={"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16},
    ),
    "reasoning_only": response([choice(message(content=None, reasoning_content="thinking, no answer yet"))], usage=USAGE),
    "tool_call": response([choice(message(content=None, tool_calls=[TOOL_CALL]), finish_reason="tool_calls")], usage=USAGE),
    "text_and_tool_call": response(
        [choice(message(content="Let me check.", tool_calls=[TOOL_CALL]), finish_reason="tool_calls")], usage=USAGE),
    "length_truncated": response([choice(message(content="as far as it g"), finish_reason="length")], usage=USAGE),
    "several_completions": response(
        [choice(message(content="first")), choice(message(content="second")), choice(message(content="third"))],
        usage=USAGE,
    ),
    "logprobs": response([choice(message(content="hi"), logprobs={"content": [{"token": "hi", "logprob": -0.2}]})], usage=USAGE),
    "usage_with_details": response(
        [choice(message(content="hi"))],
        usage={"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140,
               "completion_tokens_details": {"reasoning_tokens": 12},
               "prompt_tokens_details": {"cached_tokens": 30}},
    ),
    "cache_hit": response([choice(message(content="cached"))], usage=USAGE, cache_hit=True),
    "citations": response([choice(message(
        content="Paris is the capital.",
        provider_specific_fields={"citations": [[{"cited_text": "Paris is the capital of France.",
                                                  "document_title": "France", "url": "https://example.com/fr"}]]},
    ))], usage=USAGE),
    "no_usage": response([choice(message(content="hi"))]),
    "model_fallback": {"id": "chatcmpl-9", "choices": [choice(message(content="hi"))], "usage": USAGE},
}


def main() -> None:
    if dspy.__version__ != PINNED:
        raise SystemExit(f"expected dspy {PINNED}, found {dspy.__version__}")
    cases = [
        {"name": name, "response": body, "lm_response": completion_to_lm_response(body, REQUEST).model_dump(mode="json")}
        for name, body in CASES.items()
    ]
    fixture = {
        "source": f"dspy=={PINNED} clients/openai_format.completion_to_lm_response",
        "dspy_version": PINNED,
        "cases": cases,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
    print(f"  wrote {OUT.name}: {len(cases)} cases", file=sys.stderr)


if __name__ == "__main__":
    main()
