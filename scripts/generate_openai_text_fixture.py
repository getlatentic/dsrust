"""Record what dspy 3.3's `to_openai_text_request` emits for a typed request.

The legacy completions wire, `model_type="text"`. dspy has two paths to it: `litellm_text_completion`
hands the work to litellm, and `to_openai_text_request` builds the body itself — the typed one, and
the one a port grounded on `openai_format` follows. They agree on the prompt and disagree on the
model, since only the litellm path re-prefixes it for that router.

The prompt rule is dspy's own and derivable from nothing: each message's text parts concatenated
with no separator, the messages joined by blank lines, and `BEGIN RESPONSE:` appended to the list —
so an empty conversation still sends the marker. A part that is not text *raises*: this endpoint
carries no images, and upstream refuses rather than dropping one silently.
"""

from __future__ import annotations

import json
import pathlib
import sys

import dspy.core.types as t
from dspy.clients.openai_format import to_openai_text_request

from pins import require

PINNED = require("dspy")
OUT = (
    pathlib.Path(__file__).parent.parent
    / "crates"
    / "dsrust"
    / "tests"
    / "conformance"
    / "lm_api"
    / "openai_text.json"
)


def text(value: str) -> t.LMTextPart:
    return t.LMTextPart(text=value)


def case(name: str, model: str, messages: list, **cfg) -> dict:
    request = t.LMRequest(
        model=model, messages=messages, tools=[], config=t.LMConfig.from_kwargs(**cfg)
    )
    return {
        "name": name,
        "lm_request": request.model_dump(mode="json"),
        "expected": to_openai_text_request(request),
    }


CASES = [
    case(
        "system_and_user",
        "gpt-3.5-turbo-instruct",
        [
            t.LMMessage(role="system", parts=[text("Answer the question.")]),
            t.LMMessage(role="user", parts=[text("What is the capital of France?")]),
        ],
    ),
    case(
        "single_user",
        "gpt-3.5-turbo-instruct",
        [t.LMMessage(role="user", parts=[text("Hello")])],
    ),
    case(
        "a_demo_conversation",
        "gpt-3.5-turbo-instruct",
        [
            t.LMMessage(role="system", parts=[text("Answer the question.")]),
            t.LMMessage(role="user", parts=[text("2 + 2?")]),
            t.LMMessage(role="assistant", parts=[text("4")]),
            t.LMMessage(role="user", parts=[text("3 + 3?")]),
        ],
    ),
    # The marker is appended to the *list*, so an empty conversation is the marker alone.
    case("no_messages", "gpt-3.5-turbo-instruct", []),
    # Two text parts in one message concatenate with nothing between them — not a space, and not
    # the blank line that separates messages.
    case(
        "two_parts_in_one_message",
        "gpt-3.5-turbo-instruct",
        [t.LMMessage(role="user", parts=[text("first"), text("second")])],
    ),
    case(
        "content_holding_blank_lines",
        "gpt-3.5-turbo-instruct",
        [t.LMMessage(role="user", parts=[text("first\n\nsecond")])],
    ),
    # The sampling fields this wire takes, in the order `text_config_kwargs` writes them:
    # extensions, then temperature, max_tokens, top_p, then stop, logprobs, n.
    case(
        "every_sampling_field",
        "gpt-3.5-turbo-instruct",
        [t.LMMessage(role="user", parts=[text("hi")])],
        temperature=0.7,
        max_tokens=256,
        top_p=0.9,
        stop=["\n\n"],
        logprobs=True,
        n=2,
    ),
    case(
        "max_tokens_only",
        "gpt-3.5-turbo-instruct",
        [t.LMMessage(role="user", parts=[text("hi")])],
        max_tokens=256,
    ),
    # An unknown keyword rides through ahead of every known one, as `dict(config.extensions)` does.
    case(
        "an_extension_rides_through",
        "gpt-3.5-turbo-instruct",
        [t.LMMessage(role="user", parts=[text("hi")])],
        temperature=0.5,
        best_of=3,
    ),
]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "source": f"dspy=={PINNED} clients/openai_format.to_openai_text_request",
                "dspy_version": PINNED,
                "note": (
                    "The legacy completions body dspy builds for a typed request. `prompt` is "
                    "`messages_to_text_prompt`, whose rule is dspy's own; the rest is "
                    "`text_config_kwargs`, whose key order is the order it writes them in."
                ),
                "cases": CASES,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    print(f"  wrote {OUT.name}: {len(CASES)} cases", file=sys.stderr)


if __name__ == "__main__":
    main()
