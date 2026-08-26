#!/usr/bin/env python
"""Record the request dspy builds for `model_type="text"` — the legacy completions wire.

`LM(model_type="text")` routes to `litellm_text_completion`, which is the one place dspy turns a
message list back into a single prompt string. Two rules are dspy's own and neither is guessable:
the prompt is every message's `content` joined by a blank line with `BEGIN RESPONSE:` appended, and
the model is re-prefixed `text-completion-openai/` whatever provider it named.

Captured rather than reimplemented: `litellm.text_completion` is replaced with a recorder, so what
lands here is the keyword arguments dspy passed, not a reading of the source. The wire underneath
is litellm's, and what it does with those arguments is not dspy's to define.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import dspy
from dspy.clients import lm as lm_module

PINNED = "3.3.0b1"
OUT = Path(__file__).resolve().parent.parent / "crates" / "dsrust" / "tests" / "conformance" / "lm"

#: One entry per shape a rendered prompt arrives in: the adapters' system+user pair, a single turn,
#: a demo conversation, and the empty list that a caller can still reach.
CASES: list[tuple[str, list[dict[str, Any]], dict[str, Any]]] = [
    (
        "system_and_user",
        [
            {"role": "system", "content": "Answer the question."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
        {},
    ),
    ("single_user", [{"role": "user", "content": "Hello"}], {}),
    (
        "a_demo_conversation",
        [
            {"role": "system", "content": "Answer the question."},
            {"role": "user", "content": "2 + 2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "3 + 3?"},
        ],
        {},
    ),
    ("no_messages", [], {}),
    (
        "content_holding_blank_lines",
        [{"role": "user", "content": "first\n\nsecond"}],
        {},
    ),
    (
        "with_sampling",
        [{"role": "user", "content": "Hello"}],
        {"temperature": 0.7, "max_tokens": 100},
    ),
]


class Recorder:
    """Stands in for the litellm module, keeping what `text_completion` was called with."""

    def __init__(self) -> None:
        self.seen: dict[str, Any] | None = None

    def text_completion(self, **kwargs: Any) -> Any:
        self.seen = kwargs
        # dspy reads `.choices[0].text` off the result; nothing here does, so a bare object is
        # enough to let the call return.
        return {"choices": [{"text": ""}]}


def recorded(model: str, messages: list[dict[str, Any]], extra: dict[str, Any]) -> dict:
    recorder = Recorder()
    original = lm_module._get_litellm
    lm_module._get_litellm = lambda: recorder
    try:
        lm_module.litellm_text_completion(
            {"model": model, "messages": messages, **extra}, num_retries=0
        )
    finally:
        lm_module._get_litellm = original
    seen = dict(recorder.seen or {})
    # `headers` carries a dspy version string, and `cache` is litellm's own control block; neither
    # is part of what a port has to reproduce.
    for noise in ("headers", "cache", "num_retries", "retry_strategy", "api_key", "api_base"):
        seen.pop(noise, None)
    return seen


def main() -> None:
    fixture = {
        "source": f"generated from dspy=={PINNED} via scripts/generate_text_completion_fixture.py",
        "dspy_version": PINNED,
        "note": (
            "What `litellm_text_completion` passes to litellm for each rendered message list. "
            "`prompt` and `model` are dspy's own; the rest rides through from the request."
        ),
        "cases": [
            {
                "name": name,
                "model": "openai/gpt-3.5-turbo-instruct",
                "messages": messages,
                "extra": extra,
                "sent": recorded("openai/gpt-3.5-turbo-instruct", messages, extra),
            }
            for name, messages, extra in CASES
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "text_completion.json"
    path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
    print(f"  wrote {path.relative_to(OUT.parent.parent.parent.parent)}", file=sys.stderr)


if __name__ == "__main__":
    main()
