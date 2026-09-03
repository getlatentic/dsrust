"""Which OpenAI model names dspy treats as reasoning models — and where the answer differs.

dspy has **two** predicates for this question and they do not agree:

  - `clients/openai_format.py::_is_openai_reasoning_model` strips an `openai/` prefix, refuses any
    name containing `chat`, and asks whether the rest starts with `o1`, `o3`, `o4` or `gpt-5`. It
    decides the chat body's token key and whether a reasoning temperature is refused.
  - `clients/lm.py::_is_openai_reasoning_model` takes the last `/`-separated segment and matches a
    regex — `o[1345]` with an optional `-mini`/`-nano`/`-pro` and an optional date, or `gpt-5`
    without a `-chat` suffix. It decides what `LM.dump_state` writes.

So `o1-preview` sends `max_completion_tokens` and saves as a plain model, `o5` does the reverse,
and `gpt-5.1-chat` is a reasoning model to neither. A port with one predicate is wrong somewhere
whichever rule it picks, and a fixture sampling only `gpt-4o-mini` and `o3-mini` — which is what
`lm_api/openai_chat.json` covers — cannot see it, because those two are in the region where the two
rules agree.

What is recorded is the consequence rather than the predicate: the token key in the body
`to_openai_chat_request` builds, and the ordered keys of `LM.dump_state`, whose `max_tokens`
position moves when the state rule fires. Both are bytes.

    .venv/bin/python scripts/generate_reasoning_model_fixture.py
"""

from __future__ import annotations

import json
import logging
import pathlib
import sys
import warnings

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

import dspy
import dspy.core.types as t
from dspy.clients.openai_format import (
    _validate_openai_reasoning_temperature,
    to_openai_chat_request,
)

from pins import require

PINNED = require("dspy")
OUT = (
    pathlib.Path(__file__).parent.parent
    / "crates"
    / "dsrust"
    / "tests"
    / "conformance"
    / "lm_api"
    / "reasoning_models.json"
)

#: Chosen to straddle both rules rather than to sample the families. Every name on which the two
#: dspy predicates disagree is here — `o1-preview`, `o5`, `gpt-5.1`, `azure/o3`, a doubly-prefixed
#: id — beside the ones they agree on, so a re-implementation cannot pass by getting the easy
#: region right.
MODELS = [
    # Both rules agree: reasoning.
    "openai/o1",
    "openai/o1-mini",
    "openai/o1-pro",
    "openai/o3",
    "openai/o3-mini",
    "openai/o3-mini-2025-01-31",
    "openai/o4-mini",
    "openai/gpt-5",
    "openai/gpt-5-mini",
    "openai/gpt-5-nano",
    "openai/gpt-5-codex",
    "o3",
    "gpt-5",
    # Both rules agree: not reasoning.
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "openai/gpt-4.1",
    "openai/gpt-3.5-turbo",
    "openai/chatgpt-4o-latest",
    "openai/gpt-5-chat",
    "openai/gpt-5-chat-latest",
    "openai/o2",
    "openai/omni-moderation-latest",
    "openai/o200k-base",
    "anthropic/claude-sonnet-4-20250514",
    # The wire rule fires and the state rule does not: a `-preview` suffix the regex has no arm
    # for, and a minor version the regex anchors past.
    "openai/o1-preview",
    "openai/gpt-5.1",
    # The state rule fires and the wire rule does not: `o[1345]` covers `o5` where the prefix
    # tuple does not, and the wire rule strips only an `openai/` prefix where the state rule takes
    # the last segment of any path.
    "openai/o5",
    "azure/o3",
    "openrouter/openai/gpt-5",
    # Neither: `chat` anywhere in the name stops the wire rule, and the state rule's `(?!-chat)`
    # is anchored at the suffix — which is the one name where a substring test differs from both.
    "openai/gpt-5.1-chat",
]

#: What `dspy.LM(...)` is given, so `dump_state` has a cap to place. A reasoning model refuses any
#: other pairing at construction, so this is the one call that works for every name here.
CONSTRUCTED = {"temperature": 1.0, "max_tokens": 16000}


def chat_token_key(model: str) -> str:
    """The key the chat body carries a cap under, read off the body dspy built."""
    request = t.LMRequest(
        model=model,
        messages=[t.LMMessage(role="user", parts=[t.LMTextPart(text="hi")])],
        tools=[],
        config=t.LMConfig.from_kwargs(max_tokens=64),
    )
    body = to_openai_chat_request(request)
    return next(key for key in body if "token" in key)


def temperature_refused(model: str) -> bool:
    """Whether a reasoning effort at temperature 0.5 is refused before the call goes out."""
    config = t.LMConfig.from_kwargs(temperature=0.5, reasoning_effort="high")
    try:
        _validate_openai_reasoning_temperature(config, model=model, endpoint="chat")
    except Exception:
        return True
    return False


def main() -> None:
    cases = []
    for model in MODELS:
        state = dspy.LM(model, **CONSTRUCTED).dump_state()
        cases.append(
            {
                "model": model,
                "chat_token_key": chat_token_key(model),
                "temperature_refused": temperature_refused(model),
                "state_keys": list(state),
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "source": f"dspy=={PINNED} clients/openai_format.py and clients/lm.py",
                "dspy_version": PINNED,
                "note": (
                    "Per model name: the token key `to_openai_chat_request` emits, whether "
                    "`_validate_openai_reasoning_temperature` refuses a reasoning effort at "
                    "temperature 0.5, and the ordered keys of `LM.dump_state`. The first two "
                    "follow openai_format's predicate and the third follows lm.py's, which is why "
                    "`max_tokens` moves to the end of the block for some names and not others."
                ),
                "constructed_with": CONSTRUCTED,
                "cases": cases,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    disagree = sum(
        1
        for case in cases
        if (case["chat_token_key"] == "max_completion_tokens")
        != (case["state_keys"].index("max_tokens") > 6)
    )
    print(f"  wrote {OUT.name}: {len(cases)} models, {disagree} where the two rules disagree", file=sys.stderr)


if __name__ == "__main__":
    main()
