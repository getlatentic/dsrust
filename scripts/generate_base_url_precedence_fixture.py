"""Which endpoint litellm uses when a saved block carries both `api_base` and `base_url`.

dspy's `allow_unsafe_lm_state=True` preserves three redirect keys through a load, and two of them
name the same thing: `base_url` is litellm's alias for `api_base`. A port restoring both has to know
which one the call actually goes to, and that is not dspy's rule to state — it belongs to
`litellm/main.py`'s `completion`, which opens with

    if base_url is not None:
        api_base = base_url

unconditionally, so the alias wins whenever it is set. One line, and reading it is not running it:
what is recorded here is the endpoint the OpenAI client is **constructed with**, captured by
spying on `openai.OpenAI.__init__` under each of the three combinations.

    .venv/bin/python scripts/generate_base_url_precedence_fixture.py
"""

from __future__ import annotations

import json
import logging
import pathlib
import sys
import warnings
from unittest import mock

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

import litellm
import openai
from importlib.metadata import version

from pins import require

PINNED = require("dspy")
OUT = (
    pathlib.Path(__file__).parent.parent
    / "crates"
    / "dsrust"
    / "tests"
    / "conformance"
    / "state"
    / "base_url_precedence.json"
)

API_BASE = "https://compiled-against.example/v1"
ALIAS = "https://the-alias.example/v1"

CASES = [
    {"api_base": API_BASE},
    {"base_url": ALIAS},
    {"api_base": API_BASE, "base_url": ALIAS},
]


def endpoint_for(redirects: dict) -> str | None:
    """The `base_url` litellm builds its OpenAI client with, and nothing beyond that."""
    seen: dict[str, str | None] = {}
    original = openai.OpenAI.__init__

    def spy(self, *args, **kwargs):
        seen["base_url"] = kwargs.get("base_url")
        original(self, *args, **kwargs)
        raise RuntimeError("captured; no call goes out")

    with mock.patch.object(openai.OpenAI, "__init__", spy):
        try:
            litellm.completion(
                model="openai/gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
                api_key="sk-not-a-real-key",
                **redirects,
            )
        except Exception:
            pass
    return seen.get("base_url")


def main() -> None:
    cases = [{"block": case, "endpoint": endpoint_for(case)} for case in CASES]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "source": f"litellm {version('litellm')} main.completion, under dspy=={PINNED}",
                "litellm_version": version("litellm"),
                "note": (
                    "The endpoint litellm's OpenAI client is constructed with, per combination of "
                    "the two aliases a trusted load restores. `base_url` wins where both are set."
                ),
                "cases": cases,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    print(f"  wrote {OUT.name}: {len(cases)} combinations", file=sys.stderr)


if __name__ == "__main__":
    main()
