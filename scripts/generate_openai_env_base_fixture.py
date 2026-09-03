"""Which environment variable litellm takes the OpenAI endpoint from, when both name one.

`OPENAI_BASE_URL` is the OpenAI SDK's own spelling; `OPENAI_API_BASE` is litellm's. `completion`
reads them in that order —

    api_base
    or litellm.api_base
    or get_secret("OPENAI_BASE_URL")
    or get_secret("OPENAI_API_BASE")

— so a caller who sets only the second still reaches their endpoint, and one who sets both reaches
the first. That second fallback is the part a port forgets: reading `OPENAI_BASE_URL` alone looks
complete until someone follows litellm's own documentation, sets `OPENAI_API_BASE`, and silently
gets `api.openai.com` with a key it will refuse.

Recorded the way `generate_base_url_precedence_fixture.py` records the kwarg version: by spying on
`openai.OpenAI.__init__` and reading the `base_url` the client is actually constructed with. Reading
the four-line expression is not running it.

    .venv/bin/python scripts/generate_openai_env_base_fixture.py
"""

from __future__ import annotations

import json
import logging
import os
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
    / "openai_env_base.json"
)

BASE_URL = "https://from-base-url.example/v1"
API_BASE = "https://from-api-base.example/v1"

#: Every combination of the two, including neither — which is what pins the default.
CASES = [
    {},
    {"OPENAI_BASE_URL": BASE_URL},
    {"OPENAI_API_BASE": API_BASE},
    {"OPENAI_BASE_URL": BASE_URL, "OPENAI_API_BASE": API_BASE},
]


def endpoint_for(environment: dict) -> str | None:
    """The `base_url` litellm builds its OpenAI client with under this environment."""
    seen: dict[str, str | None] = {}
    original = openai.OpenAI.__init__

    def spy(self, *args, **kwargs):
        seen["base_url"] = kwargs.get("base_url")
        original(self, *args, **kwargs)
        raise RuntimeError("captured; no call goes out")

    names = ("OPENAI_BASE_URL", "OPENAI_API_BASE")
    saved = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ.pop(name, None)
        os.environ.update(environment)
        with mock.patch.object(openai.OpenAI, "__init__", spy):
            try:
                litellm.completion(
                    model="openai/gpt-4o-mini",
                    messages=[{"role": "user", "content": "hi"}],
                    api_key="sk-not-a-real-key",
                )
            except Exception:
                pass
    finally:
        for name, value in saved.items():
            os.environ.pop(name, None)
            if value is not None:
                os.environ[name] = value
    return seen.get("base_url")


def main() -> None:
    cases = [
        {"environment": case, "endpoint": endpoint_for(case)} for case in CASES
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "source": f"litellm {version('litellm')} main.completion, under dspy=={PINNED}",
                "litellm_version": version("litellm"),
                "note": (
                    "The endpoint litellm's OpenAI client is constructed with, per combination of "
                    "the two environment variables that name one. `OPENAI_BASE_URL` wins where "
                    "both are set; `OPENAI_API_BASE` is the fallback."
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
