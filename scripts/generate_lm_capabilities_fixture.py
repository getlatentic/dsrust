"""Record what litellm says each model can be asked for natively.

dspy answers `lm.supports_function_calling` — and its two siblings — by handing the model name to
litellm, which reads them out of its bundled `model_prices_and_context_window.json`. There is no
rule behind the answers to reimplement: `claude-3-5-sonnet-20241022` cannot take a tool list while
`claude-opus-4-1` can, `ollama/qwen2.5:7b-instruct` can while `ollama/llama3.2` cannot. Any
prefix rule this crate invented would diverge from dspy for some model, silently, in the
direction of not sending tools at all.

So the table itself is vendored, and the probes beside it pin litellm's *resolution* — which key
answers for `openai/gpt-4o` — so the Rust lookup is held to litellm rather than to a guess about
how litellm works.

    .dspy-venv/bin/python scripts/generate_lm_capabilities_fixture.py
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")

# litellm answers for an ollama model by asking the local daemon which models are pulled, so on a
# machine with ollama running this file would record that machine's library rather than anything
# about litellm — `ollama/qwen2.5:7b-instruct` reads True here and False on a colleague's laptop.
# Pointing it at a closed port takes the daemon out of the answer and leaves litellm's own
# registry, which is what the vendored table can actually reproduce — the crate asks the server
# itself at run time, as litellm does, so nothing is lost by leaving it out of the table. Set
# before the import: litellm reads the base at module load.
os.environ["OLLAMA_API_BASE"] = "http://127.0.0.1:9"

import litellm

from pins import observed

ROOT = pathlib.Path(__file__).parent.parent
TABLE = ROOT / "src" / "lm" / "capabilities.json"
CASES = ROOT / "tests" / "conformance" / "lm_api" / "capabilities.json"

LITELLM = observed("litellm")

#: In the order the Rust `Capabilities` declares them.
FLAGS = ("supports_function_calling", "supports_reasoning", "supports_response_schema")

#: litellm keeps a documentation template under this key rather than a model.
TEMPLATE = "sample_spec"

#: The one member of `supported_params` anything in dspy branches on: its JSONAdapter sends *no*
#: `response_format` to a model whose `supported_params` lacks one, rather than falling back to
#: JSON mode.
#:
#: It is recorded as a probe rather than a per-model flag because litellm answers it per
#: *provider* once a model carries a prefix — `gpt-4` lacks it, `openai/gpt-4` has it — and
#: `ModelRef` refuses anything but `provider/model-id`. So the answer is the same for every model
#: this crate can hold, and what is worth pinning is that it stays that way.
RESPONSE_FORMAT_PROBES = [
    "openai/gpt-4",
    "openai/gpt-3.5-turbo-16k",
    "openai/o1",
    "openai/a-model-litellm-has-never-heard-of",
    "anthropic/claude-2.1",
    "anthropic/claude-opus-4-1",
    "openrouter/openai/gpt-4o",
    "ollama/llama2",
    "ollama_chat/qwen2.5:7b-instruct",
]

#: Model strings whose resolution the Rust lookup is held to. Spread across every provider this
#: crate speaks, and deliberately including pairs that differ — a rule fitted to one member of a
#: pair gets the other wrong.
PROBES = [
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "openai/gpt-5",
    "openai/gpt-3.5-turbo",
    "openai/o1",
    "openai/text-davinci-003",
    "anthropic/claude-opus-4-1",
    "anthropic/claude-sonnet-4-5",
    "anthropic/claude-3-5-sonnet-20241022",
    "anthropic/claude-2.1",
    "openrouter/openai/gpt-4o",
    "openrouter/anthropic/claude-3.5-sonnet",
    "ollama/qwen2.5:7b-instruct",
    "ollama/llama3.2",
    "ollama/mistral",
    "ollama/llama2",
    "openai/a-model-litellm-has-never-heard-of",
]


def table() -> dict[str, list[bool]]:
    """Every model litellm credits with at least one capability, plus every ollama row.

    A model with no capabilities is normally left out: absent and all-false read the same at the
    lookup, and two thirds of the registry says nothing. Ollama is the exception, because there
    absent does *not* mean all-false — litellm falls through to asking the server, so a row that
    says "this model has nothing" has to be distinguishable from no row at all.
    """
    rows = {}
    for name, info in litellm.model_cost.items():
        if name == TEMPLATE:
            continue
        flags = [bool(info.get(flag)) for flag in FLAGS]
        if any(flags) or name.startswith("ollama"):
            rows[name] = flags
    if not rows:
        raise SystemExit("litellm credited no model with any capability, which cannot be right")
    return dict(sorted(rows.items()))


def takes_response_format(model: str) -> bool:
    """Whether litellm credits `model` with a `response_format` parameter, asked as dspy asks."""
    try:
        params = litellm.get_supported_openai_params(model=model)
    except Exception:
        return False
    return params is not None and "response_format" in params


def probe(model: str) -> list[bool]:
    """litellm's own answers for one model string, asked exactly as dspy asks."""
    answers = []
    for flag in FLAGS:
        try:
            answers.append(bool(getattr(litellm, flag)(model=model)))
        except Exception:
            # litellm raises for a model it cannot place. dspy lets that propagate from the
            # property; what matters here is that an unknown model grants nothing.
            answers.append(False)
    return answers


def main() -> None:
    rows = table()
    TABLE.write_text(
        json.dumps(
            {
                "source": f"litellm {LITELLM} via scripts/{pathlib.Path(__file__).name}",
                "flags": list(FLAGS),
                "models": rows,
            },
            separators=(",", ":"),
        )
        + "\n"
    )
    CASES.write_text(
        json.dumps(
            {
                "source": f"litellm {LITELLM} via scripts/{pathlib.Path(__file__).name}",
                "litellm_version": LITELLM,
                "flags": list(FLAGS),
                "cases": [{"model": model, "capabilities": probe(model)} for model in PROBES],
            "response_format": [
                {"model": model, "supported": takes_response_format(model)}
                for model in RESPONSE_FORMAT_PROBES
            ],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"  wrote {TABLE.relative_to(ROOT)} ({len(rows)} models)", file=sys.stderr)
    print(f"  wrote {CASES.relative_to(ROOT)} ({len(PROBES)} probes)", file=sys.stderr)


if __name__ == "__main__":
    main()
