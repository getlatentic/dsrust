"""Record what a `Reasoning` output does to a prompt, on a model that reasons and one that does not.

`Reasoning.adapt_to_native_lm_feature` takes the field out of the rendered signature when the model
exposes extended thinking, and the request carries a `reasoning_effort` instead — the model thinks on
its own channel rather than filling a `[[ ## reasoning ## ]]` block the prompt asked for. A model
that cannot reason renders the field as prose, unchanged.

`tests/native_reasoning.rs` asserted that as a *property* — `!prompt.contains("[[ ## reasoning ## ]]")`
— which is the check `citations-native` started with and then failed. Comparing whole prompts there
found a missing type-description line and a schema note whose key order differed, neither of which a
`contains` can see. Removing a field from a render moves the numbering of every output line after it,
so the same two failure modes apply here.

    .venv/bin/python scripts/generate_reasoning_fixture.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import dspy
from dspy.adapters.types.reasoning import Reasoning

from pins import require

OUT = (
    pathlib.Path(__file__).parent.parent / "crates" / "dsrust" / "tests" / "conformance" / "adapter"
)
PINNED = require("dspy")


class Reasoned(dspy.Signature):
    """Answer the question."""

    question: str = dspy.InputField()
    reasoning: Reasoning = dspy.OutputField()
    answer: str = dspy.OutputField()


def rendered_for(model: str, *, reasons: bool) -> dict:
    """The system prompt `Reasoned` renders to, with native reasoning available or not.

    `adapt_to_native_lm_feature` asks the lm whether it supports reasoning, so the two arms are
    driven by a model whose capability differs rather than by calling the classmethod both ways.
    """
    lm = dspy.LM(model, api_key="not-used")
    lm_kwargs: dict = {}
    prepared = Reasoned
    for name, field in Reasoned.output_fields.items():
        annotation = field.annotation
        if isinstance(annotation, type) and issubclass(annotation, dspy.Type):
            prepared = annotation.adapt_to_native_lm_feature(prepared, name, lm, lm_kwargs)
    messages = dspy.ChatAdapter().format(prepared, [], {"question": "what is six times seven"})
    return {
        "model": model,
        "reasons": reasons,
        "renders_the_field": "reasoning" in prepared.output_fields,
        "lm_kwargs": lm_kwargs,
        "system": messages[0]["content"],
    }


def main() -> None:
    # `o3` reasons; `gpt-4o-mini` does not. litellm's registry is what decides, and dspy reads it.
    reasoning_model = rendered_for("openai/o3", reasons=True)
    plain_model = rendered_for("openai/gpt-4o-mini", reasons=False)

    fixture = {
        "source": f"generated from dspy=={PINNED} via scripts/generate_reasoning_fixture.py",
        "dspy_version": PINNED,
        "note": (
            "Both arms of Reasoning.adapt_to_native_lm_feature: a model that reasons drops the "
            "field from the render and carries a reasoning_effort, one that does not renders it. "
            "The whole system prompt is recorded, not a substring — removing a field renumbers "
            "every output line after it."
        ),
        "renders": [reasoning_model, plain_model],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "reasoning_native.json"
    path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
    print(f"  wrote {path.relative_to(OUT.parent.parent.parent)}", file=sys.stderr)
    for render in fixture["renders"]:
        print(
            f"    {render['model']}: renders the field = {render['renders_the_field']}, "
            f"lm_kwargs = {render['lm_kwargs']}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
