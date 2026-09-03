"""What dspy's built-in LM-judge metrics render, and what `f1_score` returns.

`evaluate/auto_evaluation.py` is dspy's answer to "score a free-text answer without a string
match": four signatures a `ChainOfThought` asks, and two `Module`s that combine the numbers.
`SemanticF1` is the one most dspy programs reach for.

Two things are recorded, because two things can drift apart:

  - the **rendered prompt** for each of the four signatures, taken from the real `ChatAdapter` over
    the real signature. These are instructions lifted from a class docstring, so their whitespace is
    `inspect.cleandoc`'s and not anybody's typing, and every field carries a description that is
    prompt text.
  - **`f1_score`** over a grid, including the parts a reading gets wrong: both arguments are clamped
    to `[0, 1]` *before* the harmonic mean, so a model answering `precision=1.4` scores as 1.0 and a
    negative one as 0.0, and the `precision + recall == 0` guard means two zeros are 0.0 rather than
    a division by zero.

    .venv/bin/python scripts/generate_auto_evaluation_fixture.py
"""

from __future__ import annotations

import itertools
import json
import logging
import pathlib
import sys
import warnings

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

import dspy
from dspy.evaluate.auto_evaluation import (
    AnswerCompleteness,
    AnswerGroundedness,
    DecompositionalSemanticRecallPrecision,
    SemanticRecallPrecision,
    f1_score,
)

from pins import require

PINNED = require("dspy")
OUT = (
    pathlib.Path(__file__).parent.parent
    / "crates"
    / "dsrust"
    / "tests"
    / "conformance"
    / "evaluate"
    / "auto_evaluation.json"
)

SIGNATURES = {
    "SemanticRecallPrecision": SemanticRecallPrecision,
    "DecompositionalSemanticRecallPrecision": DecompositionalSemanticRecallPrecision,
    "AnswerCompleteness": AnswerCompleteness,
    "AnswerGroundedness": AnswerGroundedness,
}

#: One filled-in call per signature, so the user turn is rendered too rather than only the system
#: one. The values are deliberately dull — what is being pinned is the layout around them.
INPUTS = {
    "SemanticRecallPrecision": {
        "question": "What is the capital of France?",
        "ground_truth": "Paris, the capital since 987.",
        "system_response": "Paris.",
    },
    "DecompositionalSemanticRecallPrecision": {
        "question": "What is the capital of France?",
        "ground_truth": "Paris, the capital since 987.",
        "system_response": "Paris.",
    },
    "AnswerCompleteness": {
        "question": "What is the capital of France?",
        "ground_truth": "Paris, the capital since 987.",
        "system_response": "Paris.",
    },
    "AnswerGroundedness": {
        "question": "What is the capital of France?",
        "retrieved_context": "France's capital is Paris.",
        "system_response": "Paris.",
    },
}

#: Both arms of the clamp, both sides of the guard, and the ordinary middle.
F1_CASES = [
    (precision, recall)
    for precision, recall in itertools.product(
        [-1.0, -0.0001, 0.0, 0.25, 0.5, 0.66, 1.0, 1.0001, 2.0],
        [-1.0, 0.0, 0.3, 0.5, 1.0, 5.0],
    )
]


def described(name: str, cls: type) -> dict:
    """The signature as fields, and the prompt a `ChainOfThought` over it renders."""
    adapter = dspy.ChatAdapter()
    reasoning = dspy.ChainOfThought(cls)
    messages = adapter.format(
        signature=reasoning.predict.signature, demos=[], inputs=INPUTS[name]
    )
    return {
        "name": name,
        "instructions": cls.instructions,
        "inputs": [
            {"name": field, "desc": info.json_schema_extra.get("desc", ""), "annotation": info.annotation.__name__}
            for field, info in cls.input_fields.items()
        ],
        "outputs": [
            {"name": field, "desc": info.json_schema_extra.get("desc", ""), "annotation": info.annotation.__name__}
            for field, info in cls.output_fields.items()
        ],
        "chain_of_thought_messages": messages,
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "source": f"dspy=={PINNED} evaluate/auto_evaluation.py",
                "dspy_version": PINNED,
                "note": (
                    "The four judge signatures as dspy declares them, the prompt a "
                    "`ChainOfThought` over each renders, and `f1_score` over a grid that crosses "
                    "both clamp arms and the zero guard."
                ),
                "signatures": [described(name, cls) for name, cls in SIGNATURES.items()],
                "f1_score": [
                    {"precision": precision, "recall": recall, "f1": f1_score(precision, recall)}
                    for precision, recall in F1_CASES
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    print(
        f"  wrote {OUT.name}: {len(SIGNATURES)} signatures, {len(F1_CASES)} f1 cases",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
