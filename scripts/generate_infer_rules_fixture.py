"""dspy `InferRules`, recorded by running each of its pieces.

The optimizer bootstraps, then asks a model to write rules from the trainset, appends them to each
predictor's instruction, and keeps the candidate that scores best on a validation set. Four parts of
it are bytes rather than behaviour, and all four are recorded here:

* the text the examples are rendered into, which is not any adapter's format;
* the sentence the rules are appended under;
* the induction signature's own instruction, an f-string over `num_rules`;
* the rollout ids, drawn from a `random.Random(0)` shared across every candidate and predictor, so
  the stream is one sequence and not one per call.

Plus the split: with no validation set, upstream halves the trainset and keeps the *first* half to
learn from.

    .venv/bin/python scripts/generate_infer_rules_fixture.py
"""

from __future__ import annotations

import json
import logging
import pathlib
import random
import warnings

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

import dspy
from dspy.teleprompt.infer_rules import InferRules, RulesInductionProgram

from pins import require

PINNED = require("dspy")
OUT = (
    pathlib.Path(__file__).parent.parent
    / "crates"
    / "dsrust"
    / "tests"
    / "conformance"
    / "optimize"
    / "infer_rules.json"
)


def signature() -> type[dspy.Signature]:
    return dspy.Signature("question, hint -> answer")


def demos() -> list[dict]:
    return [
        {"question": "capital of France?", "hint": "europe", "answer": "Paris", "extra": "ignored"},
        {"question": "capital of Peru?", "hint": "south america", "answer": "Lima"},
        # A demo missing an output field, and one whose value is not a string.
        {"question": "capital of Chad?", "hint": "africa"},
        {"question": "how many?", "hint": 3, "answer": None},
    ]


def main() -> None:
    optimizer = InferRules(metric=lambda e, p, trace=None: 1.0, num_candidates=2, num_rules=7)

    formatted = optimizer.format_examples(demos(), signature())

    # `get_predictor_demos` narrows a trainset row to the signature's own fields, keeping the row's
    # order rather than the signature's.
    trainset = [dspy.Example(**row).with_inputs("question", "hint") for row in demos()]
    predictor = dspy.Predict(signature())
    narrowed = optimizer.get_predictor_demos(trainset, predictor)

    appended = dspy.Predict(signature())
    before = appended.signature.instructions
    optimizer.update_program_instructions(appended, "1. Be brief.\n2. Be exact.")

    # The induction signature's instruction is an f-string over `num_rules`.
    inductions = {
        str(n): RulesInductionProgram(n).rules_induction.predict.signature.instructions for n in (1, 7, 10)
    }

    # One `random.Random(0)` is shared by every call the optimizer makes, so the rollout ids are a
    # single stream. Recorded straight from CPython rather than from a run, since a run's length
    # depends on the trainset.
    rng = random.Random(0)
    rollouts = [rng.randint(0, 10**9) for _ in range(12)]

    # The split when no valset is given: the *first* half is what the rules are learned from.
    splits = {
        str(n): int(0.5 * n) for n in (0, 1, 2, 3, 4, 5, 9, 10, 11)
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "source": f"generated from dspy=={PINNED} via scripts/generate_infer_rules_fixture.py",
                "demos": demos(),
                "formatted_examples": formatted,
                "narrowed_demos": [dict(row) for row in narrowed],
                "instruction_before": before,
                "instruction_after": appended.signature.instructions,
                "induction_instructions": inductions,
                "rollout_ids": rollouts,
                "train_size_for_length": splits,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"  formatted examples : {len(formatted)} chars")
    print(f"  narrowed demos     : {[sorted(row) for row in narrowed]}")
    print(f"  rollout ids        : {rollouts[:4]}…")
    print(f"wrote {OUT.relative_to(pathlib.Path(__file__).parent.parent)}")


if __name__ == "__main__":
    main()
