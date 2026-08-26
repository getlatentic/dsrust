"""Record which attempts `BootstrapFewShotWithRandomSearch` actually makes, by running it.

Two of its `compile` arguments had no Rust equivalent until now, and neither is guessable from the
signature:

  - `restrict` skips any seed it does not name — `if (restrict is not None) and (seed not in
    restrict): continue`. It filters the range the search was already walking, so it can narrow the
    attempts but never add one, and it does not special-case the three baselines.
  - `labeled_sample` reaches exactly one attempt, seed `-2`, where it is handed to `LabeledFewShot`
    as its `sample`. The other `n + 2` attempts never see it.

`score_data` carries the seed of every attempt that ran, which is what makes this recordable without
reading the loop: the search is asked to compile and then says what it did.

    .venv/bin/python scripts/generate_random_search_fixture.py
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
from dspy.utils.dummies import DummyLM

from pins import require

OUT = (
    pathlib.Path(__file__).parent.parent / "crates" / "dsrust" / "tests" / "conformance" / "optimize"
)
PINNED = require("dspy")

TRAINSET = [("capital of France?", "Paris"), ("capital of Spain?", "Madrid"),
            ("capital of Italy?", "Rome"), ("capital of Japan?", "Tokyo")]

#: (num_candidate_programs, restrict, labeled_sample). The `restrict` values cover: nothing skipped,
#: a baseline kept and others dropped, a candidate-only set, and a seed the range never reaches —
#: which must be absent rather than an error or an extra attempt.
CASES = [
    (4, None, True),
    (4, [-2], True),
    (4, [-2], False),
    (4, [-2, 1, 3], True),
    (4, [0, 2], True),
    (2, [1, 99], True),
    (4, [-3], True),
    (4, None, False),
]


def compile_once(programs: int, restrict, labeled_sample: bool) -> dict:
    answers = {question: {"answer": answer} for question, answer in TRAINSET}
    dspy.configure(lm=DummyLM(answers))
    trainset = [dspy.Example(question=q, answer=a).with_inputs("question") for q, a in TRAINSET]

    search = dspy.BootstrapFewShotWithRandomSearch(
        metric=lambda example, prediction, trace=None: example.answer == prediction.answer,
        max_bootstrapped_demos=2,
        max_labeled_demos=2,
        num_candidate_programs=programs,
        num_threads=1,
    )
    compiled = search.compile(
        dspy.Predict("question -> answer"),
        trainset=trainset,
        restrict=restrict,
        labeled_sample=labeled_sample,
    )
    return {
        "num_candidate_programs": programs,
        "restrict": restrict,
        "labeled_sample": labeled_sample,
        # Every attempt that ran, in order. This is the whole point: `restrict` is only observable
        # as which seeds are absent.
        "seeds": [entry["seed"] for entry in compiled.candidate_programs],
        # The demos the labels-only attempt kept. That attempt, seed -2, is the *only* place
        # `labeled_sample` reaches — it becomes `LabeledFewShot`'s `sample`, deciding whether the
        # demos are drawn at random or taken in order. Without recording them, a port that ignored
        # the argument would pass every assertion above.
        "labels_only_demos": labels_only(compiled),
    }


def labels_only(compiled) -> list | None:
    """The demos the seed `-2` attempt left on the predictor, if that attempt ran."""
    for entry in compiled.candidate_programs:
        if entry["seed"] == -2:
            program = entry["program"]
            return [dict(demo) for _, predictor in program.named_predictors()
                    for demo in predictor.demos]
    return None


def main() -> None:
    cases = [compile_once(*case) for case in CASES]
    fixture = {
        "source": f"generated from dspy=={PINNED} via scripts/generate_random_search_fixture.py",
        "dspy_version": PINNED,
        "note": (
            "Which seeds each search attempted. `restrict` filters the range the loop already "
            "walks, so it narrows and never extends, and it does not spare the three baselines."
        ),
        "trainset": [{"question": q, "answer": a} for q, a in TRAINSET],
        "cases": cases,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "random_search.json"
    path.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"  wrote {path.relative_to(OUT.parent.parent.parent)}", file=sys.stderr)
    for case in cases:
        print(f"    restrict={str(case['restrict']):<14} -> seeds {case['seeds']}", file=sys.stderr)

    # A case set where every restrict keeps everything would pass against a port that ignores the
    # argument entirely.
    unrestricted = next(case for case in cases if case["restrict"] is None)
    if not any(
        case["restrict"] is not None and case["seeds"] != unrestricted["seeds"] for case in cases
    ):
        raise SystemExit("no restrict case actually drops an attempt")

    # `labeled_sample` is only observable in the demos the seed -2 attempt kept, and only when
    # drawing and taking-in-order actually disagree. A trainset where they coincide would let a port
    # that ignores the argument pass.
    drawn = next(c for c in cases if c["restrict"] == [-2] and c["labeled_sample"])
    ordered = next(c for c in cases if c["restrict"] == [-2] and not c["labeled_sample"])
    if drawn["labels_only_demos"] == ordered["labels_only_demos"]:
        raise SystemExit(
            "labeled_sample changes nothing on this trainset — sampling and taking in order agree"
        )
    print(
        f"    labeled_sample observable: drawn {[d.get('question') for d in drawn['labels_only_demos']]}"
        f" vs ordered {[d.get('question') for d in ordered['labels_only_demos']]}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
