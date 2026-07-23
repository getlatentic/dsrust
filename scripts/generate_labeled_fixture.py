"""Record what dspy's own LabeledFewShot selects, by running it.

LabeledFewShot makes no model calls — it fills every predictor's demos from the trainset, with
`random.Random(0).sample` when sampling. So a compile is a pure function of the trainset, `k`, and
`sample`, and what is left to compare is exactly which labelled demos it draws and in what order.

    .dspy-venv/bin/python scripts/generate_labeled_fixture.py

Keep TRAINSET in step with `src/optimize/scripted.rs::trainset`.
"""

from __future__ import annotations

import json
import pathlib
import sys

import dspy

OUT = pathlib.Path(__file__).parent.parent / "tests" / "conformance" / "optimize"
PINNED = (pathlib.Path(__file__).parent / "DSPY_VERSION").read_text().strip()

TRAINSET = [
    ("capital of France?", "Paris"),
    ("capital of Germany?", "Berlin"),
    *((f"riddle {index}?", f"riddle {index}!") for index in range(4)),
]


def trainset() -> list[dspy.Example]:
    return [dspy.Example(question=q, answer=a).with_inputs("question") for q, a in TRAINSET]


class One(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict("question -> answer")

    def forward(self, question):
        return self.predict(question=question)


class Two(dspy.Module):
    """Two predictors, so the second predictor draws its own sample from the advanced generator."""

    def __init__(self):
        super().__init__()
        self.first = dspy.Predict("question -> answer")
        self.second = dspy.Predict("question -> answer")

    def forward(self, question):
        return self.second(question=self.first(question=question).answer)


# (program factory, k, sample, predictors). Distinct answers throughout, so a drawn demo is
# identifiable; the k>len case exercises a whole-trainset reordering, sample=False the plain take.
CASES = [
    (One, 2, True, 1),
    (One, 4, True, 1),
    (One, 2, False, 1),
    (One, 10, True, 1),
    (Two, 3, True, 2),
]


def demos_of(predictor) -> list[dict]:
    return [{"question": demo.question, "answer": demo.answer} for demo in predictor.demos]


def compile_once(factory, k: int, sample: bool, predictors: int) -> dict:
    compiled = dspy.LabeledFewShot(k=k).compile(factory(), trainset=trainset(), sample=sample)
    return {
        "k": k,
        "sample": sample,
        "predictors": predictors,
        "demos": [demos_of(predictor) for _, predictor in compiled.named_predictors()],
    }


def main() -> None:
    if dspy.__version__ != PINNED:
        raise SystemExit(f"expected dspy {PINNED}, found {dspy.__version__}")
    fixture = {
        "source": f"generated from dspy=={PINNED} via scripts/generate_labeled_fixture.py",
        "dspy_version": PINNED,
        "trainset": [{"question": q, "answer": a} for q, a in TRAINSET],
        "cases": [compile_once(*case) for case in CASES],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "labeled_few_shot.json"
    path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
    print(f"  wrote {path.relative_to(OUT.parent.parent)}", file=sys.stderr)


if __name__ == "__main__":
    main()
