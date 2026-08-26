"""Record the demo sets dspy's create_n_fewshot_demo_sets builds, by running it.

This is MIPROv2's Step 1. A keyed DummyLM removes the model, so bootstrapping is deterministic, and
the CPython RNG (`random.Random(seed)`) is seeded — so the mix of strategies (zero-shot, labels-only,
unshuffled bootstrap, shuffled bootstraps whose size is drawn) is a pure function of trainset, the
budgets and the seed. What is compared is the demos each set draws.

    .dspy-venv/bin/python scripts/generate_demo_sets_fixture.py

Keep TRAINSET/ANSWERS in step with `src/optimize/scripted.rs`.
"""

from __future__ import annotations

import json
import pathlib
import random
import sys

import dspy
from dspy.teleprompt.utils import create_n_fewshot_demo_sets
from dspy.utils.dummies import DummyLM

OUT = pathlib.Path(__file__).parent.parent / "crates" / "dsrust" / "tests" / "conformance" / "optimize"
PINNED = (pathlib.Path(__file__).parent / "DSPY_VERSION").read_text().strip()

TRAINSET = [
    ("capital of France?", "Paris"),
    ("capital of Germany?", "Berlin"),
    *((f"riddle {index}?", f"riddle {index}!") for index in range(4)),
]
# The table solves the two capitals (the model's answer matches the label) and gets the riddles
# wrong, so exactly two examples can be bootstrapped.
ANSWERS = {q: {"answer": ("Paris" if "France" in q else "Berlin")} for q, _ in TRAINSET}


def trainset() -> list[dspy.Example]:
    return [dspy.Example(question=q, answer=a).with_inputs("question") for q, a in TRAINSET]


class One(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict("question -> answer")

    def forward(self, question):
        return self.predict(question=question)


def metric(example, prediction, trace=None) -> bool:
    return example.answer == prediction.answer


# (num_candidate_sets, max_labeled, max_bootstrapped, seed).
CASES = [
    (5, 2, 2, 9),
    (4, 0, 2, 7),  # max_labeled=0, so seed -2 falls through to a shuffled set
    (3, 4, 1, 0),  # only the three special sets, labels-only present
]


def demos_of(demos) -> list[dict]:
    """Every field each demo carries, not just the turn.

    This projected each demo down to its question and answer, which drops `augmented` — the marker
    a bootstrap writes and MIPROv2's own proposer gathers on. Both sides of the comparison lost it
    together, so a port that never wrote the marker matched this golden exactly, and the proposer
    it feeds would have been shown nothing while every test stayed green.
    """
    return [dict(demo.toDict()) for demo in demos]


def build_once(num_sets: int, max_labeled: int, max_bootstrapped: int, seed: int) -> dict:
    dspy.configure(lm=DummyLM(dict(ANSWERS)))
    demo_candidates = create_n_fewshot_demo_sets(
        student=One(),
        num_candidate_sets=num_sets,
        trainset=trainset(),
        max_labeled_demos=max_labeled,
        max_bootstrapped_demos=max_bootstrapped,
        metric=metric,
        teacher_settings={},
        seed=seed,
        rng=random.Random(seed),
    )
    # demo_candidates[predictor_index] = list of sets, each a list of demos.
    sets = [demos_of(one_set) for one_set in demo_candidates[0]]
    return {
        "num_sets": num_sets,
        "max_labeled": max_labeled,
        "max_bootstrapped": max_bootstrapped,
        "seed": seed,
        "sets": sets,
    }


def main() -> None:
    if dspy.__version__ != PINNED:
        raise SystemExit(f"expected dspy {PINNED}, found {dspy.__version__}")
    fixture = {
        "source": f"generated from dspy=={PINNED} via scripts/generate_demo_sets_fixture.py",
        "dspy_version": PINNED,
        "trainset": [{"question": q, "answer": a} for q, a in TRAINSET],
        "answers": {q: a["answer"] for q, a in ANSWERS.items()},
        "cases": [build_once(*case) for case in CASES],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "demo_sets.json"
    path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
    print(f"  wrote {path.relative_to(OUT.parent.parent)}", file=sys.stderr)


if __name__ == "__main__":
    main()
