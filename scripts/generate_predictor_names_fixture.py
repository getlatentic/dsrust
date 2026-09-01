"""Record what dspy calls each predictor in a composed program, by asking it.

`named_predictors` names a predictor by its path from the root, and that name is what `load_state`
indexes and what a GEPA candidate is keyed by — so two programs of the same shape that disagree
about it cannot exchange a saved state in either direction.

The rule is not "the field holding it". A leaf `Predict` calls itself `self` and the field name
replaces that; anything carrying a name of its own is *prefixed*, so a `ChainOfThought` held in
`answer_generator` is `answer_generator.predict` and not `answer_generator`. The difference only
shows once a step is itself composed, which is why it went unnoticed.

    .dspy-venv/bin/python scripts/generate_predictor_names_fixture.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import dspy

OUT = pathlib.Path(__file__).parent.parent / "crates" / "dsrust" / "tests" / "conformance" / "module"
PINNED = (pathlib.Path(__file__).parent / "DSPY_VERSION").read_text().strip()


class Inner(dspy.Module):
    """A step that is itself composed, so its own predictor has a name to be prefixed."""

    def __init__(self):
        super().__init__()
        self.step = dspy.ChainOfThought("a -> b")

    def forward(self, **kwargs):
        return self.step(**kwargs)


class Composed(dspy.Module):
    def __init__(self):
        super().__init__()
        self.flat = dspy.Predict("question -> query")
        self.cot = dspy.ChainOfThought("question, context -> answer")
        self.nested = Inner()

    def forward(self, **kwargs):
        return self.flat(**kwargs)


def named(module) -> list[str]:
    return [name for name, _ in module.named_predictors()]


def main() -> None:
    if dspy.__version__ != PINNED:
        raise SystemExit(f"expected dspy {PINNED}, found {dspy.__version__}")

    fixture = {
        "source": f"generated from dspy=={PINNED} via scripts/generate_predictor_names_fixture.py",
        "dspy_version": PINNED,
        "cases": [
            {"what": "a bare Predict", "names": named(dspy.Predict("a -> b"))},
            {"what": "a bare ChainOfThought", "names": named(dspy.ChainOfThought("a -> b"))},
            {"what": "a composed module", "names": named(Composed())},
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "predictor_names.json"
    path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
    for case in fixture["cases"]:
        print(f"  {case['what']}: {case['names']}", file=sys.stderr)


if __name__ == "__main__":
    main()
