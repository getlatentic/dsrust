#!/usr/bin/env python
"""Record what dspy's `inspect_modules` renders for a program.

`Refine`'s feedback ask carries a `modules_defn` field, and this is what fills it — every
predictor's fields and original instructions, laid out with tabs and a rule between blocks. It is
prompt bytes: a model reads this text, so a tab that became spaces or a blank line that moved
changes what the feedback model is looking at.

The Rust rendering was held by a hand-written expected string. It was right — checked by running
this — but a hand-written expectation agrees with the code it tests by construction, and would keep
agreeing after a pin bump changed the format.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import dspy
from dspy.predict.refine import inspect_modules

PINNED = (Path(__file__).parent / "DSPY_VERSION").read_text().strip()
OUT = Path(__file__).resolve().parent.parent / "crates" / "dsrust" / "tests" / "conformance" / "predict"


class Pair(dspy.Module):
    """Two predictors, so the rule between blocks and the per-predictor name both show."""

    def __init__(self) -> None:
        super().__init__()
        self.draft = dspy.Predict(
            dspy.Signature("question -> draft").with_instructions("Draft an answer.")
        )
        self.settle = dspy.Predict(dspy.Signature("draft -> answer"))

    def forward(self, question):
        return self.settle(draft=self.draft(question=question).draft)


def described(name: str, program: dspy.Module) -> dict:
    return {"name": name, "rendered": inspect_modules(program)}


CASES = [
    ("a_bare_predict", lambda: dspy.Predict("question -> answer")),
    # `reasoning` is prepended by the module, and its description is empty — which shows as a
    # trailing space after the colon that a re-implementation drops without noticing.
    ("a_chain_of_thought", lambda: dspy.ChainOfThought("question -> answer")),
    ("two_predictors", Pair),
    (
        "a_described_field",
        lambda: dspy.Predict(
            dspy.Signature("question -> answer").with_updated_fields(
                "answer", desc="One word only."
            )
        ),
    ),
    (
        "multiline_instructions",
        lambda: dspy.Predict(
            dspy.Signature("question -> answer").with_instructions(
                "Answer the question.\nBe brief.\n\nNever guess."
            )
        ),
    ),
]


def main() -> None:
    if dspy.__version__ != PINNED:
        raise SystemExit(f"expected dspy {PINNED}, found {dspy.__version__}")
    fixture = {
        "source": f"generated from dspy=={PINNED} via scripts/generate_inspect_modules_fixture.py",
        "dspy_version": PINNED,
        "note": (
            "`inspect_modules` output per program shape — the text `Refine`'s feedback ask carries "
            "as `modules_defn`. Compared exactly: tabs, the trailing space after an undescribed "
            "field's colon, and the rule width are all bytes a model reads."
        ),
        "cases": [described(name, build()) for name, build in CASES],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "inspect_modules.json"
    path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
    print(f"  wrote {path.relative_to(OUT.parent.parent.parent.parent)}", file=sys.stderr)


if __name__ == "__main__":
    main()
