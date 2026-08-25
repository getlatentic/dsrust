"""Record where dspy's evaluation gives up, by running it.

`max_errors` is the one place an evaluation stops being a score and becomes a failure, and both
halves of that boundary are easy to get wrong. Upstream's `ParallelExecutor` cancels when
`error_count >= max_errors` — so it tolerates `max_errors - 1` failing rows and raises on the
next — and the raise propagates out of `Evaluate.__call__` rather than being folded into a partial
score. A port that returns what it managed to score hands back a number that reads as a result.

    .venv/bin/python scripts/generate_evaluate_errors_fixture.py
"""

from __future__ import annotations

import json
import logging
import pathlib
import sys

import dspy

from pins import require

OUT = pathlib.Path(__file__).parent.parent / "crates" / "dsrust" / "tests" / "conformance" / "evaluate"
PINNED = require("dspy")

#: `(max_errors, failing rows, total rows)` around the boundary, which is where a port drifts.
CASES = [(3, 2, 6), (3, 3, 6), (1, 1, 6), (2, 1, 6), (2, 2, 6), (10, 0, 6)]


def run(cap: int, failing: int, total: int) -> dict:
    seen = {"count": 0}

    def program(**kwargs):
        seen["count"] += 1
        if seen["count"] <= failing:
            raise ValueError("boom")
        return dspy.Prediction(answer="x")

    devset = [
        dspy.Example(question=str(i), answer="x").with_inputs("question") for i in range(total)
    ]
    evaluate = dspy.Evaluate(
        devset=devset,
        metric=lambda example, pred, trace=None: 1.0,
        max_errors=cap,
        num_threads=1,
        display_progress=False,
    )
    try:
        outcome = evaluate(program)
        return {"raised": False, "results": len(outcome.results), "score": float(outcome.score)}
    except Exception as error:  # noqa: BLE001 — the message is the artefact
        return {"raised": True, "message": str(error)}


def main() -> None:
    logging.disable(logging.ERROR)
    fixture = {
        "source": f"generated from dspy=={PINNED} via scripts/generate_evaluate_errors_fixture.py",
        "dspy_version": PINNED,
        "cases": [
            {"max_errors": cap, "failing": failing, "rows": total, **run(cap, failing, total)}
            for cap, failing, total in CASES
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "max_errors.json"
    path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
    print(f"  wrote {path.relative_to(OUT.parent.parent)}", file=sys.stderr)


if __name__ == "__main__":
    main()
