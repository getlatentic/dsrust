"""Record what dspy's `GEPA.auto_budget` answers, by asking it.

Pure arithmetic, and every part of it is a place to be off by one: a `log2` under an `int()` that
truncates rather than rounds, a floor division whose divisor is `full_eval_steps` and not
`full_eval_steps + 1` despite the comment above it saying `m+1`, and a final full evaluation that
is added only when the trial count is *below* that divisor. The cases below sit on each of those.

    .venv/bin/python scripts/generate_gepa_budget_fixture.py
"""

from __future__ import annotations

import json
import pathlib
import sys

from dspy.teleprompt.gepa.gepa import GEPA

from pins import require

OUT = pathlib.Path(__file__).parent.parent / "crates" / "dsrust" / "tests" / "conformance" / "optimize"
PINNED = require("dspy")

#: (num_preds, num_candidates, valset_size, minibatch_size, full_eval_steps)
CASES = [
    # dspy's own defaults for the two it defaults.
    (1, 2, 100, 35, 5),
    (3, 8, 250, 35, 5),
    # One candidate: log2(1) is 0, so the 1.5 * candidates arm wins and truncation bites.
    (1, 1, 50, 35, 5),
    (4, 1, 50, 35, 5),
    # The two arms of the max, either side of where they cross.
    (1, 4, 100, 35, 5),
    (1, 32, 100, 35, 5),
    (8, 4, 100, 35, 5),
    # `full_eval_steps` at its floor, and above the trial count — the branch that adds a final
    # full evaluation only when N < m.
    (1, 2, 100, 35, 1),
    (1, 2, 100, 35, 100),
    (2, 3, 40, 10, 3),
    # Zeroes, which are allowed and change which branches run.
    (1, 2, 0, 35, 5),
    (1, 2, 100, 0, 5),
    # A large one, where a float log2 and an integer count could drift apart.
    (12, 64, 1000, 35, 5),
]

#: (kwargs, the message dspy raises). `num_trials` is derived, so the only way to make it negative
#: is a negative candidate count, which `log2` rejects first — hence only two reachable refusals.
REFUSALS = [
    ({"num_preds": 1, "num_candidates": 2, "valset_size": -1}, "must be >= 0"),
    ({"num_preds": 1, "num_candidates": 2, "valset_size": 10, "minibatch_size": -1}, "must be >= 0"),
    ({"num_preds": 1, "num_candidates": 2, "valset_size": 10, "full_eval_steps": 0}, "must be >= 1"),
]


def main() -> None:
    # A reflection LM is demanded by the constructor and never reached: `auto_budget` is
    # arithmetic over its arguments and touches nothing on the instance.
    gepa = GEPA(metric=lambda *a, **k: 0.0, max_metric_calls=1, reflection_lm=lambda *a, **k: "")
    cases = [
        {
            "num_preds": p,
            "num_candidates": c,
            "valset_size": v,
            "minibatch_size": m,
            "full_eval_steps": f,
            "budget": gepa.auto_budget(p, c, v, m, f),
        }
        for p, c, v, m, f in CASES
    ]
    refusals = []
    for kwargs, expected in REFUSALS:
        try:
            gepa.auto_budget(**kwargs)
            raise SystemExit(f"expected {kwargs} to be refused, and it was not")
        except ValueError as error:
            if expected not in str(error):
                raise SystemExit(f"unexpected refusal for {kwargs}: {error}") from None
            refusals.append({"kwargs": kwargs, "error": str(error)})

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "gepa_budget.json"
    path.write_text(
        json.dumps(
            {
                "source": f"generated from dspy=={PINNED} via scripts/{pathlib.Path(__file__).name}",
                "dspy_version": PINNED,
                "cases": cases,
                "refusals": refusals,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"  wrote {path.relative_to(OUT.parent.parent)} ({len(cases)} cases)", file=sys.stderr)


if __name__ == "__main__":
    main()
