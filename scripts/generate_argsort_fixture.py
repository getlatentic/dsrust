"""numpy's `argsort`, recorded because its tie-breaking is observable.

optuna's numerical Parzen estimator sorts the observations to measure how far each one's neighbours
are, and maps the widths back through `np.argsort`'s permutation. With duplicate observations — which
is the normal case for an integer parameter — *which* duplicate lands at a run's boundary decides
which kernel gets the wide sigma, so the permutation is not an implementation detail.

`np.argsort`'s default is `kind='quicksort'`, which is introsort: insertion sort while the partition
is 15 elements or fewer, quicksort above that. Insertion sort is stable and introsort is not, so an
array of 16 sorts like a stable sort and an array of 17 does not — which is exactly why every
recorded TPE run under seventeen observations agreed with a stable port and the first one over it
did not.

The corpus is integer-valued and duplicate-heavy on purpose: that is what the estimator sees, and
ties are the only place the two sorts can differ.

    .venv/bin/python scripts/generate_argsort_fixture.py
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from pins import observed

OUT = (
    pathlib.Path(__file__).parent.parent
    / "crates"
    / "dsrust-tpe"
    / "tests"
    / "conformance"
    / "argsort.json"
)


def encoded(values) -> list:
    """`NaN` has no JSON spelling, and `json.dumps` writes a bare `NaN` that stricter readers
    refuse. The tag survives a round trip; the value it stands for is the point of the case."""
    return ["nan" if np.isnan(v) else float(v) for v in values]


def cases() -> list[dict]:
    rng = np.random.RandomState(20260828)
    rows: list[dict] = []

    # Straddle the threshold exactly: 15, 16 and 17 are insertion, insertion, and quicksort.
    for n in (0, 1, 2, 3, 7, 8, 15, 16, 17, 18, 24, 27, 32, 40, 64, 129, 200):
        for distinct in (2, 3, 6, 50):
            values = rng.randint(0, distinct, size=n).astype(float)
            rows.append(
                {
                    "name": f"n{n}_of_{distinct}_values",
                    "values": [float(v) for v in values],
                    "argsort": [int(i) for i in np.argsort(values)],
                }
            )

    # Shapes a random draw is unlikely to produce and the partition logic cares about.
    shaped = {
        "all_equal": [2.0] * 30,
        "already_sorted": [float(i) for i in range(30)],
        "reversed": [float(30 - i) for i in range(30)],
        "two_runs": [0.0] * 15 + [1.0] * 15,
        "one_odd_one_out": [1.0] * 29 + [0.0],
        "negatives_and_zero": [float(i % 3 - 1) for i in range(25)],
        # NaN sorts last, which is the whole of `DOUBLE_LT`'s second clause.
        "nan_among_numbers": [1.0, float("nan"), 0.0, 2.0, float("nan"), 1.0] * 5,
        "all_nan": [float("nan")] * 20,
        # The partition choice — push the larger side, walk the smaller — takes a different arm
        # depending on which side the pivot lands in, so both skews are here.
        "pivot_near_the_start": [0.0] + [float(i) for i in range(1, 40)],
        "pivot_near_the_end": [float(i) for i in range(39)] + [0.0],
        "one_huge_run_then_ascending": [0.0] * 30 + [float(i) for i in range(1, 20)],
    }
    for name, values in shaped.items():
        array = np.array(values, dtype=float)
        rows.append(
            {
                "name": name,
                "values": encoded(array),
                "argsort": [int(i) for i in np.argsort(array)],
            }
        )
    return rows


def main() -> None:
    rows = cases()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "source": f"generated from numpy=={observed('numpy')} via "
                "scripts/generate_argsort_fixture.py",
                "note": (
                    "`argsort` is numpy's default kind — introsort, whose insertion-sort floor at "
                    "15 makes it stable up to 16 elements and unstable above."
                ),
                "cases": rows,
            },
            indent=2,
        )
        + "\n"
    )
    unstable = sum(
        1
        for row in rows
        if row["argsort"] != [int(i) for i in np.argsort(np.array(row["values"]), kind="stable")]
    )
    print(f"  cases: {len(rows)}")
    print(f"  of which a stable sort would get wrong: {unstable}")
    print(f"wrote {OUT.relative_to(pathlib.Path(__file__).parent.parent)}")


if __name__ == "__main__":
    main()
