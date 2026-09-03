"""Record numpy's and optuna's arithmetic for the three parzen helpers no test constrained.

A mutation run over `dsrust-tpe` left 18 survivors, every one in `parzen.rs`, in three functions
whose correctness the module doc *asserts* and nothing checked:

  - `pairwise_sum` — numpy's `add.reduce`. The `n < 8` boundary and the whole `n > 128` recursive
    split were unreached: no sum in the existing fixtures is longer than a few dozen kernels, so the
    branch that splits and recurses had never run. It was written this session, from reading numpy's
    C, and a wrong `(n / 2) - (n / 2) % 8` would have gone unnoticed.
  - `default_weights` — the `n >= 25` ramp. Every case stops short of 25 observations in the *below*
    group, so the ramp arm never ran either.
  - `pick_category` — the last cumulative weight pinned to 1, and the strict `<`.

The corpora below are chosen so a *left fold* disagrees with numpy: a large value followed by many
small ones loses the small ones to rounding in one order and not the other. A corpus of equal values
would agree either way and pin nothing.

    .venv/bin/python scripts/generate_parzen_arithmetic_fixture.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from optuna.samplers._tpe.sampler import default_weights

OUT = pathlib.Path(__file__).parent.parent / "crates" / "dsrust-tpe" / "tests" / "conformance"

#: Lengths straddling every branch: under 8, the 8-accumulator path, the 128 block boundary, and
#: past it where numpy splits and recurses.
LENGTHS = [1, 2, 7, 8, 9, 15, 16, 17, 63, 127, 128, 129, 200, 255, 256, 1000]


def corpus(n: int) -> list[float]:
    """Values whose sum depends on the order they are added in.

    One large leading value and many small ones: added left to right the small ones vanish into the
    large one's exponent, added pairwise they accumulate among themselves first. A uniform corpus
    would sum identically whatever the order and would pin nothing.
    """
    values = [1e16 if index == 0 else 1.0 + index * 1e-9 for index in range(n)]
    # A second shape, so the fixture does not rest on one arrangement.
    if n > 3:
        values[n // 2] = -1e15
    return values


def main() -> None:
    sums = []
    for n in LENGTHS:
        values = corpus(n)
        numpy_sum = float(np.array(values, dtype=np.float64).sum())
        left_fold = 0.0
        for value in values:
            left_fold += value
        sums.append(
            {
                "n": n,
                "values": values,
                "sum": repr(numpy_sum),
                # Recorded so the test can assert the corpus actually discriminates: where these two
                # agree, the case proves nothing about summation order.
                "left_fold": repr(left_fold),
                "order_matters": numpy_sum != left_fold,
            }
        )

    weights = [
        {"n": n, "weights": [repr(float(w)) for w in default_weights(n)]}
        # 29 is the smallest n whose ramp endpoint does *not* compute to exactly 1.0, so it is
        # the only length that shows `linspace` assigning the endpoint rather than arriving
        # at it. Without it the pin is unobservable and a port that dropped it would pass.
        for n in [0, 1, 2, 24, 25, 26, 29, 30, 50, 100]
    ]

    fixture = {
        "source": f"generated from numpy=={np.__version__} via scripts/generate_parzen_arithmetic_fixture.py",
        "numpy_version": np.__version__,
        "note": (
            "numpy's add.reduce is pairwise past eight elements and recurses past 128, and optuna's "
            "default_weights ramps past 25 observations. Both were reached by mutation testing "
            "before any test reached them."
        ),
        "sums": sums,
        "default_weights": weights,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "parzen_arithmetic.json"
    path.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"  wrote {path.relative_to(OUT.parent.parent.parent)}", file=sys.stderr)

    # `linspace` *assigns* its endpoint, so optuna's own output cannot show the difference — what
    # shows it is computing the last point the arithmetic way and finding it is not one. At least
    # one recorded length must be like that, or a port that skipped the assignment would pass.
    def computed_endpoint(n: int) -> float:
        ramp = n - 25
        start = 1.0 / n
        return (ramp - 1) * ((1.0 - start) / (ramp - 1)) + start

    shows_assignment = [
        w["n"]
        for w in weights
        if w["n"] > 26 and computed_endpoint(w["n"]) != 1.0
    ]
    if not shows_assignment:
        raise SystemExit("no recorded length shows linspace assigning rather than computing its endpoint")
    print(f"  endpoint assignment observable at n = {shows_assignment}", file=sys.stderr)

    discriminating = [case["n"] for case in sums if case["order_matters"]]
    if not any(n > 128 for n in discriminating):
        raise SystemExit("no case past the 128 block boundary distinguishes the summation order")
    if not any(n < 8 for n in discriminating) and len(discriminating) < 4:
        raise SystemExit("too few cases distinguish the summation order")
    print(
        f"  {len(discriminating)} of {len(sums)} lengths distinguish pairwise from a left fold: "
        f"{discriminating}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
