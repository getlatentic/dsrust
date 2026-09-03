"""Two pieces of arithmetic SIMBA leans on that Rust does not spell the same way.

  - **`np.percentile`** with the default `linear` interpolation, which SIMBA takes at 10 and 90 over
    a mini-batch's scores to decide what counts as a bad or a good trajectory. numpy places the
    request at `q/100 * (n - 1)` in the *sorted* array and interpolates between neighbours; there is
    no rounding to an element, so the answer is usually a value that no observation has.
  - **Python's `round`**, which is **banker's rounding**: `round(0.5) == 0` and `round(1.5) == 2`.
    Rust's `f64::round` is half-away-from-zero and gives 1 and 2. SIMBA picks its final candidate
    slate with `round(i * M / (N - 1))`, so a half lands differently and a different program is
    returned.

    .venv/bin/python scripts/generate_simba_arithmetic_fixture.py
"""

from __future__ import annotations

import json
import pathlib
import sys
from importlib.metadata import version

import numpy as np

OUT = (
    pathlib.Path(__file__).parent.parent
    / "crates" / "dsrust" / "tests" / "conformance" / "optimize" / "simba_arithmetic.json"
)

#: Score sets a mini-batch produces: all equal, two values, already sorted, reversed, with
#: duplicates, and a single element — the last is where an off-by-one in the index shows.
SAMPLES = [
    [0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [0.0, 0.25, 0.5, 1.0],
    [1.0, 1.0, 1.0],
    [0.0, 0.0, 1.0, 1.0],
    [0.3, 0.1, 0.4, 0.1, 0.5, 0.9, 0.2, 0.6],
    [0.5] * 7,
    [i / 31 for i in range(32)],
]
QUANTILES = [0.0, 10.0, 25.0, 50.0, 75.0, 90.0, 100.0]

#: Every half, both signs, and the `i * M / (N - 1)` shapes SIMBA's final slate takes.
ROUNDS = [
    -3.5, -2.5, -1.5, -0.5, -0.4, 0.0, 0.4, 0.5, 1.0, 1.5, 2.0, 2.5, 3.5, 4.5,
    0.3333333333333333, 0.6666666666666666, 1.3333333333333333, 2.5000000000000004,
]


def main() -> None:
    slates = []
    for winners in range(0, 6):
        for candidates in (2, 4, 7):
            m, n = winners, candidates + 1
            picked = [0] * n if m < 1 else [round(i * m / (n - 1)) for i in range(n)]
            slates.append(
                {"winners": winners, "num_candidates": candidates,
                 "indices": list(dict.fromkeys(picked))}
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "source": f"numpy {version('numpy')} percentile and CPython's round, "
                "via scripts/generate_simba_arithmetic_fixture.py",
                "numpy_version": version("numpy"),
                "note": (
                    "`np.percentile` at the linear default, and Python's banker's `round`. Rust's "
                    "`f64::round` disagrees with the second on every half."
                ),
                "percentile": [
                    {"sample": sample, "q": q, "value": float(np.percentile(sample, q))}
                    for sample in SAMPLES
                    for q in QUANTILES
                ],
                "round": [{"x": x, "rounded": round(x)} for x in ROUNDS],
                "final_slate": slates,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"  wrote {OUT.name}: {len(SAMPLES) * len(QUANTILES)} percentiles, "
          f"{len(ROUNDS)} rounds, {len(slates)} slates", file=sys.stderr)


if __name__ == "__main__":
    main()
