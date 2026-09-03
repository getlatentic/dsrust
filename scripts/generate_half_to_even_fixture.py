"""`np.round`, which is half-to-even and not the rounding Rust's `f64::round` does.

optuna snaps a drawn float back onto the integer grid with
`np.clip(low + np.round((v - low) / step) * step, low, high)`, and its `RandomSampler` untransforms
a uniform the same way. A value landing exactly halfway between two integers therefore decides a
trial, and the two rules disagree on every such value: `np.round(0.5)` is `0` and Rust's is `1`.

Recorded rather than reasoned, over the halves themselves and the floats either side of them —
`0.49999999999999994` is the largest double below a half, and rounding it up would be wrong in the
other direction.

    .venv/bin/python scripts/generate_half_to_even_fixture.py
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
    / "half_to_even.json"
)

VALUES = [
    -3.0,
    -2.5,
    -1.5,
    -0.5,
    -0.49999999999999994,
    -0.1,
    0.0,
    0.1,
    0.49999999999999994,
    0.5,
    0.5000000000000001,
    1.4999999999999998,
    1.5,
    2.0,
    2.5,
    3.5,
    4.5,
    1e15 + 0.5,
]


def main() -> None:
    rows = [{"value": value, "rounded": float(np.round(value))} for value in VALUES]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "source": f"generated from numpy=={observed('numpy')} via "
                "scripts/generate_half_to_even_fixture.py",
                "note": "`np.round` is half-to-even; Rust's `f64::round` is half-away-from-zero.",
                "cases": rows,
            },
            indent=2,
        )
        + "\n"
    )
    away = sum(
        1
        for row in rows
        if row["rounded"] != float(np.sign(row["value"]) * np.floor(abs(row["value"]) + 0.5))
    )
    print(f"  cases: {len(rows)}")
    print(f"  of which half-away-from-zero gets wrong: {away}")
    print(f"wrote {OUT.relative_to(pathlib.Path(__file__).parent.parent)}")


if __name__ == "__main__":
    main()
