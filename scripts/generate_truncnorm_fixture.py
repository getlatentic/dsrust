"""optuna's truncated-normal numerics, recorded by calling them.

`BootstrapFewShotWithOptuna` asks optuna for `suggest_int`, and optuna's default TPE sampler treats
an integer parameter as a *discrete truncated normal* rather than as a category — so reproducing
its search means reproducing this arithmetic, not just its shape.

optuna vendors the whole chain rather than depending on SciPy: `_erf` is FreeBSD's `s_erf.c`, a
rational approximation over four intervals, and `_truncnorm` is SciPy's truncated normal reduced to
what TPE needs. Every function here is pure, so each is recorded over a grid chosen to cross every
branch its source has — the erf interval boundaries at 0.84375, 1.25 and 2.857, `_log_ndtr`'s
switch at 6 and -20, and `_ndtri_exp`'s at -1e-2 and -5.

    .venv/bin/python scripts/generate_truncnorm_fixture.py
"""

from __future__ import annotations

import json
import logging
import pathlib
import warnings

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

import math

import numpy as np
from optuna.samplers._tpe import _truncnorm
from optuna.samplers._tpe._erf import erf

from pins import require

PINNED = require("optuna")
OUT = (
    pathlib.Path(__file__).parent.parent
    / "crates"
    / "dsrust-tpe"
    / "tests"
    / "conformance"
    / "truncnorm.json"
)

#: Inputs chosen to land on both sides of every branch the sources take, plus enough ordinary
#: values that a wrong polynomial cannot hide between them.
GRID = [
    0.0,
    1e-30,
    2**-29,
    2**-28,
    1e-8,
    0.1,
    0.5,
    0.84374,
    0.84375,
    0.84376,
    1.0,
    1.24999,
    1.25,
    1.25001,
    2.0,
    2.856,
    2.857142857142857,
    2.858,
    4.0,
    5.9,
    6.0,
    6.1,
    8.0,
    19.9,
    20.0,
    20.1,
    27.0,
    28.0,
    30.0,
]
GRID = sorted({value for x in GRID for value in (x, -x)})


def encoded(value: float) -> float | str:
    """JSON has no spelling for an infinity, and `json.dumps` writes one anyway.

    A tag rather than `null`: these are real answers — `log_gauss_mass(0, 0)` is `-inf` because the
    interval has no width — and a reader that saw `null` could not tell them from a missing field.
    """
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return value


def main() -> None:
    grid = np.array(GRID, dtype=float)
    record = {
        "source": f"generated from optuna=={PINNED} via scripts/generate_truncnorm_fixture.py",
        "grid": GRID,
        "erf": [encoded(v) for v in erf(grid)],
        "ndtr": [encoded(v) for v in _truncnorm._ndtr(grid)],
        "log_ndtr": [encoded(v) for v in _truncnorm._log_ndtr(grid)],
        "norm_logpdf": [encoded(v) for v in _truncnorm._norm_logpdf(grid)],
    }

    # `_log_gauss_mass(a, b)` over pairs that reach all three of its cases: both bounds below zero,
    # both above, and an interval straddling it.
    pairs = [
        (-3.0, -1.0),
        (-20.0, -19.0),
        (1.0, 3.0),
        (19.0, 20.0),
        (-1.0, 1.0),
        (-0.5, 0.5),
        (-30.0, 30.0),
        (-0.5, 0.5000001),
        (0.0, 0.0),
        (-2.5, 0.0),
        (0.0, 2.5),
    ]
    record["log_gauss_mass"] = [
        {"a": a, "b": b, "value": encoded(float(_truncnorm._log_gauss_mass(np.array([a]), np.array([b]))[0]))}
        for a, b in pairs
    ]

    # `_ndtri_exp` inverts `log_ndtr`, and its own three regimes are picked by how close the input
    # is to zero. Its Newton loop is the part a port gets wrong quietly.
    ndtri_inputs = [
        -1e-12,
        -1e-3,
        -0.01,
        -0.05,
        -0.5,
        -0.693,
        -1.0,
        -4.9,
        -5.0,
        -5.1,
        -20.0,
        -100.0,
        -400.0,
    ]
    record["ndtri_exp"] = {
        "inputs": ndtri_inputs,
        "values": [encoded(float(v)) for v in _truncnorm._ndtri_exp(np.array(ndtri_inputs, dtype=float))],
    }

    # `ppf(q, a, b)`, the quantile the sampler actually draws through.
    ppf_cases = [
        (0.5, -1.0, 1.0),
        (0.001, -1.0, 1.0),
        (0.999, -1.0, 1.0),
        (0.5, -np.inf, np.inf),
        (0.25, -0.5, 3.5),
        (0.75, -0.5, 3.5),
        (1e-6, 0.0, 10.0),
        (1 - 1e-6, 0.0, 10.0),
        (0.5, 2.0, 5.0),
        (0.5, -5.0, -2.0),
    ]
    record["ppf"] = [
        {
            "q": q,
            "a": None if np.isinf(a) else a,
            "b": None if np.isinf(b) else b,
            "value": encoded(
                float(_truncnorm.ppf(np.array([q]), np.array([a]), np.array([b]))[0])
            ),
        }
        for q, a, b in ppf_cases
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=2) + "\n")
    print(f"  grid points     : {len(GRID)}")
    print(f"  log_gauss_mass  : {len(record['log_gauss_mass'])}")
    print(f"  ndtri_exp       : {len(ndtri_inputs)}")
    print(f"  ppf             : {len(record['ppf'])}")
    print(f"wrote {OUT.relative_to(pathlib.Path(__file__).parent.parent)}")


if __name__ == "__main__":
    main()
