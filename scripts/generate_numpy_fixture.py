"""Record numpy's MT19937 draws, so the `tpe` crate's generator can be held to them.

optuna's `TPESampler` draws from a seeded `numpy.random.RandomState` (MT19937) through two methods,
`random_sample` and `choice`. The `tpe` crate reproduces that generator; this captures what numpy
actually produces so the reproduction is verified against the library, not against a transcription.

Doubles are stored as little-endian IEEE-754 u64 bit patterns, not decimals: the draws must agree to
the last bit, and a decimal round-trip through JSON can shift that bit under a parser that rounds
differently than Python's `repr`.

    .dspy-venv/bin/python scripts/generate_numpy_fixture.py
"""

from __future__ import annotations

import json
import pathlib
import struct
import sys

import numpy as np

OUT = pathlib.Path(__file__).parent.parent / "tpe" / "tests" / "conformance" / "numpy_mt19937.json"

SEEDS = [0, 1, 42, 12345]
# (seed, n, p, size) for numpy `RandomState(seed).choice(n, p=p, size=size)`.
CHOICE_CASES = [
    (42, 4, [0.1, 0.2, 0.3, 0.4], 10),
    (7, 3, [0.5, 0.3, 0.2], 8),
    (1, 2, [0.25, 0.75], 6),
]


def bits(value: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", float(value)))[0]


def state_key(seed: int, count: int) -> list[int]:
    return [int(word) for word in np.random.RandomState(seed).get_state()[1][:count]]


def sample_bits(seed: int, n: int) -> list[int]:
    generator = np.random.RandomState(seed)
    return [bits(generator.random_sample()) for _ in range(n)]


def choice_out(seed: int, n: int, p: list[float], size: int) -> list[int]:
    return [int(index) for index in np.random.RandomState(seed).choice(n, p=p, size=size)]


def main() -> None:
    fixture = {
        "source": f"numpy {np.__version__} via scripts/generate_numpy_fixture.py; doubles stored as little-endian IEEE-754 u64 bits for exact comparison",
        "numpy_version": np.__version__,
        "seed_key": {str(seed): state_key(seed, 8) for seed in SEEDS},
        "random_sample_bits": {str(seed): sample_bits(seed, 12) for seed in SEEDS},
        "choice": [
            {"seed": seed, "n": n, "p": p, "size": size, "out": choice_out(seed, n, p, size)}
            for seed, n, p, size in CHOICE_CASES
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"  wrote {OUT.relative_to(OUT.parent.parent.parent)}", file=sys.stderr)


if __name__ == "__main__":
    main()
