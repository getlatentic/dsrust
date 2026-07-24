"""Generate the CPython `random.Random` golden by running CPython itself.

dspy seeds `random.Random(0)` and the demos an optimizer keeps are whichever examples that
generator draws. Matching it is what lets a compiled program be compared to dspy's demo for
demo, so the port in `src/optimize/rng.rs` is held to this fixture rather than to a description
of Mersenne Twister.

Populations are integer ranges, so the fixture says which *indices* were drawn and carries no
Python data model across the boundary.

    .dspy-venv/bin/python scripts/generate_rng_fixture.py

Coverage is swept rather than sampled: the cases below are built from the boundaries each
algorithm actually turns on — every bit width, every rejection-loop shape, and the population
size at which `sample` changes strategy — because a handful of chosen midpoints proved able to
miss a whole branch.
"""

from __future__ import annotations

import json
import pathlib
import platform
import random
import sys
from math import ceil, log

OUT = pathlib.Path(__file__).parent.parent / "pyrng" / "tests" / "conformance"

# The canonical `mt19937ar.c` demonstration: `init_by_array` over this key, then a thousand
# consecutive draws. Its published output is the reference every Mersenne Twister is checked
# against, and CPython reaches the same key from this integer seed.
REFERENCE_KEY = [0x123, 0x234, 0x345, 0x456]
REFERENCE_DRAWS = 1000

SEEDS = [0, 1, 42]

# Every width, so the word boundary at 32 and the second word at 33 are both covered.
GETRANDBITS_WIDTHS = list(range(1, 65))

# `_randbelow` draws `bound.bit_length()` bits and redraws until it lands under the bound, so
# how much it wastes changes at each power of two. Walk the small bounds and both sides of the
# powers: a bound just over one is nearly always a redraw, a bound just under is nearly never.
RANDBELOW_BOUNDS = sorted(
    set(range(1, 40)) | {n + d for n in (64, 128, 256, 1024) for d in (-1, 0, 1)}
)

SHUFFLE_SIZES = [0, 1, 2, 3, 4, 5, 8, 13, 21, 34, 55]

# `random.choices` draws through `random()` and bisects the cumulative weights. The weight
# shapes below cover a flat distribution (every draw equally likely), a skewed one (a bisect
# that lands late), a single-element population (the draw must still return index 0), and a zero
# weight (an element the bisect can never pick), so the accumulation and the `hi` clamp are both
# exercised.
CHOICES_WEIGHTS = [
    [1.0],
    [1.0, 1.0, 1.0, 1.0],
    [0.1, 0.2, 0.3, 0.4],
    [5.0, 0.0, 1.0],
    [0.01, 100.0, 0.01],
    [2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5],
]


def setsize(k: int) -> int:
    """CPython's threshold for tracking a pool instead of a set of drawn indices."""
    return 21 + (4 ** ceil(log(k * 3, 4)) if k > 5 else 0)


def integer_setsize(k: int) -> int:
    """The same threshold reached without floating point, which is how the Rust port computes it.

    `4 ** ceil(log(k * 3, 4))` is the smallest power of four at or above `k * 3`. A loop can only
    disagree with the rounding when `k * 3` is itself a power of four, and it never is: every
    power of four is one more than a multiple of three, so none is divisible by three. `main`
    asserts the two agree rather than leaving that as an argument.
    """
    table = 1
    while table < k * 3:
        table *= 4
    return 21 + (table if k > 5 else 0)


def sample_cases() -> list[tuple[int, int, int]]:
    """(seed, population, k) covering both branches and the exact size they switch at.

    `sample` keeps a pool when the population is small and a set of drawn indices when it is
    large, and the two consume different numbers of draws. The switch is what decides which
    answer comes out, so every `k` here is tried at the threshold and on both sides of it.
    """
    cases = []
    for k in (0, 1, 2, 5, 6, 7, 12, 21, 22, 40):
        edge = setsize(k)
        sizes = {k, k + 1, edge - 1, edge, edge + 1, edge * 2}
        for size in sorted(size for size in sizes if k <= size <= 4096):
            for seed in SEEDS:
                cases.append((seed, size, k))
    return cases


def compact(fixture: dict) -> str:
    """One case per line: diffable without giving every integer its own line."""
    blocks = []
    for section, value in fixture.items():
        if not isinstance(value, list):
            blocks.append(f"  {json.dumps(section)}: {json.dumps(value)}")
            continue
        body = ",\n".join(f"    {json.dumps(case)}" for case in value)
        blocks.append(f"  {json.dumps(section)}: [\n{body}\n  ]")
    return "{\n" + ",\n".join(blocks) + "\n}\n"


def shuffled(seed: int, size: int) -> list[int]:
    items = list(range(size))
    random.Random(seed).shuffle(items)
    return items


def main() -> None:
    for k in range(6, 2000):
        if setsize(k) != integer_setsize(k):
            raise SystemExit(f"threshold disagrees at k={k}: {setsize(k)} vs {integer_setsize(k)}")

    reference_seed = sum(word << (32 * at) for at, word in enumerate(REFERENCE_KEY))
    reference = random.Random(reference_seed)

    fixture = {
        "source": "generated from CPython via scripts/generate_rng_fixture.py",
        "python_version": platform.python_version(),
        "reference": {
            "key": REFERENCE_KEY,
            "note": "canonical mt19937ar.c output; cross-checked against an independent "
            "transcription of the 2002 reference implementation",
            "draws": [str(reference.getrandbits(32)) for _ in range(REFERENCE_DRAWS)],
        },
        "getrandbits": [
            {
                "seed": seed,
                "bits": bits,
                "draws": [str(generator.getrandbits(bits)) for _ in range(4)],
            }
            for seed in SEEDS
            for bits in GETRANDBITS_WIDTHS
            for generator in [random.Random(seed)]
        ],
        "randbelow": [
            {
                "seed": seed,
                "bound": bound,
                "draws": [generator._randbelow(bound) for _ in range(6)],
            }
            for seed in SEEDS
            for bound in RANDBELOW_BOUNDS
            for generator in [random.Random(seed)]
        ],
        "shuffle": [
            {"seed": seed, "population": size, "result": shuffled(seed, size)}
            for seed in SEEDS
            for size in SHUFFLE_SIZES
        ],
        "sample": [
            {
                "seed": seed,
                "population": size,
                "k": k,
                "setsize": setsize(k),
                "result": random.Random(seed).sample(range(size), k),
            }
            for seed, size, k in sample_cases()
        ],
        "choices": [
            {
                "seed": seed,
                "weights": weights,
                "k": 8,
                # `population` is the indices, so the result is the drawn indices directly.
                "result": random.Random(seed).choices(range(len(weights)), weights=weights, k=8),
            }
            for seed in SEEDS
            for weights in CHOICES_WEIGHTS
        ],
    }

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "cpython_random.json"
    path.write_text(compact(fixture))
    counts = {name: len(fixture[name]) for name in ("getrandbits", "randbelow", "shuffle", "sample")}
    print(f"  wrote {path.relative_to(OUT.parent.parent.parent)}", file=sys.stderr)
    print(f"  {counts}, reference draws: {REFERENCE_DRAWS}", file=sys.stderr)


if __name__ == "__main__":
    main()
