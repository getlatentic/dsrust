"""Record CPython's own answers for every derived `random.Random` method this crate reproduces.

A mutation run over `pyrng` left 16 survivors, and they were not edge cases: `random()`, `randint()`
and `choice_index()` could each be replaced by a **constant** without a single test failing. The
crate's faithfulness tests all live in `dsrust`, `dsrust-tpe` and `dsrust-gepa`, so the raw
Mersenne-Twister stream was pinned and everything built on top of it was not — and `pyrng` is
published on its own, so a consumer of it alone had nothing.

What matters here is not only the values but **how far each call advances the generator**. A method
that returns the right number after consuming two words where CPython consumes one is correct once
and wrong forever after, which is why every case records a trailing `random()` and why the sequences
below interleave methods rather than testing each from a fresh seed.

    .venv/bin/python scripts/generate_pyrng_methods_fixture.py
"""

from __future__ import annotations

import json
import pathlib
import random
import sys

OUT = pathlib.Path(__file__).parent.parent / "crates" / "pyrng" / "tests" / "conformance"

SEEDS = [0, 1, 9, 42, 2024]


def draws(seed: int) -> dict:
    """A run of every method against one generator, in a fixed order.

    Interleaved deliberately: each call inherits where the last one left the stream, so a method
    that consumes the wrong number of words shows up in everything after it rather than in itself.
    """
    rng = random.Random(seed)
    out: dict = {"seed": seed, "steps": []}

    def step(name: str, value) -> None:
        out["steps"].append({"call": name, "value": value})

    step("random", repr(rng.random()))
    step("random", repr(rng.random()))
    # `randint` over ranges that straddle _randbelow's power-of-two shortcut: 4 is exact, 5 and 6
    # are not, and a wide range exercises the multi-word path.
    for low, high in [(0, 3), (0, 4), (1, 6), (0, 1_000_000_000), (5, 5)]:
        step(f"randint {low} {high}", rng.randint(low, high))
    # `choice` is `_randbelow(len(seq))`, which consumes differently from `random()`.
    for size in [2, 3, 5, 8, 16, 17]:
        step(f"choice_index {size}", rng.choice(range(size)))
    step("random", repr(rng.random()))
    # `choices` draws k times through `random() * total` and a bisect, so it consumes k words.
    # A zero weight is legal and must never be chosen — the entry shares its predecessor's
    # cumulative bound, and `random() < 1` keeps the target strictly below the total. All-zero
    # weights are not legal: CPython raises, so there is nothing to record for them.
    for weights, k in [
        ([1.0, 1.0, 1.0], 3),
        ([0.1, 0.7, 0.2], 4),
        ([5.0, 1.0], 2),
        ([0.5, 0.5, 0.0], 6),
        ([0.0, 0.5, 0.5], 4),
    ]:
        step(f"choices {json.dumps(weights, separators=(chr(44), chr(58)))} {k}", rng.choices(range(len(weights)), weights, k=k))
    step("random", repr(rng.random()))
    # `sample` uses either a selection set or a pool copy depending on k against setsize, so both
    # arms need a case: k small against n large takes the set, k close to n takes the pool.
    for n, k in [(10, 2), (10, 9), (100, 3), (100, 60), (5, 5), (30, 1)]:
        step(f"sample {n} {k}", rng.sample(range(n), k))
    step("random", repr(rng.random()))
    step("shuffle 12", shuffled(rng, 12))
    step("random", repr(rng.random()))
    return out


def shuffled(rng: random.Random, n: int) -> list[int]:
    items = list(range(n))
    rng.shuffle(items)
    return items


def main() -> None:
    fixture = {
        "source": f"generated from CPython {sys.version.split()[0]} via scripts/generate_pyrng_methods_fixture.py",
        "python_version": sys.version.split()[0],
        "note": (
            "Every derived Random method, interleaved against one generator per seed so that how "
            "far each call advances the stream is pinned as well as what it returns. A method that "
            "returns the right value after consuming the wrong number of words is correct once and "
            "wrong for the rest of the run."
        ),
        "runs": [draws(seed) for seed in SEEDS],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "random_methods.json"
    path.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"  wrote {path.relative_to(OUT.parent.parent.parent)}", file=sys.stderr)

    # Two runs that never differ would pass against a generator that ignores its seed.
    first, second = fixture["runs"][0]["steps"], fixture["runs"][1]["steps"]
    differing = sum(1 for a, b in zip(first, second) if a["value"] != b["value"])
    if differing < len(first) // 2:
        raise SystemExit(f"only {differing} of {len(first)} steps differ between seeds")
    print(
        f"  {len(first)} steps per run, {differing} of them differ between the first two seeds",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
