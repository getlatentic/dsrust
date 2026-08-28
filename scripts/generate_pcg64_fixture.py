"""numpy's `default_rng` — PCG64, not the Mersenne Twister — and the two draws SIMBA takes from it.

`pyrng` already reproduces `np.random.RandomState`, which is MT19937 and is what optuna's TPE uses.
SIMBA reaches for `np.random.default_rng(seed)` instead, and that is a **different generator**:
PCG64, a 128-bit LCG with an output permutation. Nothing already here produces its stream.

It is used for exactly one thing — `rng_np.poisson(num_demos / max_demos)`, which decides how many
demos a candidate drops — but that draw moves every later decision, so a port that guesses it
diverges from the first bucket onward.

Recorded here:

  - the bit generator's own 64-bit words, so a Rust PCG64 can be checked before anything is built
    on it;
  - `random()` doubles, which are `next_u64() >> 11` scaled — the step poisson consumes;
  - `poisson(lam)` streams at the small lambdas SIMBA reaches, where numpy uses the multiplication
    method rather than the transformed-rejection one, and the count is how many uniforms it took
    before the running product fell below `exp(-lam)`.

    .venv/bin/python scripts/generate_pcg64_fixture.py
"""

from __future__ import annotations

import json
import pathlib
import sys
from importlib.metadata import version

import numpy as np

OUT = (
    pathlib.Path(__file__).parent.parent
    / "crates" / "pyrng" / "tests" / "conformance" / "numpy_pcg64.json"
)

SEEDS = [0, 1, 7, 42, 2024]
#: `num_demos / max_demos` for the shapes SIMBA reaches: no demos yet, part-full, exactly full,
#: and over. Lambda is a ratio, so these are the values that actually occur.
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0, 9.999, 10.0, 12.5, 40.0]
#: The transformed-rejection branch above lambda ten accepts on its first test more than nine
#: attempts in ten, so its later arms — the `k < 0` retry, the `us < 0.013` squeeze, and the
#: log-density acceptance that calls `loggam` — are reached only by volume. Twenty-four draws per
#: stream left every one of them unexercised; these lambdas and a longer stream reach them.
LAMBDAS += [10.001, 25.0, 100.0, 1000.0]
#: Draws per stream. Raised from 24 for the reason above: the arms this is here to pin are rare.
DRAWS = 400

#: Streams long enough to reach the three decisions that even four hundred draws miss, each one the
#: *shortest* witness found for a mutant that survived without it — searched over 24 seeds and five
#: lambdas on freshly seeded streams, since a stream shared across lambdas witnesses at offsets no
#: `default_rng(seed)` can reproduce.
#:
#:   (11, 10.5)   draw 1113 — `k` lands exactly on zero, which separates `k < 0` from `k <= 0` and
#:                from `k == 0`. Reachable only near the branch at ten, where the proposal's tail
#:                reaches the origin at all.
#:   (0, 10.0)    draw 2295 — the squeeze's `v > us` decides. It needs `us < 0.013`, roughly one
#:                attempt in eighty, and then `v <= us` on top of that.
#:   (22, 10.001) draw 1963 — the acceptance flips on `loggam`'s second coefficient. It contributes
#:                about `1e-4` to a comparison whose sides are usually orders apart, so this is the
#:                one draw in 350,000 where they are not.
DEEP = [(11, 10.5, 1200), (0, 10.0, 2400), (22, 10.001, 2000)]


def words(seed: int, count: int) -> list[int]:
    """Raw 64-bit words straight off the bit generator, before any distribution."""
    rng = np.random.default_rng(seed)
    return [int(x) for x in rng.bit_generator.random_raw(count)]


def doubles(seed: int, count: int) -> list[float]:
    rng = np.random.default_rng(seed)
    return [float(x) for x in rng.random(count)]


def poissons(seed: int, lam: float, count: int) -> list[int]:
    rng = np.random.default_rng(seed)
    return [int(rng.poisson(lam)) for _ in range(count)]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "source": f"numpy {version('numpy')} np.random.default_rng, via scripts/generate_pcg64_fixture.py",
                "numpy_version": version("numpy"),
                "note": (
                    "PCG64's raw words, its uniform doubles, and `poisson` at the lambdas SIMBA "
                    "reaches. A different generator from `RandomState`, which is MT19937."
                ),
                "raw_words": {str(seed): words(seed, 12) for seed in SEEDS},
                "doubles": {str(seed): doubles(seed, 12) for seed in SEEDS},
                "poisson": [
                    {"seed": seed, "lam": lam, "draws": poissons(seed, lam, count)}
                    for seed, lam, count in DEEP
                ]
                + [
                    {"seed": seed, "lam": lam, "draws": poissons(seed, lam, DRAWS)}
                    for seed in SEEDS
                    for lam in LAMBDAS
                ],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"  wrote {OUT.name}: {len(SEEDS)} seeds, {len(SEEDS) * len(LAMBDAS)} poisson streams", file=sys.stderr)


if __name__ == "__main__":
    main()
