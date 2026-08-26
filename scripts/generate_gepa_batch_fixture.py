"""Record the minibatches GEPA's EpochShuffledBatchSampler hands out, by running it.

The sampler shuffles the trainset ids each epoch (random.shuffle), pads to a whole number of
minibatches with the least-frequent id, and returns consecutive slices as iterations advance. It is
deterministic under a seeded random.Random, so what is compared is the id list for each iteration —
which pins the shuffle, the padding tie-break, and the epoch rollover together.

    .dspy-venv/bin/python scripts/generate_gepa_batch_fixture.py
"""

from __future__ import annotations

import json
import pathlib
import random
import sys

from gepa.strategies.batch_sampler import EpochShuffledBatchSampler

from pins import require

OUT = pathlib.Path(__file__).parent.parent / "crates" / "dsrust-gepa" / "tests" / "conformance"
PINNED = require("gepa")


class Loader:
    """The minimal DataLoader the sampler needs: ids `0..n` and a length."""

    def __init__(self, n: int):
        self.n = n

    def all_ids(self):
        return list(range(self.n))

    def __len__(self):
        return self.n


class State:
    def __init__(self):
        self.i = 0


# (trainset_size, minibatch_size, seed, iterations). Sizes chosen to force every branch: exact fit,
# a one-short pad, a two-short pad, and enough iterations to roll several epochs.
CASES = [
    (6, 2, 0, 10),
    (7, 3, 1, 12),   # 7 % 3 = 1 -> pad 2
    (5, 2, 7, 14),   # 5 % 2 = 1 -> pad 1
    (10, 4, 9, 15),  # 10 % 4 = 2 -> pad 2
    (8, 8, 3, 6),    # single minibatch, multiple epochs
]


def build_once(n: int, mb: int, seed: int, iters: int) -> dict:
    sampler = EpochShuffledBatchSampler(minibatch_size=mb, rng=random.Random(seed))
    loader, state = Loader(n), State()
    batches = []
    for i in range(iters):
        state.i = i
        batches.append(list(sampler.next_minibatch_ids(loader, state)))
    return {"trainset_size": n, "minibatch_size": mb, "seed": seed, "batches": batches}


def main() -> None:
    fixture = {
        "source": f"generated from gepa=={PINNED} via scripts/generate_gepa_batch_fixture.py",
        "gepa_version": PINNED,
        "cases": [build_once(*case) for case in CASES],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "batch.json"
    path.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"  wrote {path.relative_to(OUT.parent.parent)}", file=sys.stderr)


if __name__ == "__main__":
    main()
