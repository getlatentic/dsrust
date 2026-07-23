"""Record which candidate GEPA's Pareto selection picks, by running the real gepa package.

gepa==0.0.27's `select_program_candidate_from_pareto_front` drops dominated programs, builds a
frequency-weighted sampling list, and draws from it with a seeded `random.Random`. It is a pure
function of the front, the scores and the seed, so what is compared is the program index it returns
across many fronts and seeds — which pins the domination sweep, the ascending set/dict ordering, and
the CPython `choice` draw all at once.

    .dspy-venv/bin/python scripts/generate_gepa_pareto_fixture.py
"""

from __future__ import annotations

import json
import pathlib
import random
import sys

from gepa.gepa_utils import select_program_candidate_from_pareto_front

from pins import require

OUT = pathlib.Path(__file__).parent.parent / "gepa" / "tests" / "conformance"
PINNED = require("gepa")

# (fronts as {testcase: [program indices]}, weighted aggregate scores per program).
CASES = [
    ([[0, 1], [1, 2], [0, 2]], [0.5, 0.7, 0.3]),
    ([[0], [0, 1], [0, 1, 2]], [0.9, 0.5, 0.1]),          # program 0 wins testcase 0 alone
    ([[0, 1, 2], [0, 1, 2], [0, 1, 2]], [0.4, 0.4, 0.4]),  # all tied, all on every front
    ([[1], [1, 3], [2, 3], [0, 1, 2, 3]], [0.2, 0.8, 0.5, 0.6]),
    ([[0, 2], [2], [0, 2, 4], [4]], [0.1, 0.0, 0.3, 0.0, 0.9]),  # sparse indices
    ([[3]], [0.0, 0.0, 0.0, 1.0]),                          # single winner
]
SEEDS = list(range(20))


def build_once(fronts: list[list[int]], scores: list[float]) -> dict:
    picks = []
    for seed in SEEDS:
        mapping = {tc: set(front) for tc, front in enumerate(fronts)}
        chosen = select_program_candidate_from_pareto_front(mapping, scores, random.Random(seed))
        picks.append(chosen)
    return {"fronts": fronts, "scores": scores, "picks": picks}


def main() -> None:
    fixture = {
        "source": f"generated from gepa=={PINNED} via scripts/generate_gepa_pareto_fixture.py",
        "gepa_version": PINNED,
        "seeds": SEEDS,
        "cases": [build_once(fronts, scores) for fronts, scores in CASES],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "pareto.json"
    path.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"  wrote {path.relative_to(OUT.parent.parent)}", file=sys.stderr)


if __name__ == "__main__":
    main()
