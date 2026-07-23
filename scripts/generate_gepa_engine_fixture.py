"""Record a full GEPA optimization run, by driving the real engine with a scripted adapter.

The engine (`gepa.optimize`) is exercised end to end: candidate selection off the Pareto front, the
epoch-shuffled minibatch, the reflective mutation, the strict-improvement accept test, the full-valset
re-evaluation, and the metric-call budget — all off one seeded `random.Random`. To make the run
deterministic and reproducible in Rust without an LLM, the adapter scores parametrically:

  - a component text is "vN"; its version is N.
  - a candidate's versions are its component versions in sorted-name order.
  - example `id` scores off `versions[id % k]` (the component that example "favors"). With one
    component that is `version * 0.125` — monotonic, a clean chain. With two it is a trade-off,
    `(favored - rival) * 0.125`, so advancing one component wins the examples it favors and *loses*
    the others: siblings that advanced different components are mutually non-dominating (a genuinely
    split Pareto front), and whether a mutation beats its parent on a minibatch depends on which ids
    the shuffle drew — so the run only reproduces if the shared RNG is consumed in GEPA's order
    (candidate-selection `choice`, then the sampler's `shuffle`).
  - reflecting on a component bumps its version by one, capped at `cap` (a capped bump is a no-op, so
    its proposal ties and is rejected — exercising the reject path).

What is compared is the engine's decisions and bookkeeping: the candidate pool it builds, each
candidate's parents and discovery eval-count, the per-candidate mean valset score, the best index, and
the eval totals. The scoring is pure data (the constants and `cap`), so the Rust mirror computes the
same scores and the comparison isolates the engine.

    .dspy-venv/bin/python scripts/generate_gepa_engine_fixture.py
"""

from __future__ import annotations

import json
import pathlib
import sys

from gepa import optimize

from pins import require
from gepa.core.adapter import EvaluationBatch

OUT = pathlib.Path(__file__).parent.parent / "gepa" / "tests" / "conformance"
PINNED = require("gepa")

WEIGHT = 0.125


class ScriptedAdapter:
    """A GEPAAdapter whose scores are a fixed function of candidate component versions."""

    def __init__(self, cap: int):
        self.cap = cap

    @staticmethod
    def _versions(candidate: dict[str, str]) -> list[int]:
        return [int(candidate[name][1:]) for name in sorted(candidate)]

    def _score(self, candidate: dict[str, str], example_id: int) -> float:
        versions = self._versions(candidate)
        k = len(versions)
        favored = versions[example_id % k]
        if k == 1:
            return favored * WEIGHT
        rival = versions[(example_id + 1) % k]
        return (favored - rival) * WEIGHT

    def evaluate(self, batch, candidate, capture_traces=False):
        scores = [self._score(candidate, example_id) for example_id in batch]
        trajectories = [{"version": self._versions(candidate)} for _ in batch] if capture_traces else None
        return EvaluationBatch(outputs=[None] * len(batch), scores=scores, trajectories=trajectories)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        return {c: [{"version": int(candidate[c][1:])}] for c in components_to_update}

    def propose_new_texts(self, candidate, reflective_dataset, components_to_update):
        return {c: f"v{min(int(candidate[c][1:]) + 1, self.cap)}" for c in components_to_update}


# (label, components, cap, trainset_size, valset_size, minibatch_size, max_metric_calls, perfect, seed).
# Varied seeds force distinct selection draws; a small budget forces an early stop; the two-component
# cases split the Pareto front and make acceptance depend on the shuffled minibatch; perfect_score=0.0
# makes the seed already-perfect, exercising the skip-and-still-spend path.
CASES = [
    ("single_seed0", ["instruction"], 4, 5, 6, 2, 40, 1.0, 0),
    ("single_small_budget", ["instruction"], 3, 4, 4, 2, 10, 1.0, 2),
    ("skip_perfect", ["instruction"], 4, 4, 4, 2, 8, 0.0, 3),
    ("two_components_seed1", ["instr_a", "instr_b"], 3, 6, 4, 2, 50, 1.0, 1),
    ("two_components_seed5", ["instr_a", "instr_b"], 4, 5, 4, 3, 60, 1.0, 5),
]


def build_once(
    label, components, cap, trainset_size, valset_size, minibatch_size, max_metric_calls, perfect, seed
) -> dict:
    trainset = list(range(trainset_size))
    valset = list(range(valset_size))
    seed_candidate = {name: "v0" for name in components}

    result = optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=valset,
        adapter=ScriptedAdapter(cap),
        max_metric_calls=max_metric_calls,
        reflection_minibatch_size=minibatch_size,
        perfect_score=perfect,
        seed=seed,
        raise_on_exception=True,
    )

    return {
        "label": label,
        "components": components,
        "cap": cap,
        "trainset_size": trainset_size,
        "valset_size": valset_size,
        "minibatch_size": minibatch_size,
        "max_metric_calls": max_metric_calls,
        "perfect_score": perfect,
        "seed": seed,
        "seed_candidate": seed_candidate,
        "result": {
            "candidates": [dict(c) for c in result.candidates],
            "parents": result.parents,
            "val_aggregate_scores": result.val_aggregate_scores,
            "best_idx": result.best_idx,
            "total_metric_calls": result.total_metric_calls,
            "num_full_val_evals": result.num_full_val_evals,
            "discovery_eval_counts": result.discovery_eval_counts,
        },
    }


def main() -> None:
    fixture = {
        "source": f"generated from gepa=={PINNED} via scripts/generate_gepa_engine_fixture.py",
        "gepa_version": PINNED,
        "weight": WEIGHT,
        "cases": [build_once(*case) for case in CASES],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "engine.json"
    path.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"  wrote {path.relative_to(OUT.parent.parent)}", file=sys.stderr)


if __name__ == "__main__":
    main()
