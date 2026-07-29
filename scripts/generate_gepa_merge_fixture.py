"""Record what gepa's merge proposer produces, by driving the real package's own functions.

Merge combines two descendants of a common ancestor that improved *different* components, so every
case here has multi-component candidates that split cleanly — each descendant changed a component
the other left alone. A clean split is also what makes the result independent of the order the
predicate loop walks the components (CPython's `PYTHONHASHSEED`-randomised string-set order): with
no component changed by both descendants, there is no coin-flip draw and the merged value of each
component is determined, so `{a: A3, b: B5}` is the answer whichever order the loop takes. The
generator is still pinned to `PYTHONHASHSEED=0` so the run is reproducible.

Two functions are exercised directly: `sample_and_attempt_merge_programs_by_common_predictors`
(the pair/ancestor draw and the component merge) and `MergeProposer.select_eval_subsample_...`
(the bucketed subsample draw). Both consume the shared generator, so the seed and the exact draw
order are what is being pinned.

    PYTHONHASHSEED=0 .dspy-venv/bin/python scripts/generate_gepa_merge_fixture.py
"""

from __future__ import annotations

import json
import os
import pathlib
import random
import sys

if os.environ.get("PYTHONHASHSEED") != "0":
    raise SystemExit("run with PYTHONHASHSEED=0 so CPython's string-set order is reproducible")

from gepa.proposer.merge import (
    MergeProposer,
    sample_and_attempt_merge_programs_by_common_predictors as attempt_merge,
)

from pins import require

OUT = pathlib.Path(__file__).parent.parent / "crates" / "dsrust-gepa" / "tests" / "conformance"
PINNED = require("gepa")


def merge_case(name, candidates, parents, scores, merge_candidates, seed) -> dict:
    result = attempt_merge(
        agg_scores=scores,
        rng=random.Random(seed),
        merge_candidates=merge_candidates,
        merges_performed=([], []),
        program_candidates=candidates,
        parent_program_for_candidate=parents,
        has_val_support_overlap=None,
        max_attempts=10,
    )
    merged = None
    if result is not None:
        candidate, id1, id2, ancestor = result
        merged = {"candidate": candidate, "id1": id1, "id2": id2, "ancestor": ancestor}
    return {
        "name": name,
        "candidates": candidates,
        "parents": [[-1 if p is None else p for p in row] for row in parents],
        "scores": scores,
        "merge_candidates": merge_candidates,
        "seed": seed,
        "merged": merged,
    }


def subsample_case(name, scores1, scores2, num, seed) -> dict:
    proposer = MergeProposer(
        logger=_Silent(),
        valset=None,
        evaluator=None,
        use_merge=True,
        max_merge_invocations=5,
        rng=random.Random(seed),
    )
    s1 = {i: v for i, v in enumerate(scores1)}
    s2 = {i: v for i, v in enumerate(scores2)}
    selected = proposer.select_eval_subsample_for_merged_program(s1, s2, num_subsample_ids=num)
    return {"name": name, "scores1": scores1, "scores2": scores2, "num": num, "seed": seed, "selected": selected}


class _Silent:
    def log(self, *args, **kwargs):
        pass


# Ancestor 0 = {a:A0, b:B0}. Each merge candidate changed exactly one component. Different seeds
# and candidate lists exercise the pair sample and the score-weighted ancestor draw.
SPLIT = [
    {"a": "A0", "b": "B0"},  # 0 ancestor / seed
    {"a": "A1", "b": "B0"},  # 1 changed a
    {"a": "A0", "b": "B2"},  # 2 changed b
    {"a": "A3", "b": "B0"},  # 3 changed a
    {"a": "A0", "b": "B0"},  # 4 unchanged (dominated)
    {"a": "A0", "b": "B5"},  # 5 changed b
    {"a": "A6", "b": "B0"},  # 6 changed a
]
SPLIT_PARENTS = [[None], [0], [0], [0], [0], [0], [0]]


def main() -> None:
    scores = [0.30, 0.55, 0.50, 0.60, 0.20, 0.70, 0.45]
    merges = [
        merge_case("a_and_b", SPLIT, SPLIT_PARENTS, scores, [3, 5, 1, 2], 0),
        merge_case("other_seed", SPLIT, SPLIT_PARENTS, scores, [3, 5, 1, 2], 4),
        merge_case("wider_pool", SPLIT, SPLIT_PARENTS, scores, [1, 2, 3, 5, 6], 7),
        merge_case("two_only", SPLIT, SPLIT_PARENTS, scores, [1, 2], 1),
        merge_case("no_pair", SPLIT, SPLIT_PARENTS, scores, [1], 0),  # too few to sample
    ]
    subsamples = [
        subsample_case("balanced", [0.9, 0.1, 0.5, 0.5, 0.3, 0.8], [0.1, 0.9, 0.5, 0.5, 0.7, 0.2], 5, 0),
        subsample_case("all_equal", [0.5] * 8, [0.5] * 8, 5, 3),
        subsample_case("id1_dominates", [0.9, 0.8, 0.7, 0.6, 0.5], [0.1, 0.2, 0.3, 0.4, 0.4], 4, 2),
        subsample_case("small_pool", [0.6, 0.4], [0.4, 0.6], 5, 1),
    ]
    fixture = {
        "source": f"generated from gepa=={PINNED} via scripts/generate_gepa_merge_fixture.py "
        "under PYTHONHASHSEED=0",
        "gepa_version": PINNED,
        "merges": merges,
        "subsamples": subsamples,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "merge.json"
    path.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"  wrote {path.relative_to(OUT.parent.parent)} "
          f"({len(merges)} merges, {len(subsamples)} subsamples)", file=sys.stderr)


if __name__ == "__main__":
    main()
