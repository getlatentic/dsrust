"""Record which candidate gepa's other two selectors pick, by running the real gepa package.

dspy annotates `candidate_selection_strategy` as `Literal["pareto", "current_best"]` and then passes
the string straight to `gepa.optimize`, whose factory map also holds `epsilon_greedy` (eps=0.1) and
`top_k_pareto` (k=5). Both are reachable from a dspy call and neither was built here.

Each is recorded across many seeds because each *draws differently*, and how far the shared generator
advances is what a later round inherits:

  - `EpsilonGreedyCandidateSelector` always draws the coin, and draws a second time only when the
    coin lands below epsilon. So one round takes one draw and another takes two.
  - `TopKParetoCandidateSelector` restricts the fronts to the top `k` by aggregate score — a *set
    intersection*, whose iteration order reaches the sampling list — and falls through to `idxmax`
    with no draw at all when the filtered mapping is empty.

The cases include tied scores at indices that collide in CPython's set table, for the reason the
pareto fixture already gives: a port written over `sorted` passes small-index cases by coincidence.

    .venv/bin/python scripts/generate_gepa_selector_fixture.py
"""

from __future__ import annotations

import json
import pathlib
import random
import sys

from gepa.strategies.candidate_selector import (
    CurrentBestCandidateSelector,
    EpsilonGreedyCandidateSelector,
    ParetoCandidateSelector,
    TopKParetoCandidateSelector,
)
from gepa.strategies.component_selector import (
    AllReflectionComponentSelector,
    RoundRobinReflectionComponentSelector,
)

from pins import require

OUT = pathlib.Path(__file__).parent.parent / "crates" / "dsrust-gepa" / "tests" / "conformance"
PINNED = require("gepa")

#: (fronts as lists of program indices, aggregate score per program). The last two carry indices
#: past 7 with every score tied, so the draw turns entirely on the set order — the regime where a
#: `sorted` port diverges rather than agreeing by accident.
_TIED_16 = [0.5] * 16
CASES = [
    ([[0, 1], [1, 2], [0, 2]], [0.5, 0.7, 0.3]),
    ([[0], [0, 1], [0, 1, 2]], [0.9, 0.5, 0.1]),
    ([[0, 1, 2], [0, 1, 2], [0, 1, 2]], [0.4, 0.4, 0.4]),
    ([[1], [1, 3], [2, 3], [0, 1, 2, 3]], [0.2, 0.8, 0.5, 0.6]),
    ([[0, 2], [2], [0, 2, 4], [4]], [0.1, 0.0, 0.3, 0.0, 0.9]),
    # More programs than k=5, so top-k genuinely filters and some fronts empty out.
    ([[6, 7], [0, 1], [2, 3, 8], [9]], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
    ([[3, 8, 5, 4], [8, 2], [15, 3]], _TIED_16),
    ([[1, 9, 11], [8], [15, 6, 0]], _TIED_16),
]
#: Seeds 0-9 land *above* epsilon and take the `idxmax` branch; 31, 32, 43, 49 and 55 land below it
#: and take the uniform draw. Both are needed: a first pass used `range(20)`, every seed of which
#: takes the same branch, so the fixture agreed with a port that ignored the coin entirely.
SEEDS = list(range(10)) + [31, 32, 43, 49, 55]

#: The coin landing *exactly on* epsilon, which no fixed epsilon reaches by chance. `random()`
#: returns a multiple of 2**-53, so an epsilon set to a seed's own first draw makes
#: `rng.random() < self.epsilon` an exact tie — the one shape separating `<` from `<=`. gepa takes
#: the `idxmax` branch and a `<=` port takes the uniform draw, at a different index and one draw
#: further along.
TIE_SEEDS = [0, 3, 5, 8]
TIE_SCORES = [0.5, 0.1, 0.9, 0.4, 0.7, 0.2]


class ComponentState:
    """The two attributes a component selector reads: the component list and the per-candidate
    cursor it advances."""

    def __init__(self, names, cursor):
        self.list_of_named_predictors = names
        self.named_predictor_id_to_update_next_for_program_candidate = {0: cursor}


class State:
    """The two attributes each selector reads, which is all of `GEPAState` they touch."""

    def __init__(self, fronts, scores):
        self._fronts = fronts
        self.per_program_tracked_scores = scores
        self.program_full_scores_val_set = scores
        self.program_candidates = list(range(len(scores)))

    def get_pareto_front_mapping(self):
        return {index: set(front) for index, front in enumerate(self._fronts)}


def picks(make, fronts, scores):
    """The program each seed selects, and where the generator was left afterwards.

    `after` is the next `random()` once the selection is made, which is the whole point for these
    two: epsilon-greedy takes one draw or two depending on the coin, and top-k-pareto takes none at
    all when the filtered mapping empties. A port that agrees on the selection and advances the
    generator differently diverges on the *next* round, and only this catches it.
    """
    chosen, after = [], []
    for seed in SEEDS:
        rng = random.Random(seed)
        selector = make(rng)
        chosen.append(selector.select_candidate_idx(State(fronts, scores)))
        after.append(rng.random())
    return chosen, after


def epsilon_ties() -> list[dict]:
    """Each tie seed's pick, with epsilon set to that seed's own first draw.

    Recorded with the index a `<=` port would land on instead, so the case cannot quietly stop
    discriminating: `main` refuses to write a fixture where the two agree.
    """
    ties = []
    for seed in TIE_SEEDS:
        epsilon = random.Random(seed).random()
        rng = random.Random(seed)
        selector = EpsilonGreedyCandidateSelector(epsilon=epsilon, rng=rng)
        pick = selector.select_candidate_idx(State([], TIE_SCORES))

        # What the `<=` spelling would do: the same coin, then the uniform draw it takes instead.
        other = random.Random(seed)
        other.random()
        ties.append(
            {
                "seed": seed,
                "epsilon": epsilon,
                "pick": pick,
                "after": rng.random(),
                "would_pick_under_le": other.randint(0, len(TIE_SCORES) - 1),
            }
        )
    return ties


def main() -> None:
    cases = []
    for fronts, scores in CASES:
        pareto, pareto_after = picks(lambda rng: ParetoCandidateSelector(rng=rng), fronts, scores)
        best, best_after = picks(lambda _rng: CurrentBestCandidateSelector(), fronts, scores)
        greedy, greedy_after = picks(
            lambda rng: EpsilonGreedyCandidateSelector(epsilon=0.1, rng=rng), fronts, scores
        )
        topk, topk_after = picks(
            lambda rng: TopKParetoCandidateSelector(k=5, rng=rng), fronts, scores
        )
        cases.append(
            {
                "fronts": fronts,
                "scores": scores,
                "pareto": {"picks": pareto, "after": pareto_after},
                "current_best": {"picks": best, "after": best_after},
                "epsilon_greedy": {"epsilon": 0.1, "picks": greedy, "after": greedy_after},
                "top_k_pareto": {"k": 5, "picks": topk, "after": topk_after},
            }
        )

    # The component selectors, which choose *which* components a reflection rewrites. `all` was
    # built from reading gepa's source and had never been compared against it either.
    # Deliberately *not* alphabetical: sorted, these are hints, instructions, style. dspy builds the
    # seed candidate from `student.named_predictors()`, so the order is the program's declaration
    # order, and a port over a sorted map walks them in the wrong one.
    candidate = {"instructions": "a", "hints": "b", "style": "c"}
    components = []
    for cursor in range(len(candidate)):
        state = ComponentState(list(candidate), cursor)
        round_robin = RoundRobinReflectionComponentSelector()(state, [], [], 0, candidate)
        every = AllReflectionComponentSelector()(state, [], [], 0, candidate)
        components.append(
            {
                "cursor": cursor,
                "round_robin": round_robin,
                "advanced_to": state.named_predictor_id_to_update_next_for_program_candidate[0],
                "all": every,
            }
        )

    fixture = {
        "source": f"generated from gepa=={PINNED} via scripts/generate_gepa_selector_fixture.py",
        "gepa_version": PINNED,
        "note": (
            "All four candidate selectors gepa's factory map holds, two of which dspy names in "
            "its Literal and two of which its string passthrough still reaches. `advanced` is how far the generator moved, which is what a later "
            "round inherits — epsilon-greedy takes one draw or two depending on the coin."
        ),
        "seeds": SEEDS,
        "cases": cases,
        "epsilon_ties": {"scores": TIE_SCORES, "cases": epsilon_ties()},
        "components": {
            "candidate": candidate,
            # Recorded as a list beside the object: JSON preserves the order but serde_json's
            # default map does not, so the crate could not read it back from `candidate` alone.
            "declaration_order": list(candidate),
            "rounds": components,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "gepa_selectors.json"
    path.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"  wrote {path.relative_to(OUT.parent.parent.parent)}", file=sys.stderr)
    # The case set has to exercise both epsilon branches, or it agrees with a port that never
    # draws the coin. Refuse to write one that does not.
    coin_below = [random.Random(seed).random() < 0.1 for seed in SEEDS]
    if not (any(coin_below) and not all(coin_below)):
        raise SystemExit("the seeds do not exercise both epsilon-greedy branches")
    print(
        f"  epsilon branch taken at {sum(coin_below)} of {len(SEEDS)} seeds", file=sys.stderr
    )
    # A tie case that both spellings answer the same way is a case that cannot fail.
    blind = [tie["seed"] for tie in fixture["epsilon_ties"]["cases"] if tie["pick"] == tie["would_pick_under_le"]]
    if blind:
        raise SystemExit(f"epsilon ties at seeds {blind} do not separate `<` from `<=`")
    print(f"  {len(TIE_SEEDS)} epsilon ties, each separating `<` from `<=`", file=sys.stderr)


if __name__ == "__main__":
    main()
