"""Record how GEPA's per-testcase Pareto front evolves, by running its GEPAState.

The front (`program_at_pareto_front_valset`) starts with the seed program on every validation
testcase, and each new program replaces a testcase's front where it scores strictly higher, joins it
on an exact tie, and is ignored where it scores worse. Deterministic, so what is compared is the
front after the seed and after each added program.

    .dspy-venv/bin/python scripts/generate_gepa_front_fixture.py
"""

from __future__ import annotations

import json
import pathlib
import sys

from gepa.core.state import GEPAState, ValsetEvaluation

OUT = pathlib.Path(__file__).parent.parent / "gepa" / "tests" / "conformance"
PINNED = "0.0.27"

# Each case: the seed program's per-testcase scores, then the scores of programs added in turn.
CASES = [
    {"seed": [0.5, 0.2, 0.7], "programs": [[0.5, 0.9, 0.3], [0.6, 0.9, 0.7], [0.1, 0.9, 0.7]]},
    {"seed": [0.0, 0.0, 0.0, 0.0], "programs": [[1.0, 1.0, 1.0, 1.0]]},  # dominates every testcase
    {"seed": [0.3, 0.3], "programs": [[0.3, 0.3], [0.3, 0.3]]},           # all tie -> fronts grow
    {"seed": [0.5, 0.5, 0.5], "programs": [[0.6, 0.4, 0.5], [0.4, 0.6, 0.5], [0.7, 0.7, 0.4]]},
]


def front_of(state) -> dict:
    return {str(val_id): sorted(front) for val_id, front in state.program_at_pareto_front_valset.items()}


def evaluation(scores: list[float]) -> ValsetEvaluation:
    return ValsetEvaluation(outputs_by_val_id={}, scores_by_val_id={i: s for i, s in enumerate(scores)}, objective_scores_by_val_id=None)


def build_once(case: dict) -> dict:
    state = GEPAState(seed_candidate={"instruction": "seed"}, base_evaluation=evaluation(case["seed"]))
    fronts = [front_of(state)]
    for program_scores in case["programs"]:
        state.update_state_with_new_program(
            parent_program_idx=[0],
            new_program={"instruction": "p"},
            valset_evaluation=evaluation(program_scores),
            run_dir=None,
            num_metric_calls_by_discovery_of_new_program=1,
        )
        fronts.append(front_of(state))
    return {"seed": case["seed"], "programs": case["programs"], "fronts": fronts}


def main() -> None:
    fixture = {
        "source": f"generated from gepa=={PINNED} via scripts/generate_gepa_front_fixture.py",
        "gepa_version": PINNED,
        "cases": [build_once(case) for case in CASES],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "front.json"
    path.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"  wrote {path.relative_to(OUT.parent.parent)}", file=sys.stderr)


if __name__ == "__main__":
    main()
