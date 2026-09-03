"""Record optuna's TPESampler trials, so the `tpe` crate's sampler can be held to them.

dspy's MIPROv2 drives its search with `optuna.samplers.TPESampler(multivariate=True)` over
categorical distributions. This runs that sampler on fixed table objectives and records the exact
sequence of trials it proposes, which the crate reproduces.

The objectives assign a distinct score to every category combination. Ties in the objective feed
symmetric acquisitions, and optuna computes those with numpy's vectorized `log`/`exp`, whose last
bit the crate's scalar transcendentals cannot always match — so a symmetric objective can push two
candidates within a few ULPs and rank them the other way. That boundary is documented on the
sampler; distinct scores keep it out of the reproduced sequences, which is what these cases hold.

Requires optuna (dspy's optional dependency):

    uv pip install --python .dspy-venv/bin/python optuna==4.5.0
    .dspy-venv/bin/python scripts/generate_tpe_fixture.py
"""

from __future__ import annotations

import itertools
import json
import pathlib
import sys

import optuna

OUT = pathlib.Path(__file__).parent.parent / "crates" / "dsrust-tpe" / "tests" / "conformance" / "optuna_tpe.json"

# (seed, cardinalities, n_trials, mix, names) — mix varies the distinct-score table between cases;
# names are the parameter names the objective suggests under, defaulting to p0, p1, … .
CASES = [
    (42, [4, 3], 20, 1),
    (7, [5], 18, 2),
    (123, [2, 2, 2], 18, 3),
    (2024, [4, 4], 30, 4),
    (99, [3, 3, 2], 28, 5),
    (555, [6], 22, 6),
    (2718, [5, 4], 35, 7),
    # MIPROv2's own regime. `auto="medium"` proposes twelve instructions per predictor and
    # `auto="heavy"` eighteen, where every case above stops at six — and the parzen estimator's
    # weights are per *category*, so agreeing at six says nothing about twelve. Found by an
    # `auto="medium"` conformance case diverging at the first non-startup draw.
    (3, [12], 30, 8),
    (11, [18], 30, 9),
    (13, [12, 6], 30, 10),
    # dspy's own names, suggested instruction-first while `demos` sorts first — with *unequal*
    # cardinalities, so the two orders genuinely disagree. optuna draws startup trials one parameter
    # at a time in suggest order and TPE trials all at once from a search space that is
    # `dict(sorted(...))`, so a sampler that picks one order for both diverges as soon as a run is
    # long enough to reach the second phase. Equal cardinalities hide it, which is why every earlier
    # case did.
    (17, [6, 12], 30, 11, ["0_predictor_instruction", "0_predictor_demos"]),
    (23, [4, 9], 26, 12, ["0_predictor_instruction", "0_predictor_demos"]),
]


def distinct_table(cardinalities: list[int], mix: int) -> dict[tuple[int, ...], float]:
    """A distinct score per category combination, unevenly spaced to avoid acquisition symmetry."""
    combinations = list(itertools.product(*[range(c) for c in cardinalities]))
    return {
        combination: 0.1 + (index * 0.7919 + mix * 0.3313) % 5.0
        for index, combination in enumerate(combinations)
    }


def run(seed: int, cardinalities: list[int], table: dict, n_trials: int, names=None) -> list[list[int]]:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    named = names or [f"p{i}" for i in range(len(cardinalities))]
    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    sequence: list[list[int]] = []

    def objective(trial: optuna.Trial) -> float:
        combination = tuple(
            trial.suggest_categorical(named[i], list(range(c))) for i, c in enumerate(cardinalities)
        )
        sequence.append(list(combination))
        return table[combination]

    study.optimize(objective, n_trials=n_trials)
    return sequence


def key_of(combination: tuple[int, ...]) -> str:
    return ",".join(map(str, combination)) if len(combination) > 1 else str(combination[0])


def build_case(seed: int, cardinalities: list[int], n_trials: int, mix: int, names=None) -> dict:
    table = distinct_table(cardinalities, mix)
    return {
        "cards": cardinalities,
        "names": names or [f"p{i}" for i in range(len(cardinalities))],
        "table": {key_of(combination): score for combination, score in table.items()},
        "seed": seed,
        "n_trials": n_trials,
        "sequence": run(seed, cardinalities, table, n_trials, names),
    }


def main() -> None:
    import warnings

    warnings.filterwarnings("ignore")
    fixture = {
        "source": f"optuna {optuna.__version__} via scripts/generate_tpe_fixture.py",
        "optuna_version": optuna.__version__,
        "cases": [build_case(*case) for case in CASES],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"  wrote {OUT.relative_to(OUT.parent.parent.parent)}", file=sys.stderr)


if __name__ == "__main__":
    main()
