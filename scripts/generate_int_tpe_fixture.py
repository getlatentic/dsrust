"""optuna's TPE over integer parameters, recorded as the trials it actually proposes.

`BootstrapFewShotWithOptuna` asks for one `suggest_int` per predictor, so its search is a seeded
`TPESampler` over `IntDistribution`s — and at optuna's defaults that is `multivariate=False`, which
means every parameter goes through `sample_independent` on its own rather than being drawn jointly.
The first ten trials come from a `RandomSampler` seeded alongside the TPE one; after that each
parameter is drawn from a truncated normal fitted to the trials so far.

Recording whole sequences rather than pieces is deliberate: the pieces (`ndtr`, `ppf`, the Parzen
shape) are each held to their own grid elsewhere, and what this adds is that they are *composed* in
optuna's order, through one generator whose every draw advances the same stream.

The objective is a fixed function of the parameters, so a run is decided entirely by the seed.

    .venv/bin/python scripts/generate_int_tpe_fixture.py
"""

from __future__ import annotations

import json
import logging
import pathlib
import warnings

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

import optuna

from pins import require

PINNED = require("optuna")
optuna.logging.set_verbosity(optuna.logging.CRITICAL)

OUT = (
    pathlib.Path(__file__).parent.parent
    / "crates"
    / "dsrust-tpe"
    / "tests"
    / "conformance"
    / "int_tpe.json"
)

#: Each case names its parameters as `(name, low, high)` and how many trials to run. Names matter:
#: they decide nothing here, since `multivariate=False` draws each parameter on its own, but they
#: are recorded so a port cannot quietly depend on them.
CASES = [
    ("one_parameter_through_startup_only", [("demo_index_for_p", 0, 3)], 8),
    ("one_parameter_into_tpe", [("demo_index_for_p", 0, 3)], 20),
    ("two_parameters_into_tpe", [("demo_index_for_a", 0, 3), ("demo_index_for_b", 0, 7)], 20),
    ("a_single_choice_parameter", [("demo_index_for_p", 0, 0)], 14),
    ("a_wide_range", [("demo_index_for_p", 0, 49)], 24),
    (
        # Past 25 trials the oldest weights fade — `np.linspace(1/n, 1.0, num=n-25)` — and every
        # shorter run in this file leaves that ramp empty, so nothing here reached it until now.
        "past_twenty_five_trials_the_oldest_weights_fade",
        [("demo_index_for_p", 0, 5)],
        32,
    ),
    (
        "three_parameters_of_different_widths",
        [("demo_index_for_a", 0, 1), ("demo_index_for_b", 0, 9), ("demo_index_for_c", 0, 4)],
        18,
    ),
]


def score(values: list[int]) -> float:
    """A fixed function of the parameters, so the seed alone decides the run.

    Deliberately not monotone: a score that rose with every index would put the same trials below
    the split every time and never exercise the estimator's shape.
    """
    return float(sum((i + 1) * ((v * 7) % 5) for i, v in enumerate(values)))


def main() -> None:
    cases = []
    for name, parameters, n_trials in CASES:
        for seed in (0, 7):
            study = optuna.create_study(
                direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
            )
            drawn: list[list[int]] = []

            def objective(trial, parameters=parameters, drawn=drawn):
                values = [trial.suggest_int(p, low, high) for p, low, high in parameters]
                drawn.append(values)
                return score(values)

            study.optimize(objective, n_trials=n_trials)
            cases.append(
                {
                    "name": name,
                    "seed": seed,
                    "parameters": [
                        {"name": p, "low": low, "high": high} for p, low, high in parameters
                    ],
                    "trials": drawn,
                    "scores": [score(values) for values in drawn],
                    "best": list(study.best_params.values()),
                }
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "source": f"generated from optuna=={PINNED} via scripts/generate_int_tpe_fixture.py",
                "n_startup_trials": optuna.samplers.TPESampler()._n_startup_trials,
                "note": (
                    "`trials` is the parameter values optuna proposed, in order; `scores` is what "
                    "the fixed objective returned for each, which is what the next trial learns "
                    "from."
                ),
                "cases": cases,
            },
            indent=2,
        )
        + "\n"
    )
    for case in cases:
        flat = [v for values in case["trials"] for v in values]
        print(f"  {case['name']:38s} seed {case['seed']}  {len(case['trials'])} trials  "
              f"first {flat[:6]}")
    print(f"wrote {OUT.relative_to(pathlib.Path(__file__).parent.parent)}")


if __name__ == "__main__":
    main()
