"""`BootstrapFewShotWithOptuna`, run end to end with its sampler pinned down.

The optimizer bootstraps once with `BootstrapFewShot`, then asks optuna for one `suggest_int` per
predictor — the index of the single demo that predictor keeps — scores the resulting program on the
validation set, and returns the program from the best trial.

**dspy calls `optuna.create_study()` with no seed**, so no two of its own runs agree and there is no
sequence to match. What *can* be matched is everything else: which demos are on offer, how a trial's
indices become a program, what that program scores, and which trial wins. So the study is created
with a seeded sampler here — the one substitution — and the seeding difference is recorded rather
than papered over.

    .venv/bin/python scripts/generate_optuna_bootstrap_fixture.py
"""

from __future__ import annotations

import json
import logging
import pathlib
import warnings
from types import SimpleNamespace

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

import dspy
import optuna

from pins import require

PINNED = require("dspy")
OPTUNA = require("optuna")
optuna.logging.set_verbosity(optuna.logging.CRITICAL)

OUT = (
    pathlib.Path(__file__).parent.parent
    / "crates"
    / "dsrust"
    / "tests"
    / "conformance"
    / "optimize"
    / "optuna_bootstrap.json"
)

CAPITALS = {
    "France": "Paris",
    "Germany": "Berlin",
    "Italy": "Rome",
    "Spain": "Madrid",
    "Japan": "Tokyo",
    "Peru": "Lima",
}


class ScriptedLM(dspy.BaseLM):
    """Answers the capital when the question names a country it knows, and wrongly otherwise.

    Deterministic and stateless, so a trial's score depends only on which demos it kept.
    """

    def __init__(self):
        super().__init__(model="scripted")

    def forward(self, prompt=None, messages=None, **kwargs):
        text = " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for message in (messages or [])
            for part in (
                message["content"]
                if isinstance(message.get("content"), list)
                else [message.get("content", "")]
            )
        )
        asked = [country for country in CAPITALS if f"of {country}?" in text]
        answer = CAPITALS[asked[-1]] if asked else "unknown"
        body = f"[[ ## answer ## ]]\n{answer}\n\n[[ ## completed ## ]]"
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=body, tool_calls=None), finish_reason="stop"
                )
            ],
            usage={},
            model="scripted",
        )


def dataset() -> list[dspy.Example]:
    return [
        dspy.Example(question=f"What is the capital of {country}?", answer=capital).with_inputs(
            "question"
        )
        for country, capital in CAPITALS.items()
    ]


def metric(example, prediction, trace=None) -> float:
    return float(getattr(prediction, "answer", None) == example.answer)


def main() -> None:
    dspy.settings.configure(lm=ScriptedLM(), adapter=dspy.ChatAdapter())

    seeded = {"trials": [], "scores": []}
    original_create = optuna.create_study

    def create(**kwargs):
        # The one substitution: dspy passes no sampler, so its study is entropy-seeded and no two
        # runs agree. Everything else below is dspy's own.
        kwargs.setdefault("sampler", optuna.samplers.TPESampler(seed=0))
        return original_create(**kwargs)

    optuna.create_study = create
    try:
        student = dspy.Predict("question -> answer")
        optimizer = dspy.BootstrapFewShotWithOptuna(
            metric=metric, max_bootstrapped_demos=4, max_labeled_demos=0, num_candidate_programs=14
        )
        study_trials: list[dict] = []
        original_objective = optimizer.objective

        def objective(trial):
            value = original_objective(trial)
            study_trials.append({"params": dict(trial.params), "value": value})
            return value

        optimizer.objective = objective
        best = optimizer.compile(
            student, max_demos=4, trainset=dataset(), valset=dataset()
        )
    finally:
        optuna.create_study = original_create

    seeded["trials"] = [list(t["params"].values()) for t in study_trials]
    seeded["scores"] = [t["value"] for t in study_trials]

    offered = {
        name: [dict(demo) for demo in predictor.demos]
        for name, predictor in optimizer.compiled_teleprompter.named_predictors()
    }
    kept = {
        name: [dict(demo) for demo in predictor.demos]
        for name, predictor in best.named_predictors()
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "source": f"generated from dspy=={PINNED} and optuna=={OPTUNA} via "
                "scripts/generate_optuna_bootstrap_fixture.py",
                "note": (
                    "dspy creates its study with no sampler, so upstream's own runs are "
                    "entropy-seeded and none of them agree. This run injected "
                    "`TPESampler(seed=0)`; nothing else was changed."
                ),
                "parameter_names": list(study_trials[0]["params"]) if study_trials else [],
                "demos_on_offer": offered,
                "trials": seeded["trials"],
                "scores": seeded["scores"],
                "best_trial": max(range(len(seeded["scores"])), key=lambda i: seeded["scores"][i]),
                "demos_kept": kept,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"  demos on offer : { {k: len(v) for k, v in offered.items()} }")
    print(f"  trials         : {len(seeded['trials'])}  indices {seeded['trials']}")
    print(f"  scores         : {seeded['scores']}")
    print(f"  kept           : { {k: len(v) for k, v in kept.items()} }")
    print(f"wrote {OUT.relative_to(pathlib.Path(__file__).parent.parent)}")


if __name__ == "__main__":
    main()
