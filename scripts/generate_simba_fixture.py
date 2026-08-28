"""Record what dspy's own SIMBA decides, by running it.

There is **no upstream test file for SIMBA**, so nothing in dspy's suite can be borrowed here: the
only oracle is a run. The trace is made deterministic the way `generate_copro_fixture.py` makes
COPRO's — a `DummyLM` answering from a table keyed by tokens that appear in exactly one kind of
call, and a single thread so the call order is stable.

With the model fixed, SIMBA is a pure function of the trainset, the metric, the configuration and
the two seeds it draws from — `random.Random(seed)` and `np.random.default_rng(seed)`. What is
recorded is every decision the search makes, in order:

  - the shuffled data indices and each step's mini-batch,
  - the rollout ids and temperatures `prepare_models_for_resampling` produces,
  - the programs softmax-sampling picks, at both temperatures,
  - each step's 10th and 90th percentile, and the buckets in their sorted order with the three
    keys they sort on,
  - the demo-drop count each candidate takes, which is the `poisson` draw,
  - which strategy each bucket invoked and whether it returned True,
  - the final slate, its scores, and the winning instruction.

Instrumented by wrapping the module's own helpers rather than by re-deriving them, so what lands
in the golden is what SIMBA did.

    .venv/bin/python scripts/generate_simba_fixture.py
"""

from __future__ import annotations

import json
import logging
import pathlib
import random
import sys
import warnings

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

import numpy as np

import dspy
from dspy.teleprompt import simba as simba_module
from dspy.teleprompt import simba_utils
from dspy.utils.dummies import DummyLM

from pins import require

PINNED = require("dspy")
OUT = (
    pathlib.Path(__file__).parent.parent
    / "crates" / "dsrust" / "tests" / "conformance" / "optimize" / "simba.json"
)

#: Six examples so a batch of three leaves a second batch, and the reshuffle at the wrap is
#: reached within the step budget.
TRAINSET = [
    ("capital of France?", "Paris"),
    ("capital of Spain?", "Madrid"),
    ("capital of Italy?", "Rome"),
    ("capital of Japan?", "Tokyo"),
    ("capital of Peru?", "Lima"),
    ("capital of Chile?", "Santiago"),
]

#: The task answers correctly for three of the six, so scores differ across a bucket and the
#: max-to-min gap that orders them is non-zero. `module_advice` is what `append_a_rule` reads.
KEYED = {
    "capital of France?": {"answer": "Paris"},
    "capital of Spain?": {"answer": "Barcelona"},
    "capital of Italy?": {"answer": "Rome"},
    "capital of Japan?": {"answer": "Kyoto"},
    "capital of Peru?": {"answer": "Lima"},
    "capital of Chile?": {"answer": "Valparaiso"},
    # `append_a_rule`'s ask, keyed on a field marker no task call carries.
    "module_advice": {
        "discussion": "The worse trajectory guessed a second city.",
        "module_advice": {"predict": "Name the seat of government, not the largest city."},
    },
}


def metric(example, prediction, trace=None):
    return float(prediction is not None and getattr(prediction, "answer", None) == example.answer)


def main() -> None:
    decisions: dict = {"steps": []}

    # --- instrument the module's own helpers ------------------------------------------------
    real_prepare = simba_utils.prepare_models_for_resampling
    real_rule = simba_utils.append_a_rule
    real_demo_factory = simba_utils.append_a_demo

    def prepare(program, n, teacher_settings=None):
        models = real_prepare(program, n, teacher_settings)
        decisions.setdefault("models", []).append(
            [{"rollout_id": m.kwargs.get("rollout_id"), "temperature": m.kwargs.get("temperature")}
             for m in models]
        )
        return models

    def rule(bucket, system, **kwargs):
        applied = real_rule(bucket, system, **kwargs)
        rule.__name__ = "append_a_rule"
        decisions.setdefault("strategies", []).append({"strategy": "append_a_rule", "applied": bool(applied)})
        return applied

    def demo_factory(maxlen):
        real = real_demo_factory(maxlen)

        def wrapped(bucket, system, **kwargs):
            applied = real(bucket, system, **kwargs)
            decisions.setdefault("strategies", []).append(
                {"strategy": "append_a_demo", "applied": bool(applied)}
            )
            return applied

        wrapped.__name__ = "append_a_demo_"
        return wrapped

    # The search's own internals, which are what discriminate a working port from one that
    # returns the student: the scripted model answers the same however the prompt changes, so the
    # final winner can tie with the baseline while every decision along the way is distinct.
    real_percentile = simba_module.np.percentile
    real_shuffle = random.Random.shuffle
    real_choices = random.Random.choices
    real_randrange = random.Random.randrange
    real_choice = random.Random.choice

    def percentile(a, q, **kw):
        value = real_percentile(a, q, **kw)
        decisions.setdefault("percentiles", []).append(
            {"q": float(q), "sample": [float(x) for x in a], "value": float(value)}
        )
        return value

    def shuffle(self, seq):
        real_shuffle(self, seq)
        decisions.setdefault("shuffles", []).append(list(seq))

    def choices(self, population, weights=None, *, cum_weights=None, k=1):
        picked = real_choices(self, population, weights=weights, cum_weights=cum_weights, k=k)
        decisions.setdefault("softmax_picks", []).append(
            {"population": list(population),
             "weights": [float(w) for w in weights] if weights else None,
             "picked": list(picked)}
        )
        return picked

    def randrange(self, *args, **kw):
        drawn = real_randrange(self, *args, **kw)
        decisions.setdefault("randranges", []).append({"args": list(args), "drawn": drawn})
        return drawn

    def strategy_choice(self, seq):
        picked = real_choice(self, seq)
        decisions.setdefault("strategy_picks", []).append(getattr(picked, "__name__", str(picked)))
        return picked

    random.Random.shuffle = shuffle
    random.Random.choices = choices
    random.Random.randrange = randrange
    random.Random.choice = strategy_choice

    # `numpy.random.Generator` is an immutable extension type, so the draw is intercepted by
    # handing SIMBA a recording proxy in place of the generator itself.
    class RecordingGenerator:
        def __init__(self, inner):
            self._inner = inner

        def poisson(self, lam=1.0, size=None):
            drawn = self._inner.poisson(lam, size)
            decisions.setdefault("poissons", []).append(
                {"lam": float(lam), "drawn": int(drawn)}
            )
            return drawn

        def __getattr__(self, name):
            return getattr(self._inner, name)

    real_default_rng = simba_module.np.random.default_rng

    def default_rng(seed=None):
        return RecordingGenerator(real_default_rng(seed))

    class RecordingRandom:
        def __init__(self):
            self.default_rng = default_rng

        def __getattr__(self, name):
            return getattr(np.random, name)

    class RecordingNumpy:
        def __init__(self):
            self.percentile = percentile
            self.random = RecordingRandom()

        def __getattr__(self, name):
            return getattr(np, name)

    simba_module.np = RecordingNumpy()

    simba_utils.prepare_models_for_resampling = prepare
    simba_module.prepare_models_for_resampling = prepare
    rule.__name__ = "append_a_rule"
    simba_utils.append_a_rule = rule
    simba_module.append_a_rule = rule
    simba_utils.append_a_demo = demo_factory
    simba_module.append_a_demo = demo_factory

    lm = DummyLM(KEYED)
    dspy.configure(lm=lm)

    # Two demos to start with, so the demo-drop path is reached on the first step: its poisson
    # draw is `num_demos / max_demos` and a candidate with none makes that lambda zero, which
    # exercises neither the draw nor the `randrange` that follows it.
    student = dspy.Predict("question -> answer")
    student.demos = [
        dspy.Example(augmented=True, question="capital of Kenya?", answer="Nairobi").toDict(),
        dspy.Example(augmented=True, question="capital of Ghana?", answer="Accra").toDict(),
    ]
    trainset = [dspy.Example(question=q, answer=a).with_inputs("question") for q, a in TRAINSET]

    optimizer = simba_module.SIMBA(
        metric=metric, bsize=3, num_candidates=2, max_steps=3, max_demos=2, num_threads=1
    )
    compiled = optimizer.compile(student, trainset=trainset, seed=0)

    # --- and the RNG streams the run consumed, replayed independently ------------------------
    random.Random.shuffle = real_shuffle
    random.Random.choices = real_choices
    random.Random.randrange = real_randrange
    random.Random.choice = real_choice
    rng = random.Random(0)
    indices = list(range(len(trainset)))
    rng.shuffle(indices)

    fixture = {
        "source": f"generated from dspy=={PINNED} via scripts/generate_simba_fixture.py",
        "dspy_version": PINNED,
        "note": (
            "Every decision dspy's SIMBA made on a scripted model, in order. There is no upstream "
            "test file for SIMBA, so a run is the only oracle there is."
        ),
        "config": {"bsize": 3, "num_candidates": 2, "max_steps": 3, "max_demos": 2, "seed": 0},
        "trainset": [{"question": q, "answer": a} for q, a in TRAINSET],
        "shuffled_indices": indices,
        "models": decisions.get("models", []),
        "strategies": decisions.get("strategies", []),
        "shuffles": decisions.get("shuffles", []),
        "percentiles": decisions.get("percentiles", []),
        "softmax_picks": decisions.get("softmax_picks", []),
        "poissons": decisions.get("poissons", []),
        "randranges": decisions.get("randranges", []),
        "strategy_picks": decisions.get("strategy_picks", []),
        "compiled_predictors": [
            {
                "name": name,
                "instructions": predictor.signature.instructions,
                "demos": [{k: str(v) for k, v in demo.items()} for demo in predictor.demos],
            }
            for name, predictor in compiled.named_predictors()
        ],
        "candidate_scores": [entry["score"] for entry in compiled.candidate_programs],
        "trial_logs": {str(k): v for k, v in compiled.trial_logs.items()},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
    print(
        f"  wrote {OUT.name}: {len(fixture['strategies'])} strategy calls, "
        f"{len(fixture['candidate_scores'])} final candidates",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
