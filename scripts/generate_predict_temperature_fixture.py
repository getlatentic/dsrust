"""Record what `Predict` resolves its temperature to when more than one completion is asked for.

dspy `predict.py::_forward_preprocess` carries a rule with no counterpart anywhere else in the
library: asking for several completions at a near-zero temperature sends **0.7** instead, "to keep
randomness". Three lines, and none of them mean what they look like.

    temperature = config.get("temperature") or lm.kwargs.get("temperature")
    num_generations = config.get("n") or lm.kwargs.get("n") or lm.kwargs.get("num_generations") or 1
    if (temperature is None or temperature <= 0.15) and num_generations > 1:
        config["temperature"] = 0.7

`or` is Python's, so **0.0 is falsy**: a caller who asks for temperature 0.0 has asked for nothing
as far as this rule can tell, and falls through to the model's — which is why `temperature=0.0, n=2`
resolves to 0.7 while `temperature=0.0` under a model set to 0.9 does not, and sends 0.0. `n` reads
through the same chain with a third arm, `num_generations`, which `dspy.LM` keeps as a kwarg of its
own.

Recorded by calling `_forward_preprocess` itself, so what lands in the golden is the dict dspy
handed the adapter rather than a reading of the source.

    .venv/bin/python scripts/generate_predict_temperature_fixture.py
"""

from __future__ import annotations

import itertools
import json
import logging
import pathlib
import sys
import warnings

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

import dspy

from pins import require

PINNED = require("dspy")
OUT = (
    pathlib.Path(__file__).parent.parent
    / "crates"
    / "dsrust"
    / "tests"
    / "conformance"
    / "predict"
    / "completion_temperature.json"
)

#: What the *model* was built with — dspy's `lm.kwargs`, the second and third arms of both chains.
MODELS = [
    {},
    {"temperature": 0.0},
    {"temperature": 0.15},
    {"temperature": 0.2},
    {"temperature": 0.9},
    {"n": 2},
    {"n": 1},
    {"temperature": 0.0, "n": 2},
    {"temperature": 0.9, "n": 2},
    {"num_generations": 3},
    {"temperature": 0.0, "num_generations": 3},
]
#: What the *module* was configured with — dspy's `self.config`, the first arm of both chains.
CONFIGS = [
    {},
    {"temperature": 0.0},
    {"temperature": 0.1},
    {"temperature": 0.15},
    {"temperature": 0.5},
    {"n": 2},
    {"n": 1},
    {"temperature": 0.0, "n": 2},
    {"temperature": 0.5, "n": 2},
    {"temperature": 0.15, "n": 3},
]


def resolved(model: dict, config: dict) -> dict:
    """The `lm_kwargs` dict `_forward_preprocess` hands the adapter."""
    predictor = dspy.Predict("question -> answer", **config)
    predictor.lm = dspy.LM("openai/gpt-4o-mini", **model)
    return predictor._forward_preprocess(question="hi")[1]


def main() -> None:
    cases = [
        {"model": model, "config": config, "resolved": resolved(model, config)}
        for model, config in itertools.product(MODELS, CONFIGS)
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "source": f"dspy=={PINNED} predict/predict.py::Predict._forward_preprocess",
                "dspy_version": PINNED,
                "note": (
                    "Per (model kwargs, module config): the config dspy resolved. `temperature` "
                    "appears in the result only where the module set one or the rule wrote 0.7, "
                    "since the model's own kwargs merge later at the LM rather than here."
                ),
                "cases": cases,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    bumped = sum(1 for case in cases if case["resolved"].get("temperature") == 0.7)
    print(f"  wrote {OUT.name}: {len(cases)} cases, {bumped} where the rule fired", file=sys.stderr)


if __name__ == "__main__":
    main()
