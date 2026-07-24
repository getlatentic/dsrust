"""Record what dspy's GEPA teleprompter compiles, driven by scripted models.

GEPA evolves a predictor's instruction by reflecting on how the program did. To make a run
deterministic and reproducible in Rust without real LLMs, two scripted models drive it:

  - the task model answers a question correctly only when the instruction in force carries `GOOD`
    (which it reads from the system prompt) — so a candidate carrying `GOOD` scores 100% against the
    seed's 0%, and the search must accept it.
  - the reflection model, whatever prompt it is shown, proposes the instruction carrying `GOOD`.

What is compared is the instruction GEPA leaves the student holding. Merge is turned off
(`use_merge=True`, dspy's default) — the student is a single `Predict`, so its one component gives
merge nothing to combine and the compiled instruction is the same either way.

    .dspy-venv/bin/python scripts/generate_gepa_optimize_fixture.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import dspy
from dspy.clients.base_lm import BaseLM
from dspy.dsp.utils.utils import dotdict
from dspy.teleprompt.gepa.gepa import GEPA

from pins import require

OUT = pathlib.Path(__file__).parent.parent / "tests" / "conformance" / "optimize"
# Both libraries produce these bytes: dspy's teleprompter drives the engine that gepa ships.
PINNED = require("dspy")
GEPA_PINNED = require("gepa")

TABLE = {"capital of France?": "Paris", "capital of Germany?": "Berlin", "capital of Spain?": "Madrid"}
PROPOSAL = "Answer with GOOD precision."


def _reply(content: str) -> dotdict:
    message = dotdict(content=content, tool_calls=None)
    return dotdict(
        choices=[dotdict(message=message, finish_reason="stop")],
        usage=dotdict(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        model="scripted",
    )


class Coach(BaseLM):
    """The task model: answers correctly only when `GOOD` is the instruction in force."""

    def __init__(self):
        super().__init__("coach", "chat", 0.0, 1000, True)

    def forward(self, prompt=None, messages=None, **kwargs):
        system, last = messages[0]["content"], messages[-1]["content"]
        question = next((q for q in TABLE if q in last), None)
        answer = TABLE[question] if (question and "GOOD" in system) else "wrong"
        return _reply(f"[[ ## answer ## ]]\n{answer}\n\n[[ ## completed ## ]]")


class Reflector(BaseLM):
    """The reflection model: proposes the `GOOD` instruction, fenced, whatever it is shown."""

    def __init__(self):
        super().__init__("reflector", "chat", 0.0, 1000, True)

    def forward(self, prompt=None, messages=None, **kwargs):
        return _reply(f"```\n{PROPOSAL}\n```")


class Program(dspy.Module):
    def __init__(self, seed_instruction: str):
        super().__init__()
        self.predict = dspy.Predict("question -> answer")
        self.predict.signature = self.predict.signature.with_instructions(seed_instruction)

    def forward(self, question):
        return self.predict(question=question)


def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    correct = gold.answer == pred.answer
    if correct:
        return dspy.Prediction(score=1.0, feedback="Correct.")
    return dspy.Prediction(score=0.0, feedback="Wrong answer; be more precise.")


# (seed_instruction, minibatch_size, max_metric_calls, seed). Varied so the crate must reflect each
# distinct seed instruction and reproduce runs of different lengths.
CASES = [
    ("Answer the question.", 2, 20, 0),
    ("Respond to the query.", 2, 20, 4),
    ("Solve it.", 3, 30, 7),
]


def compile_once(seed_instruction: str, minibatch_size: int, max_metric_calls: int, seed: int) -> dict:
    dspy.configure(lm=Coach())
    trainset = [dspy.Example(question=q, answer=a).with_inputs("question") for q, a in TABLE.items()]
    optimizer = GEPA(
        metric=metric,
        reflection_lm=Reflector(),
        max_metric_calls=max_metric_calls,
        reflection_minibatch_size=minibatch_size,
        candidate_selection_strategy="pareto",
        skip_perfect_score=True,
        # dspy's own default, matching the dsrs wrapper. The program is a single `Predict`, so it
        # has one component and merge never has two predictors to combine — the compiled result is
        # identical whether merge is on or off, which is what keeps this fixture stable across the
        # change that turned merge on.
        use_merge=True,
        seed=seed,
        track_stats=True,
    )
    compiled = optimizer.compile(Program(seed_instruction), trainset=trainset, valset=trainset)
    result = compiled.detailed_results
    return {
        "seed_instruction": seed_instruction,
        "minibatch_size": minibatch_size,
        "max_metric_calls": max_metric_calls,
        "seed": seed,
        "compiled_instruction": compiled.predict.signature.instructions,
        "num_candidates": len(result.candidates),
        "total_metric_calls": result.total_metric_calls,
    }


def main() -> None:
    fixture = {
        "source": (
            f"generated from dspy=={PINNED} + gepa=={GEPA_PINNED} "
            "via scripts/generate_gepa_optimize_fixture.py"
        ),
        "dspy_version": PINNED,
        "gepa_version": GEPA_PINNED,
        "cases": [compile_once(*case) for case in CASES],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "gepa.json"
    path.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"  wrote {path.relative_to(OUT.parent.parent.parent)}", file=sys.stderr)


if __name__ == "__main__":
    main()
