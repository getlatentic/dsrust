"""Record what dspy's GEPA adapter puts in a reflective dataset, by driving the real one.

`make_reflective_dataset` decides what the reflection model is shown, and until this file nothing
compared it: `generate_gepa_engine_fixture.py` drives a *stub* adapter returning a canned dataset,
because what it measures is the engine around it.

The case that matters is a predictor appearing **more than once in one example's trace**, which is
what a loop produces — `ReAct` pushes a step per turn. dspy picks one with
`self.rng.choice(trace_instances)`; a port taking the first agrees only by coincidence, and only at
some seeds. Several seeds are recorded for exactly that reason: at seed 0 the draw lands on the
second hop and at 1-3 on the first, so a fixture at one seed would have agreed with the wrong
implementation three times out of four.

The model is out of the loop — `DummyLM` answers from a table — so the record is a pure function of
the program, the trainset and the seed, which is what the Rust side is given.

    .venv/bin/python scripts/generate_gepa_reflective_fixture.py

Keep `ANSWERS` and the two-hop shape in step with `optimize/gepa/reflective_conformance.rs`; they
describe the same program and the comparison means nothing if they drift.
"""

from __future__ import annotations

import contextlib
import io
import json
import pathlib
import random
import sys

import dspy
from dspy.teleprompt.gepa.gepa_utils import DspyAdapter
from dspy.utils.dummies import DummyLM

from pins import require

OUT = pathlib.Path(__file__).parent.parent / "crates" / "dsrust" / "tests" / "conformance" / "optimize"
PINNED = require("dspy")

#: Question -> answer, chained: each example's first hop answers with the second hop's question.
ANSWERS = {
    "capital of France?": "Paris",
    "Paris": "the Seine",
    "capital of Germany?": "Berlin",
    "Berlin": "the Spree",
}
QUESTIONS = ["capital of France?", "capital of Germany?"]
INSTRUCTION = "Answer the question."
SEEDS = [0, 1, 2, 3]


class TwoHop(dspy.Module):
    """One predictor called twice, so each example's trace holds two instances of `step`."""

    def __init__(self):
        super().__init__()
        self.step = dspy.Predict("question -> answer")

    def forward(self, question):
        first = self.step(question=question)
        second = self.step(question=first.answer)
        return dspy.Prediction(answer=second.answer)


def metric(example, pred, trace=None, pred_name=None, pred_trace=None):
    """Module-level score with feedback naming what the *selected* instance saw.

    `pred_trace` is the instance dspy drew, so the feedback text is what makes the draw visible in
    the recorded record rather than only in the inputs.
    """
    seen = pred_trace[0][1]["question"] if pred_trace else "-"
    return dspy.Prediction(score=1.0, feedback=f"reflected on {seen}")


def case(seed: int) -> dict:
    # Keyed, not sequential: the trace has to be a pure function of the question so the Rust
    # double can reproduce it from the same table rather than from a call count.
    dspy.settings.configure(lm=DummyLM({q: {"answer": a} for q, a in ANSWERS.items()}))
    adapter = DspyAdapter(
        student_module=TwoHop(),
        metric_fn=metric,
        feedback_map={"step": _feedback_fn(metric)},
        rng=random.Random(seed),
    )
    trainset = [
        dspy.Example(question=q, answer=ANSWERS[ANSWERS[q]]).with_inputs("question")
        for q in QUESTIONS
    ]
    candidate = {"step": INSTRUCTION}
    # dspy's evaluate prints a progress bar and logs an average; neither belongs in a fixture run.
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        batch = adapter.evaluate(trainset, candidate, capture_traces=True)
        dataset = adapter.make_reflective_dataset(candidate, batch, ["step"])
    return {"seed": seed, "records": dataset["step"]}


def _feedback_fn(metric_fn):
    """dspy's `feedback_fn_creator` for one predictor, which is what `make_reflective_dataset` calls.

    Rebuilt here rather than imported because it is a closure defined inside `GEPA.compile`; this is
    the same five arguments in the same order, so what is recorded is the real call shape.
    """

    def feedback_fn(predictor_output, predictor_inputs, module_inputs, module_outputs, captured_trace):
        out = metric_fn(
            module_inputs,
            module_outputs,
            captured_trace,
            "step",
            [(None, predictor_inputs, predictor_output)],
        )
        return out

    return feedback_fn


class WithHistory(dspy.Signature):
    """Answer the question."""

    question: str = dspy.InputField()
    history: dspy.History = dspy.InputField()
    answer: str = dspy.OutputField()


class HistoryProgram(dspy.Module):
    """One predictor whose signature declares a `History` input."""

    def __init__(self):
        super().__init__()
        self.step = dspy.Predict(WithHistory)

    def forward(self, question, history):
        return self.step(question=question, history=history)


#: The exchanges the history carries, and the question asked on top of them.
HISTORY_MESSAGES = [
    {"question": "first?", "answer": "one"},
    {"question": "second?", "answer": "two"},
]
HISTORY_QUESTION = "third?"
HISTORY_ANSWER = "three"


def history_case() -> dict:
    """dspy hoists a `History` input into a fenced `Context` key and drops the original field.

    Each message is `str(message)` on a Python dict, so the block is single-quoted Python reprs
    inside a fence that says json — upstream's, not a transcription slip.
    """
    dspy.settings.configure(lm=DummyLM({HISTORY_QUESTION: {"answer": HISTORY_ANSWER}}))
    adapter = DspyAdapter(
        student_module=HistoryProgram(),
        metric_fn=metric,
        feedback_map={"step": _feedback_fn(metric)},
        rng=random.Random(0),
    )
    history = dspy.History(messages=[dict(m) for m in HISTORY_MESSAGES])
    trainset = [
        dspy.Example(
            question=HISTORY_QUESTION, history=history, answer=HISTORY_ANSWER
        ).with_inputs("question", "history")
    ]
    candidate = {"step": INSTRUCTION}
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        batch = adapter.evaluate(trainset, candidate, capture_traces=True)
        dataset = adapter.make_reflective_dataset(candidate, batch, ["step"])
    return {
        "messages": HISTORY_MESSAGES,
        "question": HISTORY_QUESTION,
        "answer": HISTORY_ANSWER,
        "records": dataset["step"],
    }


class TwoNames(dspy.Module):
    """Two *differently named* predictors declaring the same signature."""

    def __init__(self):
        super().__init__()
        self.alpha = dspy.Predict("question -> answer")
        self.beta = dspy.Predict("question -> answer")

    def forward(self, question):
        first = self.alpha(question=question)
        return self.beta(question=first.answer)


def shared_signature_case(seed: int) -> dict:
    """Both components under the *same* instruction, which is the seed candidate's shape.

    dspy filters the trace with `t[0].signature.equals(module.signature)` and `equals` compares
    instructions first, so two identically-declared predictors pool their instances the moment
    their instructions agree — which is exactly before GEPA has rewritten either. Every seed then
    has at least one component reflecting on a step belonging to the other, which matching by the
    predictor's *name* never does.
    """
    dspy.settings.configure(
        lm=DummyLM({q: {"answer": a} for q, a in SHARED_ANSWERS.items()})
    )
    candidate = {name: INSTRUCTION for name in ("alpha", "beta")}
    adapter = DspyAdapter(
        student_module=TwoNames(),
        metric_fn=metric,
        feedback_map={name: _feedback_fn(metric) for name in candidate},
        rng=random.Random(seed),
    )
    trainset = [
        dspy.Example(question=SHARED_QUESTION, answer="end").with_inputs("question")
    ]
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        batch = adapter.evaluate(trainset, candidate, capture_traces=True)
        dataset = adapter.make_reflective_dataset(candidate, batch, ["alpha", "beta"])
    return {"seed": seed, "components": {name: dataset[name] for name in candidate}}


SHARED_QUESTION = "q?"
SHARED_ANSWERS = {"q?": "mid", "mid": "end"}


def main() -> None:
    cases = [case(seed) for seed in SEEDS]
    reflected = {c["seed"]: [r["Inputs"]["question"] for r in c["records"]] for c in cases}
    fixture = {
        "source": f"generated from dspy=={PINNED} via scripts/generate_gepa_reflective_fixture.py",
        "dspy_version": PINNED,
        "note": (
            "dspy's DspyAdapter.make_reflective_dataset over a two-hop program: one predictor, two "
            "trace instances per example. Which instance is reflected on comes from "
            "rng.choice(trace_instances), so it moves with the seed."
        ),
        "answers": ANSWERS,
        "questions": QUESTIONS,
        "instruction": INSTRUCTION,
        "cases": cases,
        "history": history_case(),
        "shared_signature": {
            "question": SHARED_QUESTION,
            "answers": SHARED_ANSWERS,
            "cases": [shared_signature_case(seed) for seed in SEEDS],
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "gepa_reflective.json"
    path.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"  wrote {path.relative_to(OUT.parent.parent.parent)} ({len(cases)} seeds)", file=sys.stderr)
    for seed, questions in reflected.items():
        print(f"    seed {seed}: reflected on {questions}", file=sys.stderr)
    # A fixture whose seeds all draw the same instance agrees with a port that never draws.
    if len({tuple(v) for v in reflected.values()}) < 2:
        raise SystemExit("every seed reflected on the same instance — the draw is not exercised")


if __name__ == "__main__":
    main()
