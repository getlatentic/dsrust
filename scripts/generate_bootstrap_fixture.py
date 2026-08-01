"""Record what dspy's own BootstrapFewShot decides, by running it.

`teleprompt/test_bootstrap.py` passing proves nothing about `src/optimize/`: those tests drive
*dspy's* optimizer over this crate's adapter, so they cross into Rust through a component the
file's name does not mention. This fixture closes that by replaying identical traces through
both implementations and comparing which demos survive.

The traces are made identical by removing the model from the loop. `DummyLM` answers from a
table keyed by question, so a compile here is a pure function of the trainset, the metric and
the configuration — the same three things the Rust side is given. What is left to compare is
exactly the optimizer's decisions.

    .dspy-venv/bin/python scripts/generate_bootstrap_fixture.py

Keep `TRAINSET` and `ANSWERS` in step with `src/optimize/scripted.rs`. They describe the same
program, and the comparison is meaningless if they drift.
"""

from __future__ import annotations

import json
import pathlib
import sys

import dspy
from dspy.utils.dummies import DummyLM

OUT = pathlib.Path(__file__).parent.parent / "crates" / "dsrust" / "tests" / "conformance" / "optimize"
PINNED = (pathlib.Path(__file__).parent / "DSPY_VERSION").read_text().strip()

# `scripted::trainset()`: two the table solves and four it does not, so a validation set is left
# behind whenever the bootstrap budget is smaller than the trainset. Distinct answers throughout,
# so a demo can be told apart from any other in the report.
TRAINSET = [
    ("capital of France?", "Paris"),
    ("capital of Germany?", "Berlin"),
    *((f"riddle {index}?", f"riddle {index}!") for index in range(4)),
]

# `scripted::answer()`: the capital table, wrong the same way every time for anything else.
ANSWERS = {question: ("Paris" if "France" in question else "Berlin") for question, _ in TRAINSET}

# (max_bootstrapped_demos, max_labeled_demos, max_rounds, metric, threshold). Zero on either
# budget is a boundary upstream branches on: no labelled demos skips the priming pass, no
# bootstrapped demos leaves the labelled budget whole.
#
# The graded metric earns its place. Upstream reads a metric for Python truth when no threshold
# is set, so a score of 0.5 *succeeds*; a threshold turns the same score into a comparison it
# fails. An exact-match metric only ever returns 0.0 or 1.0, under which those two readings agree
# and a port could confuse them undetected.
CONFIGS = [
    (4, 16, 1, "exact", None),
    (0, 16, 1, "exact", None),
    (1, 16, 1, "exact", None),
    (2, 16, 1, "exact", None),
    (4, 0, 1, "exact", None),
    (4, 1, 1, "exact", None),
    (4, 2, 1, "exact", None),
    (4, 4, 1, "exact", None),
    (1, 2, 1, "exact", None),
    (2, 3, 1, "exact", None),
    (4, 16, 2, "exact", None),
    (2, 5, 2, "exact", None),
    (4, 16, 1, "graded", None),
    (4, 16, 1, "graded", 1.0),
    (2, 4, 1, "graded", None),
    (2, 4, 1, "graded", 0.75),
    (4, 16, 1, "graded", 0.25),
    # A threshold of zero is upstream's `if self.metric_threshold:`, which Python reads as false —
    # so it means *no* threshold rather than a bar at zero, and a score of 0.0 still fails. Only
    # a metric that returns 0.0 for something tells the two readings apart, which exact match
    # does and the graded one never does.
    (4, 16, 1, "exact", 0.0),
    (2, 4, 1, "exact", 0.0),
]


class Answer(dspy.Signature):
    """Answer."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class Draft(dspy.Signature):
    """Answer."""

    question: str = dspy.InputField()
    draft: str = dspy.OutputField()


class Settle(dspy.Signature):
    """Answer."""

    draft: str = dspy.InputField()
    answer: str = dspy.OutputField()


class Pair(dspy.Module):
    """Two predictors, mirroring `scripted::Pair`.

    The halves are deliberately not interchangeable: the first reads a question and writes a
    draft, the second reads that draft and writes an answer. A demo one earned is therefore a
    demo the other could not have, so misattribution shows up as the wrong fields rather than as
    a different ordering.
    """

    def __init__(self):
        super().__init__()
        self.first = dspy.Predict(Draft)
        self.second = dspy.Predict(Settle)

    def forward(self, question):
        return self.second(draft=self.first(question=question).draft)


def drafted(answer: str) -> str:
    """`scripted::drafted`: the intermediate the first half hands the second."""
    return f"draft: {answer}"


# Keyed for both halves at once: a question answers with a draft, and that draft answers with the
# settled answer. `DummyLM` matches on the final message only, so the two never collide.
PAIR_ANSWERS = {}
for _question, _ in TRAINSET:
    _settled = "Paris" if "France" in _question else "Berlin"
    PAIR_ANSWERS[_question] = {"draft": drafted(_settled)}
    PAIR_ANSWERS[drafted(_settled)] = {"answer": _settled}

# The rebinding only shows where the labelled budget leaves something to shrink, so every case
# here keeps one.
PAIR_CONFIGS = [
    (4, 16, 1, "exact", None),
    (1, 6, 1, "exact", None),
    (2, 3, 1, "exact", None),
    (4, 4, 1, "exact", None),
    (0, 5, 1, "exact", None),
]


def exact_match(example, prediction, trace=None) -> float:
    """Upstream reads the return value for Python truth when no threshold is set, so a float is
    what a metric owes it. The trace argument is upstream's optimizing-versus-evaluating
    convention; nothing here branches on it."""
    return float(example.answer == prediction.answer)


def graded(example, prediction, trace=None) -> float:
    """Half credit for an answer that is wrong but present, which the capital table always is.

    A score no threshold rejects and every threshold above it does, which is what separates
    reading a metric for truth from reading it against a bar.
    """
    if example.answer == prediction.answer:
        return 1.0
    return 0.5 if prediction.answer else 0.0


METRICS = {"exact": exact_match, "graded": graded}


# Every call any recording model sees, in order.
#
# Module level rather than an attribute, because every round after the first runs against
# `lm.copy(rollout_id=..., temperature=1.0)` and `BaseLM.copy` is a `deepcopy`. An instance
# attribute would be duplicated there and the retries would be recorded onto an object nothing
# reads — which is exactly what happened, and it under-reported a `max_rounds=2` compile as six
# attempts when dspy's own log said ten.
CALLS: list[list[dict]] = []


class RecordingLM(DummyLM):
    """A `DummyLM` that keeps what each call was shown.

    Leave-one-out withholding and the teacher's labelled priming both act on the demos a
    predictor holds *during* a call and put them back afterwards, so a fixture that only reports
    the compiled program cannot see either one. What each call saw is the evidence.
    """

    def forward(self, prompt=None, messages=None, **kwargs):
        CALLS.append(list(messages or [{"role": "user", "content": prompt}]))
        return super().forward(prompt=prompt, messages=messages, **kwargs)


def trainset() -> list[dspy.Example]:
    return [
        dspy.Example(question=question, answer=answer).with_inputs("question")
        for question, answer in TRAINSET
    ]


def demo_report(predictor) -> list[dict]:
    """The demos a predictor ended up holding, in order, with every field they carry.

    Which fields those are is itself the evidence once a program has more than one predictor: a
    demo the drafting half earned names a draft, and one the answering half earned does not name
    a question. `augmented` marks a demo the teacher earned rather than one drawn from the
    trainset, and whether the key is *there at all* is the distinction — dspy sets it only on the
    trace demos, so a labelled demo carries no such key.

    This used to add `augmented: False` to every demo that lacked it, which invented a field dspy
    does not write. The port dropped the marker entirely at the time, so the comparison filtered it
    out and nobody saw the invention.
    """
    return [dict(demo.toDict()) for demo in predictor.demos]


def demos_shown(messages: list[dict]) -> list[str]:
    """Which trainset questions one call was shown as demos, in order.

    The adapter gives every demo its own turn ahead of the live request, so the demo turns are
    what sits between the system message and the last one.
    """
    shown = []
    for message in messages[1:-1]:
        content = message.get("content") or ""
        for question, _ in TRAINSET:
            if question in content and question not in shown:
                shown.append(question)
    return shown


def call_report(messages: list[dict]) -> dict:
    asked = next(
        (question for question, _ in TRAINSET if question in (messages[-1].get("content") or "")),
        None,
    )
    return {"question": asked, "demos": demos_shown(messages)}


def compile_once(
    max_bootstrapped: int, max_labeled: int, max_rounds: int, metric: str, threshold: float | None
) -> tuple[list[dict], list[dict]]:
    CALLS.clear()
    dspy.configure(
        lm=RecordingLM({question: {"answer": answer} for question, answer in ANSWERS.items()})
    )
    student = dspy.Predict(Answer)
    optimizer = dspy.BootstrapFewShot(
        metric=METRICS[metric],
        metric_threshold=threshold,
        max_bootstrapped_demos=max_bootstrapped,
        max_labeled_demos=max_labeled,
        max_rounds=max_rounds,
    )
    compiled = optimizer.compile(student, trainset=trainset())
    predictors = [
        {"predictor": name, "demos": demo_report(predictor)}
        for name, predictor in compiled.named_predictors()
    ]
    return predictors, [call_report(messages) for messages in CALLS]


def compile_pair_once(
    max_bootstrapped: int, max_labeled: int, max_rounds: int, metric: str, threshold: float | None
) -> list[dict]:
    CALLS.clear()
    dspy.configure(lm=RecordingLM(PAIR_ANSWERS))
    optimizer = dspy.BootstrapFewShot(
        metric=METRICS[metric],
        metric_threshold=threshold,
        max_bootstrapped_demos=max_bootstrapped,
        max_labeled_demos=max_labeled,
        max_rounds=max_rounds,
    )
    compiled = optimizer.compile(Pair(), trainset=trainset())
    return [
        {"predictor": name, "demos": demo_report(predictor)}
        for name, predictor in compiled.named_predictors()
    ]


def main() -> None:
    if dspy.__version__ != PINNED:
        raise SystemExit(f"expected dspy {PINNED}, found {dspy.__version__}")

    cases = []
    for max_bootstrapped, max_labeled, max_rounds, metric, threshold in CONFIGS:
        predictors, calls = compile_once(
            max_bootstrapped, max_labeled, max_rounds, metric, threshold
        )
        cases.append(
            {
                "max_bootstrapped_demos": max_bootstrapped,
                "max_labeled_demos": max_labeled,
                "max_rounds": max_rounds,
                "metric": metric,
                "metric_threshold": threshold,
                "predictors": predictors,
                "calls": calls,
            }
        )

    pair_cases = []
    for max_bootstrapped, max_labeled, max_rounds, metric, threshold in PAIR_CONFIGS:
        pair_cases.append(
            {
                "max_bootstrapped_demos": max_bootstrapped,
                "max_labeled_demos": max_labeled,
                "max_rounds": max_rounds,
                "metric": metric,
                "metric_threshold": threshold,
                "predictors": compile_pair_once(
                    max_bootstrapped, max_labeled, max_rounds, metric, threshold
                ),
            }
        )

    fixture = {
        "source": f"generated from dspy=={PINNED} via scripts/generate_bootstrap_fixture.py",
        "dspy_version": PINNED,
        "trainset": [{"question": question, "answer": answer} for question, answer in TRAINSET],
        "answers": ANSWERS,
        "cases": cases,
        "pair_cases": pair_cases,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "bootstrap_few_shot.json"
    path.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"  wrote {path.relative_to(OUT.parent.parent.parent)} ({len(cases)} cases)", file=sys.stderr)


if __name__ == "__main__":
    main()
