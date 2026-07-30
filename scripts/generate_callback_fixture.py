"""Record which callback handlers a dspy program fires, in what order and under which parent.

A `BaseCallback` subclass is a Rust trait with defaulted methods, so the port is near-mechanical and
the thing that can still be wrong is *where* the handlers fire: how many times, in which order, and
what nests inside what. Upstream asserts one such sequence by hand in `tests/callback/test_callback.py`
— fourteen handlers for a `ChainOfThought(n=3)` — and that assertion is the only place the shape is
written down. This runs the same recording over the programs the crate has, so the Rust side is held
to a sequence dspy produced rather than to one a porter believed.

The sequence is what crosses, not the payloads: upstream hands a handler Python objects — the module
instance, a kwargs dict, a `Prediction` — and those have no Rust counterpart to compare byte for
byte. What each handler is *given* is held by the crate's own tests; what fires and when is here.

    .venv/bin/python scripts/generate_callback_fixture.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import dspy
from dspy.utils.callback import ACTIVE_CALL_ID, BaseCallback
from dspy.utils.dummies import DummyLM

from pins import require

OUT = (
    pathlib.Path(__file__).parent.parent / "crates" / "dsrust" / "tests" / "conformance" / "observe"
)
PINNED = require("dspy")


class Recording(BaseCallback):
    """Every handler, recording its name and the call it happened under.

    `ACTIVE_CALL_ID` at the moment a start handler runs is the *parent*: `with_callbacks` fires the
    start callbacks before it sets the new id, which is exactly what upstream's `test_active_id`
    asserts. So reading it here gives the tree the program ran as.
    """

    def __init__(self):
        self.calls = []
        self.depth = {}

    def _started(self, handler, call_id):
        parent = ACTIVE_CALL_ID.get()
        depth = 0 if parent is None else self.depth[parent] + 1
        self.depth[call_id] = depth
        self.calls.append({"handler": handler, "depth": depth})

    def _ended(self, handler, call_id):
        self.calls.append({"handler": handler, "depth": self.depth[call_id]})

    def on_module_start(self, call_id, instance, inputs):
        self._started("on_module_start", call_id)

    def on_module_end(self, call_id, outputs, exception):
        self._ended("on_module_end", call_id)

    def on_lm_start(self, call_id, instance, inputs):
        self._started("on_lm_start", call_id)

    def on_lm_end(self, call_id, outputs, exception):
        self._ended("on_lm_end", call_id)

    def on_adapter_format_start(self, call_id, instance, inputs):
        self._started("on_adapter_format_start", call_id)

    def on_adapter_format_end(self, call_id, outputs, exception):
        self._ended("on_adapter_format_end", call_id)

    def on_adapter_parse_start(self, call_id, instance, inputs):
        self._started("on_adapter_parse_start", call_id)

    def on_adapter_parse_end(self, call_id, outputs, exception):
        self._ended("on_adapter_parse_end", call_id)

    def on_tool_start(self, call_id, instance, inputs):
        self._started("on_tool_start", call_id)

    def on_tool_end(self, call_id, outputs, exception):
        self._ended("on_tool_end", call_id)

    def on_evaluate_start(self, call_id, instance, inputs):
        self._started("on_evaluate_start", call_id)

    def on_evaluate_end(self, call_id, outputs, exception):
        self._ended("on_evaluate_end", call_id)


def recorded(build, run, lm):
    """One program's handler sequence, recorded from a clean settings state.

    The program is built *inside* the recording because upstream fires nothing at construction and a
    difference there would be worth catching; `dspy.configure` is reset afterwards so one case cannot
    leave a callback installed for the next.
    """
    callback = Recording()
    dspy.configure(lm=lm, callbacks=[callback], adapter=dspy.ChatAdapter())
    try:
        run(build())
    finally:
        dspy.configure(callbacks=[])
    return callback.calls


def a_predict():
    return dspy.Predict("question -> answer")


def a_chain_of_thought():
    """Upstream's own case, and the only sequence dspy writes down: `test_callback_complex_module`
    asserts these fourteen handlers for `n=3`."""
    return dspy.ChainOfThought("question -> answer", n=3)


class Nested(dspy.Module):
    """Two predictors under one module, which is what makes the parent linkage observable: both
    children name the parent's call and neither names the other's."""

    def __init__(self):
        self.first = dspy.Predict("question -> answer")
        self.second = dspy.Predict("question -> answer")

    def forward(self, question):
        first = self.first(question=question)
        return self.second(question=first.answer)


def an_evaluation():
    """`Evaluate` wraps the whole devset, so every module call is inside one evaluate point."""
    return dspy.Evaluate(
        devset=[dspy.Example(question="How are you?", answer="test output").with_inputs("question")],
        metric=lambda example, prediction, trace=None: 1.0,
        num_threads=1,
        display_progress=False,
    )


CASES = [
    {
        "name": "predict",
        "note": "one predictor: the module, its render, the model call, one parse",
        "build": a_predict,
        "run": lambda program: program(question="How are you?"),
        "answers": {"How are you?": {"answer": "test output"}},
    },
    {
        "name": "chain_of_thought_n3",
        "note": "upstream's own asserted sequence: parsing runs once per output",
        "build": a_chain_of_thought,
        "run": lambda program: program(question="How are you?"),
        "answers": {
            "How are you?": {"answer": "test output", "reasoning": "No more responses"}
        },
    },
    {
        "name": "nested_modules",
        "note": "two predictors under one module, so both children name the same parent",
        "build": Nested,
        "run": lambda program: program(question="How are you?"),
        "answers": {
            "How are you?": {"answer": "second question"},
            "second question": {"answer": "test output"},
        },
    },
    {
        "name": "evaluate",
        "note": "a devset run, with the module calls it made inside it",
        "build": an_evaluation,
        "run": lambda program: program(dspy.Predict("question -> answer")),
        "answers": {"How are you?": {"answer": "test output"}},
    },
]


def main():
    cases = []
    for case in CASES:
        lm = DummyLM(case["answers"])
        cases.append(
            {
                "name": case["name"],
                "note": case["note"],
                "handlers": recorded(case["build"], case["run"], lm),
            }
        )

    fixture = {
        "source": f"generated from dspy=={PINNED} via scripts/generate_callback_fixture.py",
        "dspy_version": PINNED,
        "note": (
            "Each case is the sequence of BaseCallback handlers one program fired, with the nesting "
            "depth of the call each belongs to. `chain_of_thought_n3` is the sequence upstream "
            "asserts by hand in tests/callback/test_callback.py."
        ),
        "cases": cases,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "callbacks.json"
    path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
    print(f"  wrote {path.relative_to(OUT.parent.parent.parent)}", file=sys.stderr)


if __name__ == "__main__":
    main()
