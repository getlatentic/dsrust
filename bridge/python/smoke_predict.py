"""Smoke-test the module-crossing foundation: our Predict::forward driven by a real DummyLM.

Not part of the suite — a standalone check that predict_forward renders, calls back into the
Python LM, parses, and returns the output fields. Run with PYTHONPATH including this dir.
"""

import json

import dspy
from dspy.utils.dummies import DummyLM

import dsrs_bridge
from reflect import describe, described_outputs


def run() -> None:
    signature = dspy.Signature("question -> answer")
    lm = DummyLM([{"answer": "blue"}, {"answer": "red"}])

    values = [("question", json.dumps("What color is the sky?"), False)]
    output_json, raw = dsrs_bridge.predict_forward(
        signature.instructions,
        describe(signature.input_fields),
        described_outputs(signature),
        values,
        lm,
        None,
        None,
    )
    output = json.loads(output_json)
    print("output:", output)
    print("raw:", repr(raw))
    assert output.get("answer") == "blue", f"expected blue, got {output!r}"

    # The second call draws the next canned answer, and lm.history recorded both — proof it went
    # through BaseLM.__call__, not DummyLM.forward.
    output2, _ = dsrs_bridge.predict_forward(
        signature.instructions,
        describe(signature.input_fields),
        described_outputs(signature),
        values,
        lm,
        None,
        None,
    )
    assert json.loads(output2).get("answer") == "red", f"expected red, got {output2!r}"
    assert len(lm.history) == 2, f"expected 2 history entries, got {len(lm.history)}"
    print("SMOKE OK — our Predict ran, DummyLM answered, history recorded")


if __name__ == "__main__":
    run()
