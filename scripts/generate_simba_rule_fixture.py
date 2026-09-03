"""What `append_a_rule` actually sends, and what it does with the answer.

The strategy is where SIMBA asks a model to write a rule, so every input it builds is prompt bytes.
Recorded by calling dspy's own `append_a_rule` with a recording model and reading the message off
the wire, rather than by re-deriving the thirteen inputs from the source.

Three things a reading gets wrong, and all three are in the golden:

  - Every non-string input goes through `orjson.dumps(..., OPT_INDENT_2)` — two-space JSON — so a
    trajectory reaches the model as indented text and not as one line.
  - When the better run did **not** score above the worse one, dspy blanks one of the two: its
    trace becomes empty, its prediction becomes `{"N/A": "Prediction not available"}`, and its
    score becomes the *string* `"N/A"` — in a field the signature declares `float`. Its handler has
    a second arm, blanking the *worse* run instead, which is **unreachable**: past the gate
    `bad < p90`, and that arm needs `good <= bad` and `good > p90`, so `p90 < good <= bad < p90`.
    An exhaustive sweep of the four scores confirms it, and the result is recorded below rather
    than argued.
  - The advice is appended to the existing instructions with a blank line between, not substituted.

    .venv/bin/python scripts/generate_simba_rule_fixture.py
"""

from __future__ import annotations

import json
import logging
import pathlib
import sys
import warnings

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

import dspy
from dspy.teleprompt.simba_utils import append_a_rule
from dspy.utils.dummies import DummyLM

from pins import require

PINNED = require("dspy")
OUT = (
    pathlib.Path(__file__).parent.parent
    / "crates" / "dsrust" / "tests" / "conformance" / "optimize" / "simba_rule.json"
)

ADVICE = {"self": "Name the seat of government, not the largest city."}


def run(better: float, worse: float, gates: tuple[float, float]) -> dict:
    """One `append_a_rule` call, with the message it sent and what it changed."""
    sent: list = []

    class Recorder(DummyLM):
        def __call__(self, prompt=None, messages=None, **kwargs):
            sent.append(messages)
            return super().__call__(prompt=prompt, messages=messages, **kwargs)

    lm = Recorder({"module_advice": {"discussion": "The worse run guessed.", "module_advice": ADVICE}})
    dspy.configure(lm=lm)

    program = dspy.Predict("question -> answer")
    example = dspy.Example(question="capital of Spain?", answer="Madrid").with_inputs("question")
    predictor = program.predictors()[0]
    predictor2name = {id(predictor): "self"}

    def outcome(score, answer):
        return {
            "prediction": dspy.Prediction(answer=answer),
            "trace": [(predictor, {"question": example.question}, {"answer": answer})],
            "score": score,
            "example": example,
            "output_metadata": {},
        }

    bucket = [outcome(better, "Madrid"), outcome(worse, "Barcelona")]
    applied = append_a_rule(
        bucket,
        program,
        predictor2name=predictor2name,
        name2predictor={"self": predictor},
        batch_10p_score=gates[0],
        batch_90p_score=gates[1],
        prompt_model=lm,
    )
    return {
        "better_score": better,
        "worse_score": worse,
        "gates": list(gates),
        "applied": bool(applied),
        "sent": sent[0] if sent else None,
        "instructions_after": program.predictors()[0].signature.instructions,
    }


CASES = [
    # The ordinary shape: the better run beat the worse, and both gates are open.
    {"name": "a_rule_is_written", "better": 1.0, "worse": 0.0, "gates": (0.0, 1.0)},
    # The blanking arm: the two tied, and the better is not above the 90th, so the *better* is the
    # one blanked — its score reaches a `float` field as the string "N/A".
    {"name": "the_better_run_is_blanked_on_a_tie", "better": 0.5, "worse": 0.5, "gates": (0.0, 1.0)},
    # The same tie above the 90th. dspy's tie handler has a second arm for this — blank the
    # *worse* run instead — and it is **unreachable**: past the gate `bad < p90`, and the arm needs
    # `good <= bad` and `good > p90`, which together give `p90 < good <= bad < p90`. The gate
    # declines here instead, which is what this case records.
    {"name": "the_gate_declines_before_the_second_blanking_arm", "better": 0.9, "worse": 0.9, "gates": (0.0, 0.5)},
    # Both refusals.
    {"name": "declined_at_the_10th", "better": 0.0, "worse": 0.0, "gates": (0.0, 1.0)},
    {"name": "declined_at_the_90th", "better": 1.0, "worse": 1.0, "gates": (0.0, 0.5)},
]


def reachable() -> bool:
    """Whether dspy's second blanking arm can be reached at all, by sweeping the four scores."""
    grid = [i / 20 for i in range(21)]
    return any(
        not (good <= p10 or bad >= p90) and good <= bad and good > p90
        for good in grid
        for bad in grid
        for p10 in grid
        for p90 in grid
    )


def main() -> None:
    cases = []
    for case in CASES:
        result = run(case["better"], case["worse"], case["gates"])
        cases.append({"name": case["name"], **result})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "source": f"dspy=={PINNED} teleprompt/simba_utils.py::append_a_rule",
                "dspy_version": PINNED,
                "note": (
                    "The message `append_a_rule` sent and the instructions it left behind, per "
                    "shape. Two-space JSON for every non-string input, the blanking arm's string "
                    "\"N/A\" in a float field, and advice appended after a blank line."
                ),
                "advice": ADVICE,
                "worse_blanking_arm_reachable": reachable(),
                "cases": cases,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    fired = sum(1 for case in cases if case["applied"])
    print(f"  wrote {OUT.name}: {len(cases)} shapes, {fired} that wrote a rule", file=sys.stderr)


if __name__ == "__main__":
    main()
