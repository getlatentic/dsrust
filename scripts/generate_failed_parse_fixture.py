"""What dspy does with a completion the adapter cannot parse, run rather than read.

`teleprompt/bootstrap_trace.py` is the trace collector GEPA reaches for. Its job is to keep going
when a forward ends in an `AdapterParseError`: it appends a `FailedPrediction` to the trace and
scores it with a format reward, so a program that answered unparseably still produces something to
reflect on. Two arms, and they behave differently enough that reading the source is not enough:

* **No declared field parsed.** `parsed_result` is empty, the constant arm runs, and the example
  survives as a `FailedPrediction` carrying the raw completion.
* **Some declared field parsed.** The graded arm computes
  `format_failure_score + (failure_score - format_failure_score) * (present / expected)` where both
  are `list(...)`, so it raises `TypeError`. `Evaluate` swallows it, the result fails to unpack, and
  the example is **dropped from the batch**.

Reaching the second needs the adapter fallback: `ChatAdapter.__call__` catches its own parse error
and retries through `JSONAdapter`, so the error that escapes is the fallback's. A reply that is
valid JSON carrying *some* of the declared output fields is what makes `parsed_result` truthy.

Also recorded is what a failure becomes in GEPA's reflective dataset — the raw-response block, the
structure instruction built from `ChatAdapter.format`, and what `add_format_failure_as_feedback`
changes — because that is the text a reflection model reads.

    .venv/bin/python scripts/generate_failed_parse_fixture.py
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
from dspy.teleprompt.bootstrap_trace import FailedPrediction, bootstrap_trace_data

from pins import require

PINNED = require("dspy")
OUT = (
    pathlib.Path(__file__).parent.parent
    / "crates"
    / "dsrust"
    / "tests"
    / "conformance"
    / "optimize"
    / "failed_parse.json"
)

#: A reply that is valid JSON holding one of the two declared output fields, so the `ChatAdapter`
#: fallback into `JSONAdapter` parses *something* and `parsed_result` comes back truthy.
SOME_FIELDS = '{"answer": "half"}'
#: Prose. No adapter finds a declared field, so `parsed_result` is empty and the constant arm runs.
NO_FIELDS = "there is nothing structured here"
GOOD = "[[ ## answer ## ]]\nyes\n\n[[ ## note ## ]]\nfine\n\n[[ ## completed ## ]]"


class ScriptedLM(dspy.BaseLM):
    """Answers with one fixed body, so which arm runs is the only variable."""

    def __init__(self, reply: str):
        super().__init__(model="scripted")
        self.reply = reply

    def forward(self, prompt=None, messages=None, **kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.reply, tool_calls=None),
                    finish_reason="stop",
                )
            ],
            usage={},
            model="scripted",
        )


class OneStep(dspy.Module):
    def __init__(self):
        self.p = dspy.Predict("question -> answer, note")

    def forward(self, question):
        return self.p(question=question)


def _batch(n: int) -> list[dspy.Example]:
    return [dspy.Example(question=f"q{i}").with_inputs("question") for i in range(n)]


def _collect(reply: str, failure_score: float, format_failure_score: float) -> dict:
    dspy.settings.configure(lm=ScriptedLM(reply), adapter=dspy.ChatAdapter())
    batch = _batch(3)
    trajectories = bootstrap_trace_data(
        OneStep(),
        batch,
        metric=lambda example, pred, trace=None: 1.0,
        raise_on_error=False,
        capture_failed_parses=True,
        failure_score=failure_score,
        format_failure_score=format_failure_score,
    )
    rows = []
    for data in trajectories:
        prediction = data["prediction"]
        failed = isinstance(prediction, FailedPrediction)
        rows.append(
            {
                "score": data.get("score"),
                "prediction_failed": failed,
                "completion_text": prediction.completion_text if failed else None,
                "format_reward": prediction.format_reward if failed else None,
                "trace": [
                    {"failed": isinstance(step[2], FailedPrediction)} for step in data["trace"]
                ],
            }
        )
    return {
        "batch_size": len(batch),
        "failure_score": failure_score,
        "format_failure_score": format_failure_score,
        "kept": len(trajectories),
        "trajectories": rows,
    }


def _reflective(add_format_failure_as_feedback: bool) -> dict:
    """What a failure looks like once GEPA turns it into a record for the reflection model."""
    from dspy.teleprompt.gepa.gepa_utils import DspyAdapter

    dspy.settings.configure(lm=ScriptedLM(NO_FIELDS), adapter=dspy.ChatAdapter())
    student = OneStep()
    adapter = DspyAdapter(
        student_module=student,
        metric_fn=lambda example, pred, trace=None, pred_name=None, pred_trace=None: 1.0,
        feedback_map={
            name: (lambda **kwargs: {"score": 0.0, "feedback": "unused"})
            for name, _ in student.named_predictors()
        },
        failure_score=0.0,
        add_format_failure_as_feedback=add_format_failure_as_feedback,
    )
    candidate = {name: p.signature.instructions for name, p in student.named_predictors()}
    evaluated = adapter.evaluate(_batch(2), candidate, capture_traces=True)
    out = {
        "add_format_failure_as_feedback": add_format_failure_as_feedback,
        "scores": evaluated.scores,
    }
    try:
        out["records"] = adapter.make_reflective_dataset(candidate, evaluated, ["p"])["p"]
    except Exception as e:  # noqa: BLE001 — the refusal *is* the measurement
        # With the flag off every failure step is filtered out, no example survives, and GEPA
        # refuses the whole reflection rather than proposing from nothing.
        out["records"] = None
        out["raises"] = {"type": type(e).__name__, "message": str(e)}
    return out


def _untraced(reply: str, failure_score: float) -> dict:
    """The valset path: `DspyAdapter.evaluate` without traces goes through `Evaluate`, which
    scores a row it cannot parse at `failure_score` and never drops it — the batch keeps the
    valset's length, which the per-testcase Pareto front indexes by."""
    from dspy.teleprompt.gepa.gepa_utils import DspyAdapter

    dspy.settings.configure(lm=ScriptedLM(reply), adapter=dspy.ChatAdapter())
    student = OneStep()
    adapter = DspyAdapter(
        student_module=student,
        metric_fn=lambda example, pred, trace=None, pred_name=None, pred_trace=None: 1.0,
        feedback_map={
            name: (lambda **kwargs: {"score": 0.0, "feedback": "unused"})
            for name, _ in student.named_predictors()
        },
        failure_score=failure_score,
    )
    candidate = {name: p.signature.instructions for name, p in student.named_predictors()}
    batch = _batch(3)
    evaluated = adapter.evaluate(batch, candidate, capture_traces=False)
    return {
        "batch_size": len(batch),
        "failure_score": failure_score,
        "kept": len(evaluated.scores),
        "scores": [float(score) for score in evaluated.scores],
    }


def main() -> None:
    arms = {
        "no_declared_field_parsed": _collect(NO_FIELDS, 0.0, -1.0),
        "some_declared_field_parsed": _collect(SOME_FIELDS, 0.0, -1.0),
        "a_parsing_run_for_contrast": _collect(GOOD, 0.0, -1.0),
        # `prediction.format_reward or format_failure_score` is Python truthiness, so a reward of
        # exactly zero is discarded and the constant is used instead. Both are zero here only
        # because the arm sets the reward *to* `format_failure_score`; the case is recorded so a
        # port writing `unwrap_or` rather than `or` is caught the day the two differ.
        "a_zero_reward_falls_back": _collect(NO_FIELDS, 0.25, 0.0),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "source": f"generated from dspy=={PINNED} via scripts/generate_failed_parse_fixture.py",
                "note": (
                    "`kept` below `batch_size` means dspy dropped examples: the graded arm raises "
                    "`TypeError: unsupported operand type(s) for /: 'list' and 'list'`, which "
                    "`Evaluate` swallows and `bootstrap_trace_data` cannot unpack."
                ),
                "arms": arms,
                "untraced": {
                    "some_declared_field_parsed": _untraced(SOME_FIELDS, 0.0),
                    "no_declared_field_parsed": _untraced(NO_FIELDS, 0.0),
                    "a_parsing_run_for_contrast": _untraced(GOOD, 0.0),
                },
                "reflective": [_reflective(False), _reflective(True)],
            },
            indent=2,
        )
        + "\n"
    )
    for name, arm in json.loads(OUT.read_text())["untraced"].items():
        print(f"  untraced {name:32s} kept {arm['kept']}/{arm['batch_size']}  scores {arm['scores']}")
    for name, arm in arms.items():
        print(f"  {name:32s} kept {arm['kept']}/{arm['batch_size']}  scores "
              f"{[row['score'] for row in arm['trajectories']]}")
    print(f"wrote {OUT.relative_to(pathlib.Path(__file__).parent.parent)}")


if __name__ == "__main__":
    main()
