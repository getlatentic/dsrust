"""Record what dspy's MIPROv2 compiles to, end to end, by running it.

MIPROv2 has three steps — bootstrap demo sets, propose instructions, search the combinations with
optuna — and the pieces are verified apart (the proposer signatures byte-for-byte, the demo sets,
the TPE sampler against optuna). This pins them together: does the crate's MIPROv2 select the
instruction dspy's does?

The model is instruction-sensitive on purpose. A question-keyed DummyLM would tie every instruction's
score and both sides would trivially keep the baseline; here the model answers correctly only when
the proposed instruction (carrying `GOOD`) is in force, so that proposal scores 100% against the
original's 0% and the search *must* select it. The Rust `Coach` model mirrors this rule exactly.

Needs optuna: `uv pip install --python .dspy-venv/bin/python optuna==4.5.0`.

    .dspy-venv/bin/python scripts/generate_mipro_fixture.py
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
from dspy.clients.base_lm import BaseLM
from dspy.dsp.utils.utils import dotdict

OUT = pathlib.Path(__file__).parent.parent / "tests" / "conformance" / "optimize"
PINNED = (pathlib.Path(__file__).parent / "DSPY_VERSION").read_text().strip()

TRAINSET = [("capital of France?", "Paris"), ("capital of Germany?", "Berlin"), ("capital of Spain?", "Madrid")]
TABLE = dict(TRAINSET)
PROPOSAL = "Answer with GOOD precision."


class Coach(BaseLM):
    """Proposes an instruction carrying `GOOD`; answers the task correctly only when `GOOD` is the
    instruction in force (so the proposal outscores the original). The Rust side mirrors this."""

    def __init__(self):
        super().__init__("coach", "chat", 0.0, 1000, True)

    def forward(self, prompt=None, messages=None, **kwargs):
        system, last = messages[0]["content"], messages[-1]["content"]
        if "generate a new instruction that will be used" in system:
            content = f"[[ ## proposed_instruction ## ]]\n{PROPOSAL}\n\n[[ ## completed ## ]]"
        else:
            question = next((q for q in TABLE if q in last), None)
            answer = TABLE[question] if (question and "GOOD" in system) else "wrong"
            content = f"[[ ## answer ## ]]\n{answer}\n\n[[ ## completed ## ]]"
        message = dotdict(content=content, tool_calls=None)
        return dotdict(
            choices=[dotdict(message=message, finish_reason="stop")],
            usage=dotdict(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            model="coach",
        )


class Program(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict("question -> answer")

    def forward(self, question):
        return self.predict(question=question)


def metric(example, prediction, trace=None) -> float:
    return float(example.answer == prediction.answer)


# (num_candidates, num_trials, seed).
CASES = [(2, 3, 9), (3, 5, 7)]


def compile_once(num_candidates: int, num_trials: int, seed: int) -> dict:
    dspy.configure(lm=Coach())
    trainset = [dspy.Example(question=q, answer=a).with_inputs("question") for q, a in TRAINSET]
    optimizer = dspy.MIPROv2(
        metric=metric, prompt_model=dspy.settings.lm, task_model=dspy.settings.lm,
        auto=None, num_candidates=num_candidates, num_threads=1, seed=seed,
        max_bootstrapped_demos=0, max_labeled_demos=0,
    )
    compiled = optimizer.compile(
        Program(), trainset=trainset, valset=trainset, num_trials=num_trials, minibatch=False,
        requires_permission_to_run=False, program_aware_proposer=False, data_aware_proposer=False,
        tip_aware_proposer=True, fewshot_aware_proposer=False,
    )
    return {
        "num_candidates": num_candidates,
        "num_trials": num_trials,
        "seed": seed,
        "compiled": [p.signature.instructions for _, p in compiled.named_predictors()],
    }


def main() -> None:
    if dspy.__version__ != PINNED:
        raise SystemExit(f"expected dspy {PINNED}, found {dspy.__version__}")
    fixture = {
        "source": f"generated from dspy=={PINNED} + optuna via scripts/generate_mipro_fixture.py",
        "dspy_version": PINNED,
        "trainset": [{"question": q, "answer": a} for q, a in TRAINSET],
        "proposal": PROPOSAL,
        "cases": [compile_once(*case) for case in CASES],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "mipro.json"
    path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
    print(f"  wrote {path.relative_to(OUT.parent.parent)}", file=sys.stderr)


if __name__ == "__main__":
    main()
