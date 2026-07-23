"""Record what dspy's own COPRO decides, by running it.

`teleprompt/test_copro_optimizer.py` passing would prove nothing about `src/optimize/copro/`:
those tests drive *dspy's* optimizer. This closes that by replaying an identical trace through
both implementations and comparing every prompt each produces, in order.

The trace is made deterministic by removing a real model. A `DummyLM` answers from a table keyed
by a token that appears in exactly one kind of call:

  * `basic_instruction` — the field marker unique to the seed prompt (`BasicGenerateInstruction`),
  * `attempted_instructions` — the field marker unique to the depth prompt,
  * each trainset question — appears only in that question's task call.

So no message matches two keys, and the crate's `BTreeMap` ordering and dspy's dict ordering pick
the same answer regardless. With the model fixed, COPRO is a pure function of the trainset, the
metric and the configuration, and the evaluations run single-threaded to keep the call order
stable. What is left to compare is the exact sequence of prompts, and the compiled instruction.

    .dspy-venv/bin/python scripts/generate_copro_fixture.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import dspy
from dspy.utils.dummies import DummyLM

OUT = pathlib.Path(__file__).parent.parent / "tests" / "conformance" / "optimize"
PINNED = (pathlib.Path(__file__).parent / "DSPY_VERSION").read_text().strip()

# One trainset the table half-solves: France it answers right, Spain it answers wrong (the label is
# Madrid), so every instruction scores 50.0 and the score that reaches a depth prompt is exercised.
TRAINSET = [("capital of France?", "Paris"), ("capital of Spain?", "Madrid")]

# Keyed on markers no other call carries. The seed and depth prompts each propose one instruction
# and prefix; the task answers "Paris" to both questions, right for France and wrong for Spain.
KEYED = {
    "basic_instruction": {
        "proposed_instruction": "Proposed at seed",
        "proposed_prefix_for_output_field": "Answer:",
    },
    "attempted_instructions": {
        "proposed_instruction": "Refined from attempts",
        "proposed_prefix_for_output_field": "Response:",
    },
    "capital of France?": {"answer": "Paris"},
    "capital of Spain?": {"answer": "Paris"},
}

# (instruction, breadth, depth). The instruction is the student's starting point; breadth and depth
# are COPRO's search size. Two shapes: one with a depth round and one without.
CASES = [
    ("Answer the question directly.", 3, 2),
    ("Give a short answer.", 2, 1),
]


def exact_match(example, prediction, trace=None) -> float:
    return float(example.answer == prediction.answer)


# Every prompt any recording model saw, in order — system message and final user message, which is
# where COPRO's decisions show: the instruction in force on a task call, the attempts on a depth one.
CALLS: list[dict] = []


class RecordingLM(DummyLM):
    def forward(self, prompt=None, messages=None, **kwargs):
        turns = messages or [{"role": "user", "content": prompt}]
        CALLS.append({"system": turns[0]["content"], "user": turns[-1]["content"]})
        return super().forward(prompt=prompt, messages=messages, **kwargs)


def trainset() -> list[dspy.Example]:
    return [
        dspy.Example(question=question, answer=answer).with_inputs("question")
        for question, answer in TRAINSET
    ]


def build_student(instruction: str) -> dspy.Predict:
    signature = dspy.Signature("question -> answer").with_instructions(instruction)
    return dspy.Predict(signature)


def compile_once(instruction: str, breadth: int, depth: int) -> dict:
    CALLS.clear()
    dspy.configure(lm=RecordingLM(dict(KEYED)))
    optimizer = dspy.COPRO(metric=exact_match, breadth=breadth, depth=depth, init_temperature=1.4)
    eval_kwargs = dict(num_threads=1, display_progress=False, display_table=0)
    compiled = optimizer.compile(build_student(instruction), trainset=trainset(), eval_kwargs=eval_kwargs)
    final = [predictor.signature.instructions for _, predictor in compiled.named_predictors()]
    return {
        "instruction": instruction,
        "signature": "question -> answer",
        "breadth": breadth,
        "depth": depth,
        "trainset": [{"question": q, "answer": a} for q, a in TRAINSET],
        "keyed": [{"key": key, "fields": fields} for key, fields in KEYED.items()],
        "calls": list(CALLS),
        "final": final,
    }


def main() -> None:
    if dspy.__version__ != PINNED:
        raise SystemExit(f"expected dspy {PINNED}, found {dspy.__version__}")
    fixture = {
        "source": f"generated from dspy=={PINNED} via scripts/generate_copro_fixture.py",
        "dspy_version": PINNED,
        "cases": [compile_once(instruction, breadth, depth) for instruction, breadth, depth in CASES],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "copro.json"
    path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
    print(f"  wrote {path.relative_to(OUT.parent.parent)}", file=sys.stderr)


if __name__ == "__main__":
    main()
