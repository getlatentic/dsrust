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

OUT = pathlib.Path(__file__).parent.parent / "crates" / "dsrust" / "tests" / "conformance" / "optimize"
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

# A two-predictor program: the first drafts, the second settles. Its keyed table answers each half
# — a question becomes a draft, that draft becomes an answer — plus the two proposal markers. One
# trainset row keeps the call count down; breadth 2, depth 2 is what makes the multi-predictor paths
# fire: `all_candidates` accumulates across the two rounds, and the first predictor's chosen
# instruction is in force while the second is scored.
PAIR_KEYED = {
    "basic_instruction": {
        "proposed_instruction": "Proposed at seed",
        "proposed_prefix_for_output_field": "Answer:",
    },
    "attempted_instructions": {
        "proposed_instruction": "Refined from attempts",
        "proposed_prefix_for_output_field": "Response:",
    },
    "capital of France?": {"draft": "France draft"},
    "France draft": {"answer": "Paris"},
}
PAIR_TRAINSET = [("capital of France?", "Paris")]
# (first_instruction, second_instruction, breadth, depth).
PAIR_CASES = [("Draft the answer.", "Settle the answer.", 2, 2)]


class Pair(dspy.Module):
    def __init__(self, first_instruction: str, second_instruction: str):
        super().__init__()
        self.first = dspy.Predict(dspy.Signature("question -> draft").with_instructions(first_instruction))
        self.second = dspy.Predict(dspy.Signature("draft -> answer").with_instructions(second_instruction))

    def forward(self, question):
        return self.second(draft=self.first(question=question).draft)


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


def run(module: dspy.Module, keyed: dict, trainset_rows: list, breadth: int, depth: int) -> tuple[list[str], list[dict]]:
    CALLS.clear()
    dspy.configure(lm=RecordingLM(dict(keyed)))
    optimizer = dspy.COPRO(metric=exact_match, breadth=breadth, depth=depth, init_temperature=1.4)
    eval_kwargs = dict(num_threads=1, display_progress=False, display_table=0)
    devset = [dspy.Example(question=q, answer=a).with_inputs("question") for q, a in trainset_rows]
    compiled = optimizer.compile(module, trainset=devset, eval_kwargs=eval_kwargs)
    final = [predictor.signature.instructions for _, predictor in compiled.named_predictors()]
    return final, list(CALLS)


def compile_once(instruction: str, breadth: int, depth: int) -> dict:
    final, calls = run(build_student(instruction), KEYED, TRAINSET, breadth, depth)
    return {
        "module": "predict",
        "instructions": [instruction],
        "breadth": breadth,
        "depth": depth,
        "trainset": [{"question": q, "answer": a} for q, a in TRAINSET],
        "keyed": [{"key": key, "fields": fields} for key, fields in KEYED.items()],
        "calls": calls,
        "final": final,
    }


def compile_pair_once(first: str, second: str, breadth: int, depth: int) -> dict:
    final, calls = run(Pair(first, second), PAIR_KEYED, PAIR_TRAINSET, breadth, depth)
    return {
        "module": "pair",
        "instructions": [first, second],
        "breadth": breadth,
        "depth": depth,
        "trainset": [{"question": q, "answer": a} for q, a in PAIR_TRAINSET],
        "keyed": [{"key": key, "fields": fields} for key, fields in PAIR_KEYED.items()],
        "calls": calls,
        "final": final,
    }


def main() -> None:
    if dspy.__version__ != PINNED:
        raise SystemExit(f"expected dspy {PINNED}, found {dspy.__version__}")
    cases = [compile_once(instruction, breadth, depth) for instruction, breadth, depth in CASES]
    cases += [compile_pair_once(first, second, breadth, depth) for first, second, breadth, depth in PAIR_CASES]
    fixture = {
        "source": f"generated from dspy=={PINNED} via scripts/generate_copro_fixture.py",
        "dspy_version": PINNED,
        "cases": cases,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "copro.json"
    path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
    print(f"  wrote {path.relative_to(OUT.parent.parent)}", file=sys.stderr)


if __name__ == "__main__":
    main()
