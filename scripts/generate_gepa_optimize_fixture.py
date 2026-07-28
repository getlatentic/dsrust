"""Record what dspy's GEPA teleprompter compiles, driven by scripted models — in the regime where
the search dynamics decide the answer.

GEPA evolves a predictor's instruction by reflecting on how the program did. Two scripted models
make a run deterministic and reproducible in Rust without real LLMs, and *discriminating* — a
single always-best proposal would let a search that ignores its seed still accept it. So:

  - the reflection model proposes a **distinct** instruction per ask (`GOOD-1`, `GOOD-2`, … in
    call order), so the candidate pool is real;
  - the task model answers question q correctly iff the instruction in force carries a marker
    whose **profile** contains q — proposals range from partial to perfect, so whether a proposal
    survives its minibatch (a seeded draw) decides whether it enters the pool;
  - what is recorded is the whole evolution — every candidate in discovery order, its parents, its
    validation score, the eval bookkeeping — not just the winner, so the Rust side is held to the
    search path. Different seeds genuinely leave different instructions compiled.

Merge stays on (`use_merge=True`, dspy's default); the student is a single `Predict`, so merge has
nothing to combine and the runs stay budget-bound.

    .venv/bin/python scripts/generate_gepa_optimize_fixture.py
"""

from __future__ import annotations

import json
import pathlib
import re
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

#: Which questions each proposed instruction answers correctly: `GOOD-k` solves PROFILES[k].
#: Proposals climb from partial toward perfect and then fall away, so acceptance turns on which
#: minibatch a seeded draw shows them; the seed instruction carries no marker and scores zero.
PROFILES = {
    1: {"capital of France?"},
    2: {"capital of Germany?"},
    3: {"capital of France?", "capital of Germany?"},
    4: {"capital of France?", "capital of Germany?", "capital of Spain?"},
}


def _reply(content: str) -> dotdict:
    message = dotdict(content=content, tool_calls=None)
    return dotdict(
        choices=[dotdict(message=message, finish_reason="stop")],
        usage=dotdict(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        model="scripted",
    )


class Coach(BaseLM):
    """The task model: answers question q correctly iff the instruction in force carries a marker
    whose profile contains q. Cache off — replies are deterministic, but keeping every call real
    keeps the call accounting honest."""

    def __init__(self):
        super().__init__("coach", "chat", 0.0, 1000, False)

    def forward(self, prompt=None, messages=None, **kwargs):
        system, last = messages[0]["content"], messages[-1]["content"]
        question = next((q for q in TABLE if q in last), None)
        marker = re.search(r"GOOD-(\d+)", system)
        solved = PROFILES.get(int(marker.group(1)), set()) if marker else set()
        answer = TABLE[question] if question in solved else "wrong"
        return _reply(f"[[ ## answer ## ]]\n{answer}\n\n[[ ## completed ## ]]")


class Reflector(BaseLM):
    """The reflection model: proposes `GOOD-k`, fenced, on its k-th ask. The tally lives in a dict
    so a shallow `.copy()` of the model shares it, and the cache is off so a repeated prompt can
    never swallow an increment — the same two shapes the MIPRO coach needs."""

    def __init__(self):
        super().__init__("reflector", "chat", 0.0, 1000, False)
        self.tally = {"calls": 0}
        self.proposals = []

    def forward(self, prompt=None, messages=None, **kwargs):
        self.tally["calls"] += 1
        proposal = f"Answer with GOOD-{self.tally['calls']} precision."
        self.proposals.append(proposal)
        return _reply(f"```\n{proposal}\n```")


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


#: (seed_instruction, minibatch_size, max_metric_calls, seed). Chosen from a sweep for distinct
#: dynamics: a first proposal accepted; a first proposal *rejected* and the budget dying with the
#: seed instruction still compiled (a pool of one); an accepted-but-not-best second proposal; two
#: rejections then the perfect proposal; and a rejected second reflection. The committed set must
#: leave more than one distinct instruction compiled, checked in main().
CASES = [
    ("Answer the question.", 2, 8, 0),
    ("Answer the question.", 2, 8, 2),
    ("Respond to the query.", 2, 12, 0),
    ("Answer the question.", 2, 12, 2),
    ("Solve it.", 2, 14, 1),
]


def compile_once(seed_instruction: str, minibatch_size: int, max_metric_calls: int, seed: int) -> dict:
    dspy.configure(lm=Coach())
    reflector = Reflector()
    trainset = [dspy.Example(question=q, answer=a).with_inputs("question") for q, a in TABLE.items()]
    optimizer = GEPA(
        metric=metric,
        reflection_lm=reflector,
        max_metric_calls=max_metric_calls,
        reflection_minibatch_size=minibatch_size,
        candidate_selection_strategy="pareto",
        skip_perfect_score=True,
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
        "reflection_calls": reflector.tally["calls"],
        "proposals": list(reflector.proposals),
        # The whole evolution, in discovery order: each candidate's instruction, where it came
        # from, what it scored on the valset, and what its discovery had cost.
        "candidates": [c.predict.signature.instructions for c in result.candidates],
        "parents": [[p for p in parents if p is not None] for parents in result.parents],
        "val_aggregate_scores": result.val_aggregate_scores,
        "best_idx": result.best_idx,
        "discovery_eval_counts": result.discovery_eval_counts,
        "num_full_val_evals": result.num_full_val_evals,
        "total_metric_calls": result.total_metric_calls,
        "compiled_instruction": compiled.predict.signature.instructions,
    }


def main() -> None:
    cases = [compile_once(*case) for case in CASES]
    distinct = {case["compiled_instruction"] for case in cases}
    # The whole point of the case set: if every case compiles the same instruction, the golden
    # cannot tell a seeded search from one that ignores its seed. Refuse to write it.
    if len(distinct) < 2:
        raise SystemExit(f"case set is not discriminating: every case compiled {distinct!r}")
    fixture = {
        "source": (
            f"generated from dspy=={PINNED} + gepa=={GEPA_PINNED} "
            "via scripts/generate_gepa_optimize_fixture.py"
        ),
        "dspy_version": PINNED,
        "gepa_version": GEPA_PINNED,
        "trainset": [{"question": q, "answer": a} for q, a in TABLE.items()],
        "profiles": {str(k): sorted(v) for k, v in PROFILES.items()},
        "cases": cases,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "gepa.json"
    path.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"  wrote {path.relative_to(OUT.parent.parent.parent)}", file=sys.stderr)
    for case in cases:
        print(
            f"  ({case['minibatch_size']},{case['max_metric_calls']},{case['seed']})"
            f" -> {case['compiled_instruction']!r} via {case['candidates']}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
