"""Record what dspy's MIPROv2 compiles to, end to end, by running it — in the regime where the
search dynamics decide the answer.

MIPROv2 has three steps — bootstrap demo sets, propose instructions, search the combinations with
optuna — and the pieces are verified apart (the proposer signatures byte-for-byte, the demo sets,
the TPE sampler against optuna). This pins them together, and it must do so *discriminatingly*: a
single always-best proposal would let a search that ignores the seed, the trial budget and the
sampler still pick the winner. So:

  - the proposer answers each proposal ask with a **distinct** instruction (`GOOD-1`, `GOOD-2`, …
    in call order), so the candidate set is real;
  - each instruction has its own **score profile** (which questions it answers correctly), with a
    deliberate tie at the top, so which candidates the sampler tries and in which order decides
    the compiled instruction;
  - there are **more candidates than trials** in most cases, so the tried set itself is
    seed-dependent — different seeds genuinely compile different instructions;
  - the whole **trial sequence** is recorded off the optuna study (params and score per trial),
    not just the winner, so the Rust side is held to the search path, not the destination.

Needs optuna in the venv (`uv sync` provides it).

    .venv/bin/python scripts/generate_mipro_fixture.py
"""

from __future__ import annotations

import json
import logging
import pathlib
import re
import sys
import warnings

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

import dspy
import optuna
from dspy.propose.grounded_proposer import TIPS
from dspy.clients.base_lm import BaseLM
from dspy.dsp.utils.utils import dotdict

OUT = pathlib.Path(__file__).parent.parent / "crates" / "dsrust" / "tests" / "conformance" / "optimize"
PINNED = (pathlib.Path(__file__).parent / "DSPY_VERSION").read_text().strip()

TRAINSET = [("capital of France?", "Paris"), ("capital of Germany?", "Berlin"), ("capital of Spain?", "Madrid")]
TABLE = dict(TRAINSET)

#: Which questions each proposed instruction answers correctly: `GOOD-k` solves PROFILES[k].
#: 1 and 2 tie at the top so the trial *order* decides between them; 3 beats both but may go
#: untried when trials < candidates; 4 is a distractor; anything later scores zero. The baseline
#: instruction carries no marker and scores zero.
PROFILES = {
    1: {"capital of France?", "capital of Germany?"},
    2: {"capital of France?", "capital of Germany?"},
    3: {"capital of France?", "capital of Germany?", "capital of Spain?"},
    4: {"capital of Spain?"},
}


class Coach(BaseLM):
    """Proposes `GOOD-k` on the k-th proposal ask; answers question q correctly iff the instruction
    in force carries a marker whose profile contains q. The Rust side mirrors this exactly.

    Two shapes matter. The proposer runs each ask on `prompt_model.copy(...)` — a *shallow* copy —
    so the call tally lives in a dict every copy shares by reference; a plain int would rebind per
    copy and pin every proposal to `GOOD-1`. And the cache is off: the tally makes replies
    stateful, and a cache hit would skip `forward` and swallow an increment.
    """

    def __init__(self, table=None, profiles=None):
        super().__init__("coach", "chat", 0.0, 1000, False)
        self.tally = {"proposal_calls": 0}
        self.proposals = []
        self.table = TABLE if table is None else table
        self.profiles = PROFILES if profiles is None else profiles

    def forward(self, prompt=None, messages=None, **kwargs):
        system, last = messages[0]["content"], messages[-1]["content"]
        if "generate a new instruction that will be used" in system:
            self.tally["proposal_calls"] += 1
            proposal = f"Answer with GOOD-{self.tally['proposal_calls']} precision."
            self.proposals.append(proposal)
            content = f"[[ ## proposed_instruction ## ]]\n{proposal}\n\n[[ ## completed ## ]]"
        else:
            question = next((q for q in self.table if q in last), None)
            marker = re.search(r"GOOD-(\d+)", system)
            solved = self.profiles.get(int(marker.group(1)), set()) if marker else set()
            answer = self.table[question] if question in solved else "wrong"
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


#: (num_candidates, num_trials, seed, max_bootstrapped_demos, max_labeled_demos). Trials below
#: candidates so the tried set is seed-dependent; one case with trials above candidates so repeat
#: suggestions are exercised too. The committed set must compile more than one distinct instruction
#: across cases — checked in main().
#:
#: The last four are the few-shot regime, where the search space is *two* parameters per predictor
#: rather than one and the trial sequence therefore differs at the sampler. Zero-shot cases alone
#: could not tell an interleaved search space from a grouped one, because with one parameter per
#: predictor the two orders are the same list.
CASES = [
    (5, 3, 0, 0, 0),
    (5, 3, 1, 0, 0),
    (5, 3, 5, 0, 0),
    (5, 3, 9, 0, 0),
    (6, 8, 7, 0, 0),
    (4, 6, 3, 0, 0),
    (5, 3, 0, 4, 4),
    (5, 6, 1, 4, 4),
    (5, 4, 5, 2, 0),
    (4, 6, 3, 2, 4),
]

#: The minibatch regime needs a valset bigger than one batch, and `auto` needs one bigger than
#: `MIN_MINIBATCH_SIZE` *after* its own subsample — so these cases get their own 140-row trainset
#: rather than the three above. Answers stay `capital of C? -> CAP-C` so one table serves both.
BIG_TRAINSET = [(f"capital of C{i}?", f"CAP-{i}") for i in range(140)]
BIG_TABLE = dict(BIG_TRAINSET)

#: Every third question for `GOOD-1`, every fourth for `GOOD-2`, and so on: overlapping profiles
#: whose *means over a 35-row subsample* differ from their means over the whole 100, which is what
#: makes a minibatch run pick differently from a full one.
BIG_PROFILES = {
    marker: {q for i, (q, _) in enumerate(BIG_TRAINSET) if i % (marker + 2) == 0}
    for marker in range(1, 8)
}

#: (auto, minibatch, minibatch_size, minibatch_full_eval_steps, seed, max_bootstrapped, max_labeled,
#: valset_given).
#: `auto` set means `num_candidates` and `num_trials` must both be None — upstream raises otherwise.
#: The `None` auto cases pin minibatching on its own, where the counts are explicit and only the
#: batching differs; the preset cases pin the whole of `_set_hyperparams_from_run_mode`, whose valset
#: subsample is the first draw off the shared generator and moves every later one.
#: The two `valset_given=False` cases pin `_set_and_validate_datasets`, which is not a formality:
#: no valset means the *last 80%* of the trainset becomes the valset and the first 20% stays behind
#: to bootstrap from, so the two sets do not overlap and neither is the whole.
MINIBATCH_CASES = [
    ("light", None, 35, 5, 0, 0, 0, True),
    ("light", None, 35, 5, 7, 0, 0, True),
    ("medium", None, 35, 5, 3, 0, 0, True),
    ("light", None, 35, 5, 1, 4, 4, True),
    ("medium", None, 20, 2, 5, 4, 4, True),
    ("light", None, 35, 5, 2, 0, 0, False),
    (None, True, 35, 5, 0, 0, 0, True),
    (None, True, 40, 3, 2, 0, 0, True),
    (None, True, 25, 2, 4, 4, 4, True),
    (None, True, 35, 5, 6, 0, 0, False),
]

#: `max_bootstrapped_demos=0` with `max_labeled_demos>0` is not a configuration upstream supports:
#: `create_n_fewshot_demo_sets` reaches `rng.randint(min_num_samples, max_bootstrapped_demos)` for
#: every shuffled set, and `randint(1, 0)` raises `ValueError: empty range`. It is only reachable
#: because `zeroshot` requires *both* to be zero, so `(0, k>0)` falls through to the bootstrap path
#: with nothing to bootstrap. Measured, not read — it is why that combination is not a case here.


def compile_once(
    num_candidates: int,
    num_trials: int,
    seed: int,
    max_bootstrapped_demos: int,
    max_labeled_demos: int,
) -> dict:
    coach = Coach()
    dspy.configure(lm=coach)
    trainset = [dspy.Example(question=q, answer=a).with_inputs("question") for q, a in TRAINSET]

    # The study is created inside compile; capture it so the golden carries the whole trial
    # sequence rather than only the winner.
    studies = []
    orig_create_study = optuna.create_study

    def capture(*args, **kwargs):
        study = orig_create_study(*args, **kwargs)
        studies.append(study)
        return study

    optuna.create_study = capture
    try:
        optimizer = dspy.MIPROv2(
            metric=metric, prompt_model=dspy.settings.lm, task_model=dspy.settings.lm,
            auto=None, num_candidates=num_candidates, num_threads=1, seed=seed,
            max_bootstrapped_demos=max_bootstrapped_demos, max_labeled_demos=max_labeled_demos,
        )
        compiled = optimizer.compile(
            Program(), trainset=trainset, valset=trainset, num_trials=num_trials, minibatch=False,
            requires_permission_to_run=False, program_aware_proposer=False, data_aware_proposer=False,
            tip_aware_proposer=True, fewshot_aware_proposer=False,
        )
    finally:
        optuna.create_study = orig_create_study

    (study,) = studies
    trials = [{"params": dict(t.params), "score": t.value} for t in study.trials]
    return {
        "num_candidates": num_candidates,
        "num_trials": num_trials,
        "seed": seed,
        "max_bootstrapped_demos": max_bootstrapped_demos,
        "max_labeled_demos": max_labeled_demos,
        "proposals": list(coach.proposals),
        "trials": trials,
        "compiled": [p.signature.instructions for _, p in compiled.named_predictors()],
        # The demos the winning trial left on each predictor, as the field values a set carries.
        # The instruction alone would not say which demo set was chosen, and choosing the set is
        # half of what a few-shot run does.
        "compiled_demos": [
            [{k: v for k, v in demo.items()} for demo in p.demos]
            for _, p in compiled.named_predictors()
        ],
    }


def compile_minibatch(
    auto: str | None,
    minibatch: bool | None,
    minibatch_size: int,
    minibatch_full_eval_steps: int,
    seed: int,
    max_bootstrapped_demos: int,
    max_labeled_demos: int,
    valset_given: bool,
) -> dict:
    """One run in the minibatch regime, over the 140-row set.

    `auto` and the two explicit counts are mutually exclusive upstream — passing both raises — so
    the preset cases send `num_candidates=None` and `num_trials=None` and let the preset decide.
    """
    coach = Coach(BIG_TABLE, BIG_PROFILES)
    dspy.configure(lm=coach)
    trainset = [dspy.Example(question=q, answer=a).with_inputs("question") for q, a in BIG_TRAINSET]

    studies = []
    orig_create_study = optuna.create_study

    def capture(*args, **kwargs):
        study = orig_create_study(*args, **kwargs)
        studies.append(study)
        return study

    optuna.create_study = capture
    try:
        optimizer = dspy.MIPROv2(
            metric=metric, prompt_model=dspy.settings.lm, task_model=dspy.settings.lm,
            auto=auto, num_candidates=None if auto else 6, num_threads=1, seed=seed,
            max_bootstrapped_demos=max_bootstrapped_demos, max_labeled_demos=max_labeled_demos,
        )
        compiled = optimizer.compile(
            Program(), trainset=trainset, valset=trainset if valset_given else None,
            num_trials=None if auto else 9, minibatch=minibatch,
            minibatch_size=minibatch_size, minibatch_full_eval_steps=minibatch_full_eval_steps,
            requires_permission_to_run=False, program_aware_proposer=False,
            data_aware_proposer=False, tip_aware_proposer=True, fewshot_aware_proposer=False,
        )
    finally:
        optuna.create_study = orig_create_study

    (study,) = studies
    return {
        "auto": auto,
        "minibatch": minibatch,
        "minibatch_size": minibatch_size,
        "minibatch_full_eval_steps": minibatch_full_eval_steps,
        "seed": seed,
        "max_bootstrapped_demos": max_bootstrapped_demos,
        "max_labeled_demos": max_labeled_demos,
        "valset_given": valset_given,
        "proposals": list(coach.proposals),
        # Every trial the study holds, minibatch and full-evaluation alike, in optuna's own order —
        # which is by trial number, so a full evaluation lands *after* the trial that triggered it
        # even though `add_trial` is called first.
        "trials": [{"params": dict(t.params), "score": t.value} for t in study.trials],
        "compiled": [p.signature.instructions for _, p in compiled.named_predictors()],
        "compiled_demos": [
            [{k: v for k, v in demo.items()} for demo in p.demos]
            for _, p in compiled.named_predictors()
        ],
    }


def main() -> None:
    if dspy.__version__ != PINNED:
        raise SystemExit(f"expected dspy {PINNED}, found {dspy.__version__}")
    cases = [compile_once(*case) for case in CASES]
    minibatch_cases = [compile_minibatch(*case) for case in MINIBATCH_CASES]
    distinct = {case["compiled"][0] for case in cases}
    # The whole point of the case set: if every case compiles the same instruction, the golden
    # cannot tell a seeded search from one that ignores its seed. Refuse to write it.
    if len(distinct) < 2:
        raise SystemExit(f"case set is not discriminating: every case compiled {distinct!r}")
    fixture = {
        "source": f"generated from dspy=={PINNED} + optuna=={optuna.__version__} via scripts/generate_mipro_fixture.py",
        "dspy_version": PINNED,
        "optuna_version": optuna.__version__,
        "trainset": [{"question": q, "answer": a} for q, a in TRAINSET],
        "profiles": {str(k): sorted(v) for k, v in PROFILES.items()},
        # The proposer's tip texts, in declaration order — which is the order `list(TIPS.keys())`
        # yields and therefore the order `random.choice` indexes into, so the order is as much of
        # the golden as the strings. Recorded rather than transcribed: a tip upstream reworded
        # would otherwise diverge silently, since nothing else in this fixture renders one.
        "tips": list(TIPS.values()),
        "cases": cases,
        "minibatch_trainset": [{"question": q, "answer": a} for q, a in BIG_TRAINSET],
        "minibatch_profiles": {str(k): sorted(v) for k, v in BIG_PROFILES.items()},
        "minibatch_cases": minibatch_cases,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "mipro.json"
    path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
    print(f"  wrote {path.relative_to(OUT.parent.parent)}", file=sys.stderr)
    for case in cases:
        print(
            f"  ({case['num_candidates']},{case['num_trials']},{case['seed']}) -> {case['compiled'][0]!r}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
