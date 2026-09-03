"""dspy `Hasher`, and the demo BootstrapFewShot picks with it.

`teleprompt/bootstrap.py` collapses a predictor that answered more than once in a single example:

    rng = random.Random(Hasher.hash(tuple(demos)))
    demos = [rng.choice(demos[:-1]) if rng.random() < 0.5 else demos[-1]]

Every part of that is load-bearing bytes. `Hasher.hash` is `sha256(pickle.dumps(value))`, so the
seed is the digest of a *pickle* — the class path, the memo table, the `_store` ordering and
protocol 4's framing all decide it. `random.Random(str)` then runs the version-2 rule from
`random.py`, `int.from_bytes(a + sha512(a).digest())`, before MT19937 ever sees a key.

So this records the whole chain by running it: the pickle bytes for a matrix of demo tuples, their
digests, the seeded generator's first draws, and the index dspy actually keeps. The matrix is
chosen to move the parts that are easy to get wrong — the opcode a string length selects, an
integer's width, a float, a value repeated across demos, a payload over the 64 KiB frame target.

    .venv/bin/python scripts/generate_hasher_fixture.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import pathlib
import pickle
import random
import sys
import warnings

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

import dspy

from pins import require

PINNED = require("dspy")
OUT = (
    pathlib.Path(__file__).parent.parent
    / "crates"
    / "dsrust"
    / "tests"
    / "conformance"
    / "optimize"
    / "hasher.json"
)


def fresh(text: str) -> str:
    """A string CPython has not interned, as one parsed out of a completion is.

    Identity is what the pickle memo keys on, so a case that used a literal here would record a
    back-reference the real path never produces. Every adapter was checked: `ChatAdapter`,
    `JSONAdapter` and `XMLAdapter` all return output names and values that are new objects, shared
    with neither the signature nor the previous parse nor each other.
    """
    return "".join(list(text))


def demo(inputs: dict, outputs: dict) -> dspy.Example:
    """A demo as `BootstrapFewShot._bootstrap_one_example` builds one.

    `augmented` first, then the call's inputs, then the parsed outputs, and no `with_inputs` after
    it — so `_input_keys` stays `None`. The caller passes the *same objects* for equal input values
    across demos, because that is what a program looping over one predictor does: the question
    flows from one variable into every hop.
    """
    return dspy.Example(augmented=True, **inputs, **outputs)


def case(name: str, inputs: list[dict], outputs: list[dict]) -> tuple[str, tuple, list[str]]:
    demos = tuple(demo(i, {fresh(k): v for k, v in o.items()}) for i, o in zip(inputs, outputs))
    return name, demos, sorted({k for i in inputs for k in i})


#: A value a program holds in one variable and passes to every hop, which is what makes the two
#: demos share an object rather than merely agree.
SHARED = "what is up?"
SHARED_LIST = ["one", "two"]

#: Each case is a tuple of demos, named for the part of the writer it moves.
CASES: list[tuple[str, tuple, list[str]]] = [
    case("one_demo", [{"question": "what?"}], [{"answer": fresh("this")}]),
    case(
        "an_input_value_shared_across_demos",
        # The realest shape there is: one predictor called twice with the same question.
        [{"question": SHARED, "hop": "first"}, {"question": SHARED, "hop": "second"}],
        [{"answer": fresh("alpha")}, {"answer": fresh("beta")}],
    ),
    case(
        "equal_outputs_never_share",
        [{"question": "a"}, {"question": "b"}],
        [{"answer": fresh("same"), "note": fresh("same")}, {"answer": fresh("same")}],
    ),
    case(
        "three_demos",
        [{"question": f"q{i}"} for i in range(3)],
        [{"answer": fresh(f"a{i}")} for i in range(3)],
    ),
    case(
        "four_demos_take_the_mark_arm",
        [{"question": f"q{i}"} for i in range(4)],
        [{"answer": fresh(f"a{i}")} for i in range(4)],
    ),
    case(
        "seven_demos",
        [{"question": f"q{i}"} for i in range(7)],
        [{"answer": fresh(f"a{i}")} for i in range(7)],
    ),
    case("no_fields_beyond_augmented", [{}, {}], [{}, {}]),
    case(
        "value_types",
        [{"a": None, "b": True, "c": False, "d": 0}, {"a": -1, "b": 255, "c": 256, "d": 65536}],
        [{"e": 2**31, "f": 1.5, "g": -0.0}, {"e": [1, "two", None], "f": {"in": {"deep": [True]}}}],
    ),
    case(
        "a_container_valued_input_shared_across_demos",
        [{"passages": SHARED_LIST}, {"passages": SHARED_LIST}],
        [{"answer": fresh("x")}, {"answer": fresh("y")}],
    ),
    case(
        "long_values_change_the_opcode",
        [{"short": "x" * 255, "long": "y" * 256}, {"short": "z" * 300, "long": "w"}],
        [{"answer": fresh("a")}, {"answer": fresh("b")}],
    ),
    case(
        "unicode_is_counted_in_utf8",
        [{"q": "héllo → 世界"}, {"q": "ß"}],
        [{"answer": fresh("naïve")}, {"answer": fresh("🙂")}],
    ),
    case(
        "over_the_frame_target",
        [{"big": "Q" * 70_000}, {"big": "R" * 3}],
        [{"answer": fresh("a")}, {"answer": fresh("b")}],
    ),
]


def _from_a_real_compile() -> tuple[str, tuple, list[str]]:
    """The demos an actual `BootstrapFewShot` compile hands to `Hasher.hash`.

    Every other case builds the shape by hand and could be wrong about it in the same way twice.
    This one runs the optimizer over a two-hop program — one predictor, called with the same
    question and a different `hop` — and captures the tuple upstream really passes. It is the case
    that decides whether the identity model is right, because nothing here chose which objects the
    demos hold.
    """
    from dspy.utils.dummies import DummyLM
    from dspy.utils.hasher import Hasher

    class TwoHop(dspy.Module):
        def __init__(self):
            self.gen = dspy.Predict("question, hop -> answer, note")

        def forward(self, question):
            self.gen(question=question, hop="first")
            second = self.gen(question=question, hop="second")
            return dspy.Prediction(answer=second.answer)

    captured: dict = {}
    original = Hasher.hash
    Hasher.hash = staticmethod(lambda value: (captured.setdefault("demos", value), original(value))[1])
    try:
        dspy.settings.configure(lm=DummyLM([{"answer": "same", "note": "same"}] * 8))
        dspy.BootstrapFewShot(metric=lambda example, pred, trace=None: True).compile(
            TwoHop(), trainset=[dspy.Example(question="q?").with_inputs("question")]
        )
    finally:
        Hasher.hash = original
    demos = captured["demos"]
    if len(demos) != 2:
        raise SystemExit(f"the two-hop compile produced {len(demos)} demos, not 2")
    return "captured_from_a_real_compile", demos, ["hop", "question"]


def picked(demos: tuple) -> dict:
    """dspy's collapse, run rather than described."""
    rng = random.Random(hashlib.sha256(pickle.dumps(demos)).hexdigest())
    first = rng.random()
    if first < 0.5:
        chosen = rng.choice(demos[:-1])
        index = next(i for i, d in enumerate(demos) if d is chosen)
    else:
        index = len(demos) - 1
    return {"first_random": first, "index": index}


def main() -> None:
    cases = []
    for name, demos, input_keys in CASES + [_from_a_real_compile()]:
        raw = pickle.dumps(demos)
        digest = hashlib.sha256(raw).hexdigest()
        row = {
            "name": name,
            "demos": [dict(d.toDict()) for d in demos],
            "input_keys": input_keys,
            "pickle_hex": raw.hex(),
            "hash": digest,
        }
        # `rng.choice` needs at least two demos, which is the only shape dspy calls this on.
        if len(demos) > 1:
            row |= picked(demos)
        cases.append(row)

    # `random.Random(str)` on its own, so the seeding rule is held apart from the pickling.
    seeds = []
    for seed in ["", "0", "a" * 64, cases[0]["hash"], "héllo"]:
        rng = random.Random(seed)
        seeds.append(
            {
                "seed": seed,
                "random": [rng.random() for _ in range(3)],
                "below_7": [random.Random(seed)._randbelow(7) for _ in range(1)],
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "source": f"generated from dspy=={PINNED} on CPython {sys.version.split()[0]} "
                f"via scripts/generate_hasher_fixture.py",
                "protocol": pickle.DEFAULT_PROTOCOL,
                "frame_size_target": pickle.Pickler.__module__ and 64 * 1024,
                "note": (
                    "`pickle_hex` is `pickle.dumps(tuple(demos))`; `hash` is its sha256 hexdigest, "
                    "which is what `random.Random` is seeded with. `index` is the demo "
                    "BootstrapFewShot keeps."
                ),
                "cases": cases,
                "string_seeds": seeds,
            },
            indent=2,
        )
        + "\n"
    )
    for c in cases:
        print(f"  {c['name']:36s} {len(c['pickle_hex'])//2:6d} bytes  {c['hash'][:16]}  {c.get('index','-')}")
    print(f"wrote {OUT.relative_to(pathlib.Path(__file__).parent.parent)}")


if __name__ == "__main__":
    main()
