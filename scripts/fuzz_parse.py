"""Generate random replies, run dspy's parsers over them, and record the answers to compare against.

The 51 hand-written cases in `chat_parse.json` were each chosen by reading a branch of `parse`. That
is the method that leaves the branches nobody thought of, and this is the answer to it: both sides
are pure functions from a string to a value-or-error, and dspy is on hand as the reference, which is
about as clean a differential-testing setup as exists.

This writes a *campaign* artifact, not a committed golden. Run it with a seed, run
`cargo test -p dsrust --test parse_fuzz` to see what disagrees, and promote anything it finds into
`generate_parse_fixture.py` as a named case with its reason. A corpus of ten thousand random strings
is evidence; it is not documentation, and it does not belong in git.

    .venv/bin/python scripts/fuzz_parse.py            # 2000 cases, seed 0
    .venv/bin/python scripts/fuzz_parse.py 20000 7    # more, different seed
"""

from __future__ import annotations

import json
import pathlib
import random
import sys

import dspy

OUT = pathlib.Path(__file__).parent.parent / "target" / "parse_fuzz.json"

FIELDS = ["reasoning", "answer"]


def names(rng: random.Random) -> list[str]:
    """Field names for one reply, **with repeats**.

    `rng.sample` draws without replacement, which is what this was, and it meant no generated reply
    ever named a field twice — so the first-occurrence-wins rule every adapter has went untested.
    Found by breaking that rule on purpose and watching the fuzzer stay green.
    """
    pool = FIELDS + ["completed", "unknown"]
    return [rng.choice(pool) for _ in range(rng.randint(1, 5))]
NOISE = ["", " ", "\n", "\t", "  ", "\n\n", "x", "…", "\\", '"', "'", "{", "}", "<", ">", "#"]
WORDS = ["Paris", "Because.", "7", "true", "None", "[]", "{}", '["a"]', "-1.5", "", "a b c"]


class QA(dspy.Signature):
    """Answer the question."""

    question: str = dspy.InputField()
    reasoning: str = dspy.OutputField()
    answer: str = dspy.OutputField()


def marker_reply(rng: random.Random) -> str:
    """A reply in the marker format, with the parts a model gets wrong."""
    parts = []
    for name in names(rng):
        indent = rng.choice(["", " ", "   ", "\t"])
        # A marker the model mangled: a missing bracket, an extra hash, a stray space.
        head = rng.choice(
            [
                f"[[ ## {name} ## ]]",
                f"[[ ## {name} ## ]",
                f"[ ## {name} ## ]]",
                f"[[ ##{name}## ]]",
                f"[[ ## {name} ##]]",
                f"[[ ## {name} ## ]]{rng.choice(NOISE)}",
            ]
        )
        body = rng.choice(WORDS)
        parts.append(f"{indent}{head}\n{body}")
    return rng.choice(["", "Sure! ", "```\n"]) + "\n\n".join(parts) + rng.choice(["", "\n", "\n```"])


def tag_reply(rng: random.Random) -> str:
    """A reply in the tag format, including the shapes the pattern refuses."""
    parts = []
    for name in names(rng):
        body = rng.choice(WORDS)
        parts.append(
            rng.choice(
                [
                    f"<{name}>{body}</{name}>",
                    f"<{name}>{body}",
                    f"<{name}>{body}</wrong>",
                    f"<{name} id='1'>{body}</{name}>",
                    f"<{name}><{name}>{body}</{name}></{name}>",
                    f"<{name}>{body}</{name}>{rng.choice(NOISE)}",
                ]
            )
        )
    return rng.choice(["", "Here: "]) + rng.choice(["\n", " "]).join(parts)


def json_reply(rng: random.Random) -> str:
    """A reply in JSON, mangled the ways a model mangles one."""
    pairs = [
        f'{rng.choice(["", chr(34)])}{name}{rng.choice(["", chr(34)])}: "{rng.choice(WORDS)}"'
        for name in names(rng)
    ]
    body = ", ".join(pairs) + rng.choice(["", ","])
    text = "{" + body + rng.choice(["}", "", "}}"])
    if rng.random() < 0.3:
        text = text.replace('"', "'")
    return rng.choice(["", "Sure! ", "```json\n", "[", "text "]) + text + rng.choice(
        ["", " done", "\n```", "]"]
    )


SHAPES = {"chat": marker_reply, "xml": tag_reply, "json": json_reply}


def parsed(adapter, completion: str) -> dict:
    """What the adapter answered, or how it refused — the message, not only the class.

    Every refusal these generators produce is an `AdapterParseError`, so a corpus recording only the
    class name gives the Rust side nothing to compare: any refusal matches any refusal, across the
    88% of each campaign that dspy rejects. The message is what says *which* field was missing and
    which adapter was reading.
    """
    try:
        return {"ok": True, "fields": adapter.parse(QA, completion)}
    except Exception as error:
        return {"ok": False, "error": type(error).__name__, "message": str(error)}


def main() -> None:
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    rng = random.Random(seed)
    adapters = {"chat": dspy.ChatAdapter(), "xml": dspy.XMLAdapter(), "json": dspy.JSONAdapter()}

    cases, seen = [], set()
    while len(cases) < count:
        which = rng.choice(list(SHAPES))
        completion = SHAPES[which](rng)
        if (which, completion) in seen:
            continue
        seen.add((which, completion))
        cases.append(
            {
                "adapter": which,
                "completion": completion,
                "expected": parsed(adapters[which], completion),
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "seed": seed,
                "signature": "question -> reasoning: str, answer: str",
                "cases": cases,
            },
            indent=1,
            ensure_ascii=False,
        )
        + "\n"
    )
    accepted = sum(1 for case in cases if case["expected"]["ok"])
    print(f"  wrote {OUT} — {len(cases)} cases, seed {seed}", file=sys.stderr)
    print(f"  dspy accepted {accepted}, refused {len(cases) - accepted}", file=sys.stderr)
    # A corpus dspy refuses outright, or accepts outright, exercises one arm and says little.
    if not 0.05 < accepted / len(cases) < 0.95:
        raise SystemExit(
            f"lopsided corpus: {accepted}/{len(cases)} accepted — vary the generators"
        )


if __name__ == "__main__":
    main()
