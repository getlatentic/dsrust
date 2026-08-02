"""Generate random malformed JSON, run `json_repair` over it, and record what it answered.

`scripts/fuzz_parse.py` reaches this library only through `JSONAdapter.parse`, so it exercises one
shape of reply and nothing else: no nesting, no comments, no escapes, no tuples. That is enough to
check the *adapter* and far too little to check a 3,500-line heuristic parser. This grammar builds
malformed JSON directly, and every generator below names a decision in the library rather than a
kind of typo.

A campaign artifact, not a golden: run it with a seed, run
`cargo test -p dsrust-json-repair --test fuzz` to see what disagrees, and promote anything it finds
into `scripts/json_repair_corpus.py` as a named case with its reason.

    .venv/bin/python scripts/fuzz_json_repair.py            # 5000 cases, seed 0
    .venv/bin/python scripts/fuzz_json_repair.py 20000 7    # more, different seed
"""

from __future__ import annotations

import json
import pathlib
import random
import sys

import json_repair

OUT = pathlib.Path(__file__).parent.parent / "target" / "json_repair_fuzz.json"

#: The quote characters the library knows, including the two smart pairs and the low one that
#: opens a span no other rule closes.
QUOTES = ['"', "'", "“", "„", "”"]
SCALARS = ["1", "-2.5", "1e3", "1_000", "1,234", "3/4", "true", "True", "null", "None", "NaN", ""]
NOISE = ["", " ", "\n", "\t", ",", ":", "}", "]", "#", "//", "```", "\\", "...", "„", "”"]
#: Keys and values. The `_`-prefixed and digit-bearing ones are here because the bare-key scan
#: accepts `_` and `-` inside a name and alphanumerics to start one, and a grammar spelling every
#: key in plain letters never tells those rules apart.
WORDS = ["a", "answer", "Paris", "北京", "x y", "he said \"hi\"", "[a-z\"]+", "", "café",
         "_id", "k2", "a-b", "_", "9", "true story"]

#: Shapes that reach the lookahead helpers and nothing else does: a fenced snippet after a closing
#: brace, a container opened straight after a separator, a comment before a member, a stray `...`.
INTERJECTIONS = [
    "} ```json {\"k\": 1}```",
    ", {\"k\": [1, 2]}",
    ", [{\"k\": 1}]",
    ", # note\n\"k\": 1",
    ", // note\n`k`: 1",
    ", /* note */ k: 1",
    ", ...",
    ", \"k\": ",
]


def quoted(rng: random.Random, text: str) -> str:
    """A string, quoted the several ways a model gets wrong."""
    style = rng.random()
    if style < 0.45:
        return f'"{text}"'
    if style < 0.6:
        return f"'{text}'"
    if style < 0.7:
        return f"“{text}”"
    if style < 0.8:
        return text  # no quotes at all
    if style < 0.9:
        return f'"{text}'  # never closed
    return f'{text}"'  # never opened


def escaped(rng: random.Random) -> str:
    """A backslash run, which has its own normalisation rules.

    No `\\ud800`. A lone surrogate is a Python `str` and not a Rust `String`, which is a declared
    divergence with a named case and a test asserting it — leaving it in this grammar buries that
    one known gap under a hundred rediscoveries of it every run, and a fuzz report nobody reads is
    a fuzz report that has stopped working.
    """
    return rng.choice([r"\n", r"\t", r"\\", r"\\\\", r"\x41", r"\u00e9", r"\q", "\\"])


def value(rng: random.Random, depth: int) -> str:
    roll = rng.random()
    if depth > 0 and roll < 0.18:
        return container(rng, depth - 1)
    if roll < 0.3:
        return rng.choice(SCALARS)
    if roll < 0.4:
        return f"({', '.join(rng.choice(SCALARS) for _ in range(rng.randint(0, 3)))})"
    if roll < 0.5:
        return quoted(rng, rng.choice(WORDS) + escaped(rng))
    if roll < 0.58:
        # An unterminated value with something after it that only the lookahead can classify.
        return f'"{rng.choice(WORDS)}{rng.choice(INTERJECTIONS)}'
    return quoted(rng, rng.choice(WORDS))


def member(rng: random.Random, depth: int) -> str:
    key = quoted(rng, rng.choice(WORDS))
    separator = rng.choice([":", ":", ":", "", " :", ": "])
    return f"{key}{separator}{value(rng, depth)}"


def container(rng: random.Random, depth: int) -> str:
    """An object, an array, or a Python tuple, with the bracket a model sometimes forgets.

    The parenthesised form is a third of the library's container handling — `parser_parenthesized.py`
    and the two lookaheads that decide whether a `(` opens a tuple or prose — and no token table
    here held a `(`, so none of it ever saw a generated input. Mutation testing put a number on the
    consequence: `parenthesized.rs` carried the second-largest cluster of surviving mutants in the
    crate, against a module the committed corpus reached at 59.8% of its statements.

    A tuple is not a shape a model emits often, which is the argument for a low weight rather than
    for leaving it out: it is the shape `json_repair` grew a whole module to accept.
    """
    count = rng.randint(0, 3)
    roll = rng.random()
    if roll < 0.12:
        body = ", ".join(value(rng, depth) for _ in range(count))
        # A trailing comma is what tells a one-element tuple from a grouped value, so it is worth
        # more here than in the other two.
        if count == 1 and rng.random() < 0.5:
            body += ","
        opener, closer = "(", rng.choice([")", ")", ")", "", "]", "))"])
    elif roll < 0.6:
        body = ", ".join(member(rng, depth) for _ in range(count))
        opener, closer = "{", rng.choice(["}", "}", "}", "", "]", "}}"])
    else:
        body = ", ".join(value(rng, depth) for _ in range(count))
        opener, closer = "[", rng.choice(["]", "]", "]", "", "}"])
    if rng.random() < 0.15:
        body += ","
    if rng.random() < 0.1:
        body += rng.choice([" # note", " // note", " /* note */"])
    return f"{opener}{body}{closer}"


def reply(rng: random.Random) -> str:
    """One whole input, wrapped the ways a model wraps an answer."""
    body = container(rng, depth=2)
    # A second top-level value one time in six. `_parse_top_level` gathers repeated values into a
    # list, and decides between *appending* and *replacing* by asking `ObjectComparer.is_same_object`
    # — same type, same keys, values ignored — and by whether a comma separated them. A grammar of
    # one container per reply never reaches any of it.
    if rng.random() < 0.17:
        separator = rng.choice(["", " ", ", ", ",", "\n"])
        body += separator + container(rng, depth=1)
    prefix = rng.choice(["", "", "Sure! ", "```json\n", "Here (see below): ", "# note\n"])
    suffix = rng.choice(["", "", "\n```", " done", "\n", rng.choice(NOISE)])
    return prefix + body + suffix


def run(text: str, strict: bool = False) -> dict[str, object]:
    try:
        return {"ok": True, "dumps": json.dumps(json_repair.loads(text, strict=strict))}
    except Exception as error:  # noqa: BLE001 — the refusal is part of the record
        return {"ok": False, "error": type(error).__name__, "message": str(error)}


def main() -> None:
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    rng = random.Random(seed)

    cases, seen = [], set()
    while len(cases) < count:
        text = reply(rng)
        if text in seen:
            continue
        seen.add(text)
        # Half the corpus in strict mode. It is where the grammar was blindest: eight of the
        # library's raise sites are strict-only, and nothing generated ever reached one.
        strict = len(cases) % 2 == 1
        cases.append({"input": text, "strict": strict} | run(text, strict))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"seed": seed, "cases": cases}, indent=1, ensure_ascii=False) + "\n")

    # A corpus that is mostly valid JSON tests CPython's scanner; one whose answers are mostly the
    # empty string tests the parser giving up. Both look like a passing fuzz run and check nothing.
    malformed = sum(1 for case in cases if not valid_json(case["input"]))
    empty = sum(1 for case in cases if case.get("dumps") == '""')
    print(f"  wrote {OUT} — {len(cases)} cases, seed {seed}", file=sys.stderr)
    print(f"  {malformed} malformed, {empty} parsed to the empty string", file=sys.stderr)
    if malformed < len(cases) * 0.8:
        raise SystemExit(f"only {malformed}/{len(cases)} malformed — most never reach the repairs")
    if empty > len(cases) * 0.2:
        raise SystemExit(f"{empty}/{len(cases)} answered '' — the grammar mostly produces nothing")


def valid_json(text: str) -> bool:
    try:
        json.loads(text)
    except ValueError:
        return False
    return True


if __name__ == "__main__":
    main()
