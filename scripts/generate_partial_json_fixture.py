"""What `jiter`'s trailing-strings partial parse sees in a half-written JSON object.

dspy's JSON stream listener decides a field has ended by partial-parsing everything it has
accumulated and asking whether a *second* key appeared:

    parsed = jiter.from_json(accumulated, partial_mode="trailing-strings")
    if len(parsed) > 1: ...  # the next field started, so ours is done

That predicate is the whole of what the listener needs, and it is not the same question
`dsrust-json-repair` answers. The repairer reproduces Python's `json_repair`, a different library,
and the two disagree exactly where the decision is made:

    '{"answer": "x", "judgement":'    jiter -> 1 key      json_repair -> 2 keys

jiter's rule is that a key whose value has not *begun* is not yet a key; the repairer fills it with
an empty string. A listener built on the repairer would close the field one delta early. Four other
accumulated shapes agree, which is what makes it worth pinning rather than eyeballing.

So this records jiter's answer for **every prefix** of each accumulated string, which is exactly the
sequence a listener walks as deltas arrive. A Rust scanner reproducing the predicate is held to
this, prefix by prefix.

    .venv/bin/python scripts/generate_partial_json_fixture.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import jiter

from pins import require

OUT = pathlib.Path(__file__).parent.parent / "crates" / "dsrust" / "tests" / "conformance" / "lm"
PINNED = require("dspy")

#: Accumulated strings a listener actually builds. The first is upstream's own recorded stream from
#: `test_stream_listener_returns_correct_chunk_json_adapter`, joined; the rest are the shapes that
#: decide the predicate — a key with no value, a number, a literal, an escape, a nested object.
ACCUMULATED = {
    "recorded_gpt_4o_mini": '{"answer":"To get to the other side of the frying pan!"}\n',
    "second_key_begun": '{"answer": "x", "judgement": "fun',
    "second_key_named_only": '{"answer": "x", "judgement":',
    "second_key_partially_named": '{"answer": "x", "judg',
    "trailing_comma": '{"answer": "x",',
    "numeric_value": '{"answer": 42, "judgement": 7',
    "literal_values": '{"answer": true, "judgement": nul',
    "escaped_quote_in_value": '{"answer": "he said \\"hi\\"", "judgement": "y',
    "nested_object_value": '{"answer": {"inner": "x"}, "judgement": "y',
    "brace_inside_a_string": '{"answer": "a } brace", "judgement": "y',
    "empty_object": "{",
}


def seen(text: str) -> dict:
    """jiter's answer for one prefix: the keys it can see, and whether a strict parse succeeds."""
    try:
        parsed = jiter.from_json(text.encode("utf-8"), partial_mode="trailing-strings")
        keys = list(parsed) if isinstance(parsed, dict) else None
    except ValueError:
        keys = None
    try:
        jiter.from_json(text.encode("utf-8"))
        complete = True
    except ValueError:
        complete = False
    return {"keys": keys, "complete": complete}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fixture = {
        "_source": f"jiter, the partial parser dspy {PINNED}'s JSON stream listener decides on, "
        f"via {pathlib.Path(__file__).name}",
        "cases": [
            {
                "name": name,
                "accumulated": text,
                # Every prefix, because a listener sees them all in turn and the predicate has to
                # flip on the same one dspy's does.
                "prefixes": [
                    {"text": text[:length], **seen(text[:length])}
                    for length in range(1, len(text) + 1)
                ],
            }
            for name, text in ACCUMULATED.items()
        ],
    }
    path = OUT / "partial_json.json"
    path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
    print(f"  wrote {path.relative_to(OUT.parent.parent)}", file=sys.stderr)


if __name__ == "__main__":
    main()
