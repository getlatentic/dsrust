"""What dspy's `StreamListener` hands a caller, chunk by chunk, captured by running it.

`tests/streaming/test_streaming.py` is excused from the bridge because most of it drives dspy's
async `streamify` plumbing around a Python program. What the excuse hid is the part that is pure
logic and is the whole point of a stream listener: **where the chunk boundaries fall**. A caller
renders what it is handed, so a listener that regroups `"To"`, `" get"`, `" to"` into `"To"`,
`" ge"`, `"t to"` concatenates to the same text and splits words down the middle. This crate did
exactly that, and nothing said so, because nothing compared the boundaries to dspy's.

`receive` is driven directly rather than through `streamify`: the listener is the thing being
pinned, and going through the program would make the fixture a test of asyncio.

    .venv/bin/python scripts/generate_streaming_fixture.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import dspy
from dspy.streaming.streaming_listener import StreamListener
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

from pins import require

OUT = pathlib.Path(__file__).parent.parent / "crates" / "dsrust" / "tests" / "conformance" / "lm"
PINNED = require("dspy")

#: Each case is a field to listen for and the deltas a model produced.
CASES: dict[str, tuple[str, list[str]]] = {
    # Recorded from openai/gpt-4o-mini in upstream's own
    # `test_stream_listener_returns_correct_chunk_chat_adapter`. The marker arrives over five
    # deltas and closes inside a sixth, which is what makes the boundaries interesting.
    "recorded_gpt_4o_mini": (
        "answer",
        ["[[", " ##", " answer", " ##", " ]]\n\n", "To", " get", " to", " the", " other",
         " side", " of", " the", " dinner", " plate", "!\n\n[[ ##", " completed", " ##", " ]]"],
    ),
    "marker_split_across_deltas": (
        "answer",
        ["[[ ## ans", "wer ## ]]\nBer", "lin", "\n\n[[ ## completed ## ]]"],
    ),
    "a_preceding_field_is_discarded": (
        "answer",
        ["[[ ## reasoning ## ]]\nbecause the sky", " scatters blue", "\n\n[[ ## answer ## ]]\n",
         "Par", "is", "\n\n[[ ## completed ## ]]"],
    ),
    # A cache hit, or a model whose chunk is the whole reply — gemini's can be.
    "whole_reply_in_one_delta": ("answer", ["[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]"]),
    # The stream stops while the buffer still holds something that could have been the marker.
    "ends_holding_a_bracket": ("answer", ["[[ ## answer ## ]]\n", "Paris", "["]),
    # And the ordinary case: every token went out as it arrived, so nothing is left to finalize.
    "ends_holding_nothing": ("answer", ["[[ ## answer ## ]]\n", "Par", "is"]),
    # More than ten deltas that could each be the marker forming, which is where upstream's
    # ten-delta window starts releasing the oldest rather than holding everything.
    "more_held_than_the_window": (
        "answer",
        ["[[ ## answer ## ]]\n", *["[" for _ in range(13)], "\n\n[[ ## completed ## ]]"],
    ),
    # A bracket in the prose that never becomes a marker.
    "a_bracket_that_is_prose": (
        "answer",
        ["[[ ## answer ## ]]\n", "see [", "note", "] there", "\n\n[[ ## completed ## ]]"],
    ),
}


#: The same, over the JSON wire. `start_identifier` is `"field":` rather than a marker, and the end
#: is found by partial-parsing the accumulated object rather than by the next field's marker — so
#: these pin a different rule with the same shape.
JSON_CASES: dict[str, tuple[str, list[str]]] = {
    # Upstream's own recorded stream from `test_stream_listener_returns_correct_chunk_json_adapter`.
    "json_recorded_gpt_4o_mini": (
        "answer",
        ['{"', "answer", '":', '"To', " get", " to", " the", " other", " side", " of", " the",
         " frying", " pan", '!"', "}\n", "None", "None", "None"],
    ),
    "json_a_second_field_ends_it": (
        "answer",
        ['{"answer":', ' "Paris"', ', "judgement"', ': "fun', 'ny"', "}"],
    ),
    "json_the_field_is_not_first": (
        "judgement",
        ['{"answer": "Paris", ', '"judgement":', ' "fun', 'ny"', "}"],
    ),
    "json_a_brace_inside_the_value": (
        "answer",
        ['{"answer":', ' "a } brace"', "}"],
    ),
}


def chunks_of(field: str, deltas: list[str], adapter=None) -> list[dict]:
    """What `StreamListener` yields for one stream, `finalize` included."""
    dspy.configure(adapter=adapter or dspy.ChatAdapter())
    listener = StreamListener(signature_field_name=field)
    out = []
    for text in deltas:
        chunk = ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=text))]
        )
        answered = listener.receive(chunk)
        if answered is not None:
            out.append({"text": answered.chunk, "is_last": answered.is_last_chunk})
    tail = listener.finalize()
    if tail is not None:
        out.append({"text": tail.chunk, "is_last": tail.is_last_chunk})
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fixture = {
        "_source": f"dspy {PINNED} streaming/streaming_listener.py, via {pathlib.Path(__file__).name}",
        "cases": [
            {"name": name, "field": field, "deltas": deltas, "chunks": chunks_of(field, deltas)}
            for name, (field, deltas) in CASES.items()
        ],
        "json_cases": [
            {
                "name": name,
                "field": field,
                "deltas": deltas,
                "chunks": chunks_of(field, deltas, dspy.JSONAdapter()),
            }
            for name, (field, deltas) in JSON_CASES.items()
        ],
    }
    path = OUT / "streaming.json"
    path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
    print(f"  wrote {path.relative_to(OUT.parent.parent)}", file=sys.stderr)


if __name__ == "__main__":
    main()
