"""dspy `pretty_print_history`, recorded by printing.

`dspy.inspect_history()` renders the last few calls for reading in a terminal — the prompts, the
replies, and the tool calls either side, with ANSI colour when it is writing to a terminal and none
when it is writing to a file. All of it is bytes, and the branches are easy to miss: an image block
prints its base64 length rather than its data, an audio block its format and length, a file block
three of its fields, and a reply with more than one completion ends with a count and no newline.

Recorded through the `file=` argument, which is upstream's own no-colour path, and again through a
capture of stdout for the coloured one — so both sides of `use_colors` come from the same function.

    .venv/bin/python scripts/generate_inspect_history_fixture.py
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import pathlib
import warnings

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

from dspy.utils.inspect_history import pretty_print_history

from pins import require

PINNED = require("dspy")
OUT = (
    pathlib.Path(__file__).parent.parent
    / "crates"
    / "dsrust"
    / "tests"
    / "conformance"
    / "history"
    / "inspect_history.json"
)

PIXEL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="


def entry(messages: list[dict], outputs: list, timestamp: str | None = "2026-01-01T00:00:00Z") -> dict:
    row = {"messages": messages, "outputs": outputs, "prompt": None}
    if timestamp is not None:
        row["timestamp"] = timestamp
    return row


CASES: list[tuple[str, list[dict]]] = [
    (
        "one_plain_exchange",
        [entry([{"role": "user", "content": "  What is 2+2?  "}], ["  4  "])],
    ),
    (
        "a_system_turn_is_capitalised",
        [entry([{"role": "system", "content": "Be terse."}, {"role": "user", "content": "Hi"}], ["Hello"])],
    ),
    (
        "several_completions_end_with_a_count",
        [entry([{"role": "user", "content": "pick"}], ["one", "two", "three"])],
    ),
    (
        "a_dict_output_carries_text_and_tool_calls",
        [
            entry(
                [{"role": "user", "content": "call it"}],
                [
                    {
                        "text": "  calling  ",
                        "tool_calls": [
                            {"function": {"name": "search", "arguments": '{"q": "rust"}'}},
                            {"name": "lookup", "args": {"id": 7}},
                            {"function": {}, "arguments": "not json at all"},
                        ],
                    }
                ],
            )
        ],
    ),
    (
        "a_dict_output_with_no_text_prints_no_response_line",
        [entry([{"role": "user", "content": "quiet"}], [{"tool_calls": [{"name": "noop"}]}])],
    ),
    (
        # `if outputs[0].get("text"):` is truthiness, so an empty string is skipped exactly as an
        # absent key is — which a case carrying no key at all cannot tell apart.
        "an_empty_text_is_skipped_like_an_absent_one",
        [entry([{"role": "user", "content": "quiet"}], [{"text": "", "tool_calls": [{"name": "noop"}]}])],
    ),
    (
        "content_blocks_of_every_kind",
        [
            entry(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "  look  "},
                            {"type": "image_url", "image_url": {"url": PIXEL}},
                            {"type": "image_url", "image_url": {"url": "https://example.invalid/a.png"}},
                            {"type": "input_audio", "input_audio": {"format": "wav", "data": "AAAA"}},
                            {"type": "file", "file": {"filename": "a.pdf", "file_id": "f1", "file_data": "abcd"}},
                            {"type": "input_file", "input_file": {"filename": "b.pdf"}},
                        ],
                    }
                ],
                ["seen"],
            )
        ],
    ),
    (
        "a_message_carrying_tool_calls",
        [
            entry(
                [{"role": "assistant", "content": "thinking", "tool_calls": [{"function": {"name": "f", "arguments": '{"a": 1}'}}]}],
                ["done"],
            )
        ],
    ),
    (
        "no_timestamp_reads_as_unknown",
        [entry([{"role": "user", "content": "when?"}], ["now"], timestamp=None)],
    ),
    (
        "the_prompt_stands_in_for_absent_messages",
        [{"messages": None, "prompt": "bare prompt", "outputs": ["ok"], "timestamp": "t"}],
    ),
    (
        "only_the_last_n_are_printed",
        [
            entry([{"role": "user", "content": f"q{i}"}], [f"a{i}"])
            for i in range(4)
        ],
    ),
]


def rendered(history: list[dict], n: int, colours: bool) -> str:
    if colours:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            pretty_print_history(history, n)
        return buffer.getvalue()
    buffer = io.StringIO()
    pretty_print_history(history, n, file=buffer)
    return buffer.getvalue()


def main() -> None:
    cases = []
    for name, history in CASES:
        n = 2 if name == "only_the_last_n_are_printed" else 1
        cases.append(
            {
                "name": name,
                "n": n,
                "history": history,
                "plain": rendered(history, n, colours=False),
                "coloured": rendered(history, n, colours=True),
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "source": f"generated from dspy=={PINNED} via scripts/generate_inspect_history_fixture.py",
                "note": (
                    "`plain` is what upstream writes through its `file=` argument, which disables "
                    "colour; `coloured` is the same call to stdout."
                ),
                "cases": cases,
            },
            indent=2,
        )
        + "\n"
    )
    for case in cases:
        print(f"  {case['name']:44s} {len(case['plain']):5d} plain  {len(case['coloured']):5d} coloured")
    print(f"wrote {OUT.relative_to(pathlib.Path(__file__).parent.parent)}")


if __name__ == "__main__":
    main()
