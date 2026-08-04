"""Compare what goes in and what comes out as **bytes**, not as text.

Every other fixture holds its inputs and outputs as JSON strings, which means the comparison passes
through two JSON encoders and a Rust `&str`. That is fine right up until something along the way
normalises — a lone combining mark composed, a BOM eaten, `\\r\\n` folded to `\\n` — and then the
strings match while the bytes do not.

So this records both sides in hex. Nothing between the interpreter and the assertion can touch a
hex digit.

The corpus is chosen for the places an encoder is tempted to interfere: a byte-order mark, CRLF, a
NUL, an astral character that is one code point and two UTF-16 units, a combining sequence that has
a composed form, and both settings of `ensure_ascii`.

    .venv/bin/python scripts/generate_json_repair_bytes_fixture.py
"""

from __future__ import annotations

import json
import pathlib
import sys
import unicodedata

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import json_repair  # noqa: E402
from pins import require  # noqa: E402

OUT = (
    pathlib.Path(__file__).parent.parent
    / "crates"
    / "dsrust-json-repair"
    / "tests"
    / "conformance"
    / "json_repair_bytes.json"
)

#: `(name, why, text, ensure_ascii)`.
CASES = [
    ("bom", "a byte-order mark, which is a character and not a marker", "﻿{a: 1}", True),
    ("crlf", "CRLF line endings, which nothing may fold", '{"a": "x\r\ny"}', True),
    ("nul", "a NUL inside a string", '{"a": "x\x00y"}', True),
    ("astral", "one code point, two UTF-16 units, one surrogate pair on the way out",
     '{"a": "🙂"}', True),
    ("astral_raw", "and the same with ensure_ascii off, where it stays four UTF-8 bytes",
     '{"a": "🙂"}', False),
    ("combining", "a combining sequence, which has a composed form nothing may normalise to",
     '{"a": "é"}', False),
    ("composed", "and the composed form, which must stay distinct from it", '{"a": "é"}', False),
    ("cjk", "a reply in Chinese, escaped", '{答案: "北京"}', True),
    ("cjk_raw", "and unescaped", '{答案: "北京"}', False),
    ("smart_quotes", "the curly pair, which the parser reads and the writer must give back",
     "{“a”: “b”}", False),
    ("low_smart_quote", "the low one, which opens a span no other rule closes",
     '{"a": "say „hi" there”"}', False),
    ("control_run", "every control character JSON has a short name for", '{"a": "\b\f\n\r\t"}', False),
    ("control_no_short_name", "and one that has none, which stays escaped even with ensure_ascii "
     "off, since JSON requires it either way", '{"a": "x\x01y"}', False),
    ("del", "DEL, which is escaped only when ensure_ascii is on", '{"a": "\x7f"}', True),
    ("del_raw", "and left alone when it is not", '{"a": "\x7f"}', False),
    ("latin1_high", "a high Latin-1 character, two UTF-8 bytes", '{"a": "ÿ"}', False),
    # The byte scanner and the span-copying writer pivot on encoding-length boundaries, so each
    # boundary code point crosses both ways: the last one-byte character, the first and last
    # two-byte, three-byte, and four-byte ones. A slice landing one byte off any of these panics,
    # which is the denial-of-service a byte-indexed scanner risks and a char-indexed one does not.
    ("boundary_two_byte_first", "U+0080, the first two-byte character", '{"a": "\u0080"}', True),
    ("boundary_two_byte_first_raw", "and raw", '{"a": "\u0080"}', False),
    ("boundary_two_byte_last", "U+07FF, the last", '{"a": "\u07ff"}', False),
    ("boundary_three_byte_first", "U+0800, the first three-byte", '{"a": "\u0800"}', False),
    ("boundary_bmp_last", "U+FFFF, the top of the basic plane", '{"a": "\uffff"}', True),
    ("boundary_bmp_last_raw", "and raw", '{"a": "\uffff"}', False),
    ("boundary_astral_first", "U+10000, the smallest surrogate pair", '{"a": "\U00010000"}', True),
    ("boundary_astral_last", "U+10FFFF, the largest code point there is",
     '{"a": "\U0010ffff"}', True),
    ("boundary_astral_last_raw", "which is four UTF-8 bytes raw", '{"a": "\U0010ffff"}', False),
    ("escape_beside_multibyte", "an escape flanked by multi-byte characters, which is where a "
     "span-copying writer's bookkeeping is off by one if it ever is", '{"a": "é\né北\t🙂"}', False),
    ("escape_beside_multibyte_ascii", "the same, escaped to ASCII", '{"a": "é\né北\t🙂"}', True),
    ("multibyte_key_ascii", "a key past ASCII under ensure_ascii, since keys escape too",
     '{"北京": 1}', True),
]


def main() -> None:
    version = require("json_repair")
    recorded = []
    for name, why, text, ascii_only in CASES:
        output = json_repair.repair_json(text, ensure_ascii=ascii_only)
        recorded.append(
            {
                "name": name,
                "why": why,
                "ensure_ascii": ascii_only,
                "input_hex": text.encode().hex(),
                "output_hex": output.encode().hex(),
            }
        )

    check_the_corpus_discriminates(recorded)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "source": f"json_repair {version}, CPython {sys.version.split()[0]}, "
                f"Unicode {unicodedata.unidata_version}",
                "what": (
                    "Input and output as hex, so the comparison never passes through a text "
                    "encoder that could normalise either one."
                ),
                "cases": recorded,
            },
            indent=1,
        )
        + "\n"
    )
    print(f"  wrote {OUT} — {len(recorded)} cases", file=sys.stderr)


def check_the_corpus_discriminates(cases: list[dict]) -> None:
    """Refuse a corpus that hex would have been pointless for."""
    if all(case["input_hex"] == case["output_hex"] for case in cases):
        raise SystemExit("every case comes back unchanged — this tests nothing")

    non_ascii = [case for case in cases if any(int(case["input_hex"][i : i + 2], 16) > 0x7F
                                               for i in range(0, len(case["input_hex"]), 2))]
    if len(non_ascii) < len(cases) // 2:
        raise SystemExit(f"only {len(non_ascii)} cases carry a byte above ASCII")

    if not any(case["ensure_ascii"] for case in cases) or all(case["ensure_ascii"] for case in cases):
        raise SystemExit("the corpus exercises one setting of ensure_ascii")

    # A pair whose *inputs* differ only by normalisation, so a normalising step is visible.
    combining = {case["name"]: case["output_hex"] for case in cases if case["name"] in ("combining", "composed")}
    if len(combining) != 2 or len(set(combining.values())) != 2:
        raise SystemExit("the combining and composed cases must stay distinct, or nothing detects folding")


if __name__ == "__main__":
    main()
