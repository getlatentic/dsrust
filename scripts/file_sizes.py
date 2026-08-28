#!/usr/bin/env python3
"""Report each source file's non-test size, against the ~400-line guideline.

Counting to the first `#[cfg(test)]` undercounts: a file may carry ordinary code *after* its
test module, and twice now a file has looked comfortably inside the guideline while being well
outside it. This walks the whole file and skips each test module wherever it sits.

Finding that module's end means counting braces, and counting them in raw text is how the same
undercount came back on 2026-08-28. `lm/openai/responses/mod.rs` reported 393 lines against a real
448: its tests hold JSON, the braces inside those string literals never balanced, and the module
was therefore reported to end at the end of the file — swallowing every line of ordinary code that
followed it. So the scan below skips strings, chars and comments, which is the difference between
counting Rust and counting text that looks like it.
"""

import argparse
import pathlib
import sys

GUIDELINE = 400


def non_test_lines(text: str) -> int:
    """Lines outside every `#[cfg(test)]` module, blank lines not counted."""
    lines = text.splitlines()
    total = index = 0
    while index < len(lines):
        if lines[index].strip().startswith("#[cfg(test)]"):
            index = end_of_module(lines, index + 1)
            continue
        if lines[index].strip():
            total += 1
        index += 1
    return total


def end_of_module(lines: list[str], index: int) -> int:
    """The line after the module opening at or below `index` closes."""
    depth = 0
    opened = False
    in_block_comment = False
    while index < len(lines):
        opens, closes, in_block_comment = braces(lines[index], in_block_comment)
        depth += opens - closes
        opened = opened or opens > 0
        index += 1
        if opened and depth <= 0:
            break
    return index


def braces(line: str, in_block_comment: bool) -> tuple[int, int, bool]:
    """The braces this line opens and closes, ignoring those inside strings and comments.

    Rust's raw strings (`r"…"`, `r#"…"#`) are handled by their hash count, which is what lets a
    `json!` literal in a test hold an unmatched brace without moving the depth.
    """
    opens = closes = 0
    index = 0
    while index < len(line):
        rest = line[index:]
        if in_block_comment:
            end = rest.find("*/")
            if end < 0:
                return opens, closes, True
            index += end + 2
            in_block_comment = False
        elif rest.startswith("//"):
            break
        elif rest.startswith("/*"):
            in_block_comment = True
            index += 2
        elif rest[0] in "\"'" or (rest[0] == "r" and rest[1:2] in ('"', "#")):
            index += literal_length(line, index)
        else:
            opens += rest[0] == "{"
            closes += rest[0] == "}"
            index += 1
    return opens, closes, in_block_comment


def literal_length(line: str, index: int) -> int:
    """How far the literal starting at `index` runs, or 1 if it does not close on this line."""
    hashes = 0
    at = index
    if line[at] == "r":
        at += 1
        while at < len(line) and line[at] == "#":
            hashes += 1
            at += 1
        if at >= len(line) or line[at] != '"':
            return 1
    quote = line[at]
    at += 1
    closing = quote + "#" * hashes
    while at < len(line):
        if line[at] == "\\" and not hashes and quote == '"':
            at += 2
            continue
        if line.startswith(closing, at):
            return at + len(closing) - index
        at += 1
    # An unterminated literal on this line: step past the opener rather than reading its content
    # as code, which is the direction that cannot silently extend a module.
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default="crates/dsrust/src", type=pathlib.Path)
    parser.add_argument(
        "--limit",
        type=int,
        default=GUIDELINE,
        help=f"lines a file may carry before it is reported (default {GUIDELINE})",
    )
    args = parser.parse_args()

    sizes = sorted(
        ((non_test_lines(path.read_text()), path) for path in args.root.rglob("*.rs")),
        reverse=True,
    )
    over = [(count, path) for count, path in sizes if count > args.limit]
    for count, path in sizes:
        mark = "  <-- over" if count > args.limit else ""
        print(f"{count:>5}  {path}{mark}")
    print(f"\n{len(over)} of {len(sizes)} files over {args.limit} non-test lines")
    return 1 if over else 0


if __name__ == "__main__":
    sys.exit(main())
