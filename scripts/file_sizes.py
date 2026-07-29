#!/usr/bin/env python3
"""Report each source file's non-test size, against the ~400-line guideline.

Counting to the first `#[cfg(test)]` undercounts: a file may carry ordinary code *after* its
test module, and twice now a file has looked comfortably inside the guideline while being well
outside it. This walks the whole file and skips each test module wherever it sits.
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
    while index < len(lines):
        depth += lines[index].count("{") - lines[index].count("}")
        opened = opened or "{" in lines[index]
        index += 1
        if opened and depth <= 0:
            break
    return index


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
