"""Write what the last upstream run observed into `backlog.toml`'s `[status]` block.

Those numbers were prose: hand-written, never checked, and stale — 452 passing where the suite
reported 645. A plan that misreports its own evidence is worse than one that reports none, because
it reads as measured.

So they are generated now. The runner pipes pytest's output here after a green run; anything this
cannot read off that output is left alone rather than guessed at, and a red run writes nothing.
Drift then shows up as an uncommitted diff instead of quietly persisting.

    ... | python3 scripts/record_status.py --suites 33
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).parent.parent
BACKLOG = ROOT / "backlog.toml"

#: Each status key, and the pattern that finds its value in a pytest run's output.
OBSERVED = {
    "upstream_tests_passing": re.compile(r"^(\d+) passed", re.M),
    "upstream_tests_crossing": re.compile(r"^-+ (\d+) of \d+ tests rendered or parsed", re.M),
    "upstream_tests_deciding_signatures": re.compile(
        r"^-+ (\d+) of \d+ tests decided a signature", re.M
    ),
    "xfail_backlog": re.compile(r"(\d+) xfailed"),
}


def observed(output: str) -> dict[str, int]:
    found = {}
    for key, pattern in OBSERVED.items():
        match = pattern.search(output)
        if match is not None:
            found[key] = int(match.group(1))
    return found


def rewrite(text: str, values: dict[str, int]) -> tuple[str, list[str]]:
    """The backlog with `[status]` updated, and which keys actually moved."""
    changed = []
    for key, value in values.items():
        pattern = re.compile(rf"^({re.escape(key)} = )(\d+)$", re.M)
        match = pattern.search(text)
        if match is None:
            continue
        if int(match.group(2)) != value:
            changed.append(f"{key}: {match.group(2)} -> {value}")
        text = pattern.sub(rf"\g<1>{value}", text, count=1)
    return text, changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suites", type=int, required=True, help="how many files the runner ran")
    args = parser.parse_args()

    output = sys.stdin.read()
    # A red run says nothing trustworthy about coverage, so it leaves the block alone.
    if re.search(r"^\d+ failed", output, re.M) or " error" in output.lower().split("=====")[-1]:
        print("  status: not recorded (the run was not green)", file=sys.stderr)
        return

    values = observed(output)
    if not values:
        print("  status: nothing to record (no pytest summary found)", file=sys.stderr)
        return
    values["suites_run"] = args.suites

    text, changed = rewrite(BACKLOG.read_text(), values)
    BACKLOG.write_text(text)
    if changed:
        print(f"  status: {'; '.join(changed)}", file=sys.stderr)
    else:
        print("  status: unchanged", file=sys.stderr)


if __name__ == "__main__":
    main()
