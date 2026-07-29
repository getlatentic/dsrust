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
import ast
import pathlib
import re
import sys

import status

CONFTEST = pathlib.Path(__file__).parent.parent / "bridge" / "python" / "conftest.py"

#: Each status key, and the pattern that finds its value in a pytest run's output.
OBSERVED = {
    "upstream_tests_passing": re.compile(r"^(\d+) passed", re.M),
    "upstream_tests_crossing": re.compile(r"^-+ (\d+) of \d+ tests rendered or parsed", re.M),
    "upstream_tests_deciding_signatures": re.compile(
        r"^-+ (\d+) of \d+ tests decided a signature", re.M
    ),
    "upstream_tests_xfailed": re.compile(r"(\d+) xfailed"),
}


def observed(output: str) -> dict[str, int]:
    found = {}
    for key, pattern in OBSERVED.items():
        match = pattern.search(output)
        if match is not None:
            found[key] = int(match.group(1))
    return found


def strict_xfails() -> int:
    """How many gaps this port has declared, counted from the dict rather than from pytest.

    `xfail_backlog` used to record pytest's `N xfailed`, which is not the same thing and was
    larger: dspy marks two of its own image cases xfail inside the test body, for a gap upstream
    has rather than one this port has. Reading the declaration itself is what makes the number
    mean what its name says.
    """
    for node in ast.walk(ast.parse(CONFTEST.read_text())):
        named = isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "NOT_YET_IMPLEMENTED"
            for target in node.targets
        )
        if named:
            return len(node.value.keys)
    raise SystemExit(f"{CONFTEST.name} has no NOT_YET_IMPLEMENTED to count")


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
    values["xfail_backlog"] = strict_xfails()

    changed = status.record(values)
    print(f"  status: {'; '.join(changed) if changed else 'unchanged'}", file=sys.stderr)


if __name__ == "__main__":
    main()
