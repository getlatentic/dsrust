"""Run every fixture generator, and say which goldens moved.

There are fifty-five `generate_*.py` scripts and, until this, no command that ran them. Each one is
its own oracle — it imports the pinned library, runs it, and writes what came back — so a pin bump
means running all fifty-five and reading the diff. A pin bump done by remembering which generators
matter is a pin bump that misses one, and the golden it misses keeps asserting what the *old*
library did while every gate stays green.

Two modes, and the first is what makes the second trustworthy:

    .venv/bin/python scripts/regenerate_fixtures.py --check   # run all, expect no diff, restore
    .venv/bin/python scripts/regenerate_fixtures.py           # run all, keep what changed

`--check` is the control. Run it *before* touching a pin: every golden should regenerate byte for
byte, because it was generated from the library that is installed. A diff there is a golden that had
already drifted from its generator — hand-edited, or written under a different environment — and it
has to be settled before a bump, or its diff gets attributed to the new version.

A generator that fails is reported and does not stop the others: at a new pin, several failing with
`AttributeError` is *the finding* — it names what upstream moved.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).parent.parent
SCRIPTS = ROOT / "scripts"
PYTHON = ROOT / ".venv" / "bin" / "python"

#: Generators that take an argument or write outside the goldens, so a bare run is not what they are
#: for. `fuzz_parse.py` writes a campaign corpus rather than a golden — its committed slice comes
#: from `--sweep`, which is listed here with its arguments.
ARGUMENTS = {
    "fuzz_parse.py": ["1500", "0", "--sweep"],
}

#: Not fixture generators despite the name shape.
SKIP = {"generate_python_char_tables.py"}

#: `PYTHONHASHSEED=0` because several generators record the iteration order of a CPython `set` of
#: strings, and that order is salted per process unless the seed is fixed. The generators that need
#: it refuse to run without it rather than writing an order nobody can reproduce — which is the
#: right call, and means this runner has to set it or they never run at all.
ENVIRONMENT = {"PYTHONHASHSEED": "0"}


def generators() -> list[pathlib.Path]:
    found = sorted(SCRIPTS.glob("generate_*.py"))
    found += [SCRIPTS / name for name in ARGUMENTS if (SCRIPTS / name).exists()]
    return [script for script in found if script.name not in SKIP]


def tracked_diff() -> list[str]:
    """Every tracked file the working tree has changed, as `git status` sees it."""
    out = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [line[3:] for line in out.splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="expect no golden to change, and restore anything that did",
    )
    args = parser.parse_args()

    if not PYTHON.exists():
        print(f"no {PYTHON.relative_to(ROOT)} — run: uv sync", file=sys.stderr)
        return 1
    before = set(tracked_diff())

    failed: list[tuple[str, str]] = []
    for script in generators():
        argv = [str(PYTHON), str(script)] + ARGUMENTS.get(script.name, [])
        done = subprocess.run(argv, capture_output=True, text=True, env={**os.environ, **ENVIRONMENT})
        mark = "ok " if done.returncode == 0 else "FAIL"
        print(f"  {mark} {script.name}")
        if done.returncode != 0:
            tail = (done.stderr or done.stdout).strip().splitlines()
            failed.append((script.name, tail[-1] if tail else "no output"))

    moved = sorted(set(tracked_diff()) - before)
    print()
    if failed:
        print(f"{len(failed)} generator(s) failed:")
        for name, why in failed:
            print(f"    {name}: {why}")
        print()

    if moved:
        print(f"{len(moved)} golden(s) changed:")
        for path in moved:
            print(f"    {path}")
    else:
        print("no golden changed")

    if args.check:
        if moved:
            subprocess.run(["git", "-C", str(ROOT), "checkout", "--", *moved], check=False)
            print(
                "\nrestored. A golden that changes when regenerated against the library it was\n"
                "generated from has drifted from its generator — settle that before a pin bump, or\n"
                "the diff gets blamed on the new version."
            )
        return 1 if (moved or failed) else 0
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
