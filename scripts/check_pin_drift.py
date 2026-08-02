#!/usr/bin/env python3
"""Where the pinned dspy and main disagree about a ported module's public surface.

Not a gate on the crate — a gate on *what to read before porting*. Every other check here compares
this crate to the pin. This one compares the pin to main, and answers the question nothing else does:
**is the thing I am about to build a moving target?**

It exists because `ChatModel::call` was built against `BaseLM.__call__`'s signature on main, where it
takes variadic items, while the pinned tag's takes none. Every gate passed. The suite runs at the
pin, so no upstream test could reach it; the goldens are generated from the pin, so none exists for
it; and the ledger had no entry to be wrong. Nothing in the repository could have said so, and this
is what says so now.

Two things it is deliberately not. It does not fail a build — main moves, and a red gate that tracks
someone else's branch is a gate people learn to ignore. And it does not say to port anything: where
main is ahead, the answer is to move the pin when a release lands, not to build without an oracle.
See HANDOFF's "Check main, not only the pin".

    ./scripts/check_pin_drift.py            # the summary
    ./scripts/check_pin_drift.py --detail   # every name, per module
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DSPY = ROOT / "third_party" / "dspy"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from api_surface import PORTED_MODULES, surface_of_source  # noqa: E402


def at(ref: str, module: str) -> str | None:
    shown = subprocess.run(
        ["git", "-C", str(DSPY), "show", f"{ref}:dspy/{module}"],
        capture_output=True,
        text=True,
    )
    return shown.stdout if shown.returncode == 0 else None


def names(source: str) -> set[str]:
    """Every public name a module offers, each carrying its parameter list.

    The parameters are the point. Comparing names alone would have said `clients/base_lm.py` was
    unchanged, because `__call__` exists at the pin and on main — while its signature went from
    `(prompt, messages, **kwargs)` to `(*items, prompt, messages, request, **kwargs)`, which is the
    whole of what was missed. A name whose arguments moved is a moving target as surely as one that
    appeared.
    """
    surface = surface_of_source(source)
    found = {f"{fn}()" for fn in surface["functions"]}
    tree = ast.parse(source)
    for stmt in ast.walk(tree):
        if not isinstance(stmt, ast.ClassDef) or stmt.name.startswith("_"):
            continue
        found.add(stmt.name)
        for member in stmt.body:
            if not isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if member.name.startswith("_") and not member.name.startswith("__"):
                continue
            found.add(f"{stmt.name}.{member.name}{signature(member)}")
    return found


def signature(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """A method's parameter names, in order, with the two variadic markers kept."""
    spec = fn.args
    parts = [arg.arg for arg in [*spec.posonlyargs, *spec.args] if arg.arg != "self"]
    if spec.vararg:
        parts.append(f"*{spec.vararg.arg}")
    elif spec.kwonlyargs:
        parts.append("*")
    parts.extend(arg.arg for arg in spec.kwonlyargs)
    if spec.kwarg:
        parts.append(f"**{spec.kwarg.arg}")
    return "(" + ", ".join(parts) + ")"


class Unreadable(Exception):
    """One side of the comparison was not there, which is not the same as the two agreeing."""


def drift(module: str, pin: str) -> tuple[set[str], set[str]] | None:
    """What main adds and what it drops, or None where the module is unchanged.

    Unreadable raises rather than returning None. Folding the two together is what let this report
    `every ported module's public surface is identical on main` in a worktree with an empty
    submodule — the strongest sentence it can print, from a comparison it never made.
    """
    pinned, current = at(pin, module), at("origin/main", module)
    if pinned is None or current is None:
        raise Unreadable(module)
    if pinned == current:
        return None
    was, now = names(pinned), names(current)
    added, gone = now - was, was - now
    return (added, gone) if added or gone else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detail", action="store_true", help="every name, not just the count")
    arguments = parser.parse_args()

    pin = (ROOT / "scripts" / "DSPY_VERSION").read_text().strip()
    print(f"==> Public surface: pinned {pin} against origin/main")

    moved: dict[str, tuple[set[str], set[str]]] = {}
    unreadable: list[str] = []
    for module in PORTED_MODULES:
        try:
            if found := drift(module, pin):
                moved[module] = found
        except Unreadable:
            unreadable.append(module)

    if unreadable:
        print(f"    {len(unreadable)} of {len(PORTED_MODULES)} modules could not be read at all:")
        for module in unreadable[:5]:
            print(f"      ? {module}")
        print(
            f"\n    {DSPY.relative_to(ROOT)} is missing, or has no origin/main. This report says\n"
            "    nothing about drift until it can read both sides:\n"
            "      git submodule update --init third_party/dspy"
        )
        return 1

    if not moved:
        print("    every ported module's public surface is identical on main")
        return 0

    print(f"    {len(moved)} of {len(PORTED_MODULES)} ported modules have moved:\n")
    for module, (added, gone) in sorted(moved.items()):
        print(f"  {module}  +{len(added)} -{len(gone)}")
        if arguments.detail:
            for name in sorted(added):
                print(f"      + {name}")
            for name in sorted(gone):
                print(f"      - {name}")
    print(
        "\n  A name here is a moving target. Read the *pin* when porting it, and if what you want\n"
        "  only exists on main, the answer is to move the pin when a release lands — not to build\n"
        "  something no golden and no upstream test can reach."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
