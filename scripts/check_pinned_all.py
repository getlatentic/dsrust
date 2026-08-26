#!/usr/bin/env python3
"""Every name a ported module *exports* is a name the ledger has an answer for.

`api_surface.py` reads a module's classes and functions by walking its AST. That misses two things,
and both have hidden a real gap:

* **A name a module re-exports rather than defines.** `dspy.core.types` lists `User` and `Assistant`
  in `__all__` and defines them as functions, so the walk sees them — but a module that imports a
  name and re-exports it has no definition to walk, and the name silently leaves the surface.
* **A name the ledger answers for that the *pin* does not have.** `ChatModel::call` was written
  against `BaseLM.__call__`'s signature on main, where it takes variadic items; at the pinned tag it
  takes none. Nothing caught that, because nothing compared the ledger against the pin's own idea of
  what it exports.

So this reads `__all__` from the pinned tree — the module's own statement of its public surface —
and asks the ledger about each name. It is the audit's cheapest method and the one its plan says to
run before reading anything, because it needs no judgement.

    ./scripts/check_pinned_all.py
"""

from __future__ import annotations

import ast
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DSPY = ROOT / "third_party" / "dspy"
LEDGER = ROOT / "scripts" / "api_ledger.toml"
#: How many ported modules declare `__all__` at the pinned tag. Measured, not chosen.
DECLARING = 3

sys.path.insert(0, str(Path(__file__).resolve().parent))
from api_surface import PORTED_MODULES  # noqa: E402


def pinned_tag() -> str:
    return (ROOT / "scripts" / "DSPY_VERSION").read_text().strip()


def pinned_tree(tag: str) -> str:
    """The commit to read the pinned source from: the gitlink the superproject records.

    Not the tag. `actions/checkout` fetches a submodule's *commit* and not its tags, so resolving
    `3.3.0:dspy/...` fails on CI while the correct tree is sitting right there checked out — which
    read as "the submodule is missing" and failed the gate on a tree that was perfectly fine. The
    gitlink is also the better authority: it is what review saw and what `git submodule update`
    restores, where a tag is a name that has to be fetched separately to mean anything.

    The tag still verifies the *version* wherever it is available, which on a development machine
    is always. Where it is not, the commit is checked and the name is taken on trust — said out
    loud rather than assumed, because a gate that quietly checks a different tree is the failure
    this whole file exists to have stopped once already.
    """
    recorded = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", f"HEAD:third_party/dspy"],
        capture_output=True,
        text=True,
    )
    if recorded.returncode != 0:
        return tag
    commit = recorded.stdout.strip()
    tagged = subprocess.run(
        ["git", "-C", str(DSPY), "rev-parse", f"{tag}^{{commit}}"],
        capture_output=True,
        text=True,
    )
    if tagged.returncode != 0:
        print(f"  {tag} is not a tag in the submodule here; reading the recorded commit {commit[:8]}")
    elif tagged.stdout.strip() != commit:
        raise SystemExit(
            f"the submodule gitlink {commit[:8]} is not {tag} ({tagged.stdout.strip()[:8]}) — "
            "the pin and scripts/DSPY_VERSION disagree"
        )
    return commit


def exported(module: str, tree: str) -> list[str] | None:
    """A module's own `__all__` at the pinned tag, [] where it declares none, None where unreadable.

    The three-way answer is the point. Folding "declares no `__all__`" together with "I could not
    read this file" is what let this gate report `0 of 60 ported modules declare __all__` and then
    OK, in a worktree where the submodule is an empty directory — a pass that read nothing.
    """
    shown = subprocess.run(
        ["git", "-C", str(DSPY), "show", f"{tree}:dspy/{module}"],
        capture_output=True,
        text=True,
    )
    if shown.returncode != 0:
        return None
    tree = ast.parse(shown.stdout)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(t.id == "__all__" for t in node.targets if isinstance(t, ast.Name)):
            continue
        return [
            item.value
            for item in getattr(node.value, "elts", [])
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        ]
    return []


def answered(ledger: dict) -> set[str]:
    """Every dspy name the ledger says anything about, as a bare identifier."""
    names = set()
    for table in ("symbols", "methods", "constructors"):
        for key in ledger.get(table, {}):
            names.add(key.split("::")[-1].split(".")[-1])
    return names


def main() -> int:
    tag = pinned_tag()
    tree = pinned_tree(tag)
    ledger = tomllib.loads(LEDGER.read_text())
    known = answered(ledger)

    print(f"==> Every `__all__` name in the pinned tree ({tag})")
    missing: list[tuple[str, str]] = []
    unreadable: list[str] = []
    declaring = 0
    for module in PORTED_MODULES:
        names = exported(module, tree)
        if names is None:
            unreadable.append(module)
            continue
        if not names:
            continue
        declaring += 1
        for name in names:
            if name.startswith("_") or name in known:
                continue
            missing.append((module, name))

    if unreadable:
        print(f"\n`__all__` gate FAILED:\n  {len(unreadable)} ported module(s) not in the pin:")
        for module in unreadable[:5]:
            print(f"      ? {module}")
        print(
            f"\n  The submodule at {DSPY.relative_to(ROOT)} is missing or does not have {tag}.\n"
            "  Run: git submodule update --init third_party/dspy"
        )
        return 1

    print(f"    {declaring} of {len(PORTED_MODULES)} ported modules declare `__all__`")
    # Three do, and the gate is worth nothing against zero of them. A floor at the measured number
    # rather than at one: losing two of the three would otherwise still read as a pass.
    if declaring < DECLARING:
        print(
            f"\n`__all__` gate FAILED:\n  only {declaring} module(s) declare `__all__`, "
            f"against {DECLARING} when this floor was measured.\n"
            "  Either the pin moved and this number moves with it, or the walk stopped seeing them."
        )
        return 1
    if not missing:
        print("\n`__all__` gate: OK (every exported name has a ledger answer)")
        return 0

    print(f"\n`__all__` gate FAILED:\n  {len(missing)} exported name(s) the ledger never mentions:")
    for module, name in missing:
        print(f"      ? {module}::{name}")
    print(
        "\n  Each is a name the pinned dspy tells its users about. Map it, or write the reason it\n"
        "  diverges — the point is that no name leaves the surface without one."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
