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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from api_surface import PORTED_MODULES  # noqa: E402


def pinned_tag() -> str:
    return (ROOT / "scripts" / "DSPY_VERSION").read_text().strip()


def exported(module: str, tag: str) -> list[str]:
    """A module's own `__all__` at the pinned tag, or [] where it declares none."""
    shown = subprocess.run(
        ["git", "-C", str(DSPY), "show", f"{tag}:dspy/{module}"],
        capture_output=True,
        text=True,
    )
    if shown.returncode != 0:
        return []
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
    ledger = tomllib.loads(LEDGER.read_text())
    known = answered(ledger)

    print(f"==> Every `__all__` name in the pinned tree ({tag})")
    missing: list[tuple[str, str]] = []
    declaring = 0
    for module in PORTED_MODULES:
        names = exported(module, tag)
        if not names:
            continue
        declaring += 1
        for name in names:
            if name.startswith("_") or name in known:
                continue
            missing.append((module, name))

    print(f"    {declaring} of {len(PORTED_MODULES)} ported modules declare `__all__`")
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
