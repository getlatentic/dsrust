#!/usr/bin/env python3
"""Every name a user reaches as `dspy.X` is a name the ledger has an answer for.

`check_pinned_all.py` asks this of a *ported module*'s `__all__`. That is the right question for a
module already in scope and the wrong one for the surface as a whole: 40 of dspy's 134 modules are
not on `PORTED_MODULES`, so nothing reads them and nothing can report what they export.
A name is answered from either of the two places this repo decides things: the ledger, for a symbol
in a ported module, and `unported_modules.toml`, for one whose whole module was decided against.
Both are needed. `dspy.ColBERTv2` has no ledger row and never will — its module is `out_of_scope`,
and moving it into `PORTED_MODULES` to earn one is what `check_unported_modules.py` exists to
refuse, the two lists being exhaustive and disjoint on purpose.

The entry point is where a user's parity question lands. `dspy/__init__.py` names some imports
outright and star-imports six packages, each of which declares `__all__`, so the reachable surface
resolves from the pinned tree without importing dspy.

    ./scripts/check_top_level.py
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
UNPORTED = ROOT / "scripts" / "unported_modules.toml"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_pinned_all import answered, exported, pinned_tag, pinned_tree  # noqa: E402


def decided_elsewhere(tree: str) -> set[str]:
    """Names defined by a module `unported_modules.toml` has already ruled on.

    A module in that table is out of `PORTED_MODULES` by construction, so nothing it defines can
    hold a ledger row. Its decision is no weaker for that — `predict/avatar/avatar.py` is
    `upstream_dead` with a golden naming the `AttributeError`, which is a firmer answer about
    `dspy.AvatarOptimizer` than a `todo` row would be.
    """
    modules = tomllib.loads(UNPORTED.read_text()).get("modules", {})
    names: set[str] = set()
    for module in modules:
        shown = subprocess.run(
            ["git", "-C", str(DSPY), "show", f"{tree}:dspy/{module}"],
            capture_output=True,
            text=True,
        )
        if shown.returncode != 0:
            continue
        for stmt in ast.parse(shown.stdout).body:
            if isinstance(stmt, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(stmt.name)
            elif isinstance(stmt, ast.Assign):
                names.update(
                    t.id for t in stmt.targets if isinstance(t, ast.Name) and not t.id.startswith("_")
                )
    return names


def constructed_from(module: ast.Module, tree: str) -> dict[str, str]:
    """For each imported name, the class it is an instance of, where it is one.

    `dspy.settings` is `settings = Settings()` in `dsp/utils/settings.py`, and `Settings` carries a
    long divergence row. The instance is not a second thing to classify — but it is what a user
    touches, so the gate has to reach the class rather than shrug at the name.
    """
    sources = {
        alias.asname or alias.name: (node.module or "").removeprefix("dspy.").replace(".", "/")
        for node in ast.walk(module)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
        if alias.name != "*"
    }
    built: dict[str, str] = {}
    for name, where in sources.items():
        shown = subprocess.run(
            ["git", "-C", str(DSPY), "show", f"{tree}:dspy/{where}.py"],
            capture_output=True,
            text=True,
        )
        if shown.returncode != 0:
            continue
        for stmt in ast.parse(shown.stdout).body:
            if not isinstance(stmt, ast.Assign) or not isinstance(stmt.value, ast.Call):
                continue
            if not isinstance(stmt.value.func, ast.Name):
                continue
            if any(isinstance(t, ast.Name) and t.id == name for t in stmt.targets):
                built[name] = stmt.value.func.id
    return built


def entry_point(tree: str) -> ast.Module | None:
    """`dspy/__init__.py` at the pinned tree, or None where the submodule cannot be read."""
    shown = subprocess.run(
        ["git", "-C", str(DSPY), "show", f"{tree}:dspy/__init__.py"],
        capture_output=True,
        text=True,
    )
    return ast.parse(shown.stdout) if shown.returncode == 0 else None


def aliased(node: ast.Assign) -> str | None:
    """The name a `dspy.X = Y` binding stands for, where it stands for one.

    `BootstrapRS = BootstrapFewShotWithRandomSearch` is a second name for a ported optimizer, not a
    second thing to port, and asking the ledger about the alias reports a gap that is not there.

    For `configure = settings.configure` the link is the *object*, not the attribute. Resolving to
    the attribute answered `dspy.context` with `Document.context` — a different `context`
    entirely — because the ledger is asked about bare names. The object reaches `Settings`, which
    is what actually carries the decision.
    """
    if isinstance(node.value, ast.Name):
        return node.value.id
    if isinstance(node.value, ast.Attribute) and isinstance(node.value.value, ast.Name):
        return node.value.value.id
    return None


def reachable(module: ast.Module, tree: str) -> tuple[set[str], dict[str, str], list[str]]:
    """What `dspy.X` resolves to, which of those are aliases, and the packages starred in.

    A star import re-exports the package's `__all__`, so the six of them carry most of the surface
    — every module, optimizer and signature name a program uses arrives that way rather than being
    written in the entry point.
    """
    names: set[str] = set()
    aliases: dict[str, str] = {}
    starred: list[str] = []
    for node in ast.walk(module):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    names.add(alias.asname or alias.name)
                    continue
                package = (node.module or "").removeprefix("dspy.").replace(".", "/")
                starred.append(package)
                names.update(exported(f"{package}/__init__.py", tree) or [])
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if not isinstance(target, ast.Name) or target.id.startswith("_"):
                    continue
                names.add(target.id)
                if stands_for := aliased(node):
                    aliases[target.id] = stands_for
    return {n for n in names if not n.startswith("_")}, aliases, starred


def main() -> int:
    tag = pinned_tag()
    tree = pinned_tree(tag)
    print(f"==> Every `dspy.X` a user can reach ({tag})")

    module = entry_point(tree)
    if module is None:
        print(
            f"\ntop-level gate FAILED: cannot read dspy/__init__.py at {tag}.\n"
            "  Run: git submodule update --init third_party/dspy"
        )
        return 1

    names, aliases, starred = reachable(module, tree)
    known = answered(tomllib.loads(LEDGER.read_text())) | decided_elsewhere(tree)
    instances = constructed_from(module, tree)

    def resolved(name: str) -> bool:
        """Whether this name, or what it stands for, is decided.

        A name that stands for something is judged by that something and never by its own
        spelling: `cache = DSPY_CACHE` matched `Embeddings.cache` on the bare word alone, which
        is an answer about a different thing. Following the chain instead — alias to object,
        object to the class it is an instance of — is what makes the match mean something.
        """
        step = aliases.get(name) or instances.get(name)
        if step is None:
            return name in known
        seen: set[str] = set()
        while step and step not in seen:
            if step in known:
                return True
            seen.add(step)
            step = aliases.get(step) or instances.get(step)
        return False

    missing = sorted(n for n in names if not resolved(n))

    print(f"    {len(names)} names, from the entry point and {len(starred)} star-imported packages")
    if missing:
        print(f"\ntop-level gate FAILED: {len(missing)} name(s) the ledger says nothing about:\n")
        for name in missing:
            print(f"      ? dspy.{name}")
        print(
            "\n  Each needs a row in api_ledger.toml — mapped, divergence, todo or deferred.\n"
            "  A name nothing classifies is a gap nobody decided on."
        )
        return 1
    print("    every one answered — a ledger row, or a module decided in unported_modules.toml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
