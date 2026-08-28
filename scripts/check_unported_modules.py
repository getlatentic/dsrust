"""Every dspy module is claimed by exactly one list, and the reason for skipping one is checked.

`api_surface.py`'s PORTED_MODULES says which modules this crate reproduces. Nothing said anything
about the rest: 73 modules and 137 public symbols sat outside every gate, justified only by a
parenthesis in a docstring — "not ported (KNN, the avatar/simba optimizers)" — that no rule read
and that still named SIMBA the day after SIMBA landed.

Four claims are checked here, not asserted:

* **Exhaustive and disjoint.** A pinned `.py` in neither list fails, so a module arriving upstream
  lands as backlog or as a decision instead of vanishing. A module in both fails too.
* **`no_surface` is re-walked.** The gate asks `api_surface` for the module's public classes and
  functions and fails if any appeared, because "defines nothing public" is a measurement that can
  go stale in one upstream commit.
* **`upstream_dead` is re-run.** The cited golden must name the module and record a non-null
  `raises`. Upstream repairing `AvatarOptimizer` turns this red, which is the point: the module
  becomes a port backlog item the moment it can run.
* **The backlog only shrinks.** BACKLOG is a floor, so porting one lowers it and an unclassified
  new module cannot hide inside it.

    .venv/bin/python scripts/check_unported_modules.py
"""

from __future__ import annotations

import ast
import json
import pathlib
import sys
import tomllib

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from api_surface import DSPY, PORTED_MODULES, surface_of

ROOT = pathlib.Path(__file__).resolve().parent.parent
TABLE = ROOT / "scripts" / "unported_modules.toml"
GOLDENS = ROOT / "crates" / "dsrust" / "tests" / "conformance"

#: Modules in scope for 1.0 that are not ported yet. A floor: porting one lowers it, and nothing
#: may raise it — a new upstream module is either ported, decided about, or lowers nothing.
BACKLOG = 4

STATUSES = {"backlog", "upstream_dead", "dead_consumer", "out_of_scope", "no_surface"}


def _imports_of(path: pathlib.Path) -> set[str]:
    """The dspy modules `path` imports, as PORTED_MODULES-relative paths.

    Both spellings, because the avatar package uses each: `from dspy.predict.avatar.models import
    Tool` and the relative `from .models import Tool`. A name that resolves to a package rather
    than a module is reported as its `__init__.py`, which is what importing a package runs.
    """
    tree = ast.parse(path.read_text())
    here = path.parent.relative_to(DSPY)
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base = here
                for _ in range(node.level - 1):
                    base = base.parent
                dotted = str(base / (node.module or "").replace(".", "/"))
            elif (node.module or "").startswith("dspy."):
                dotted = node.module[len("dspy.") :].replace(".", "/")
            else:
                continue
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("dspy."):
                    found |= _resolve(alias.name[len("dspy.") :].replace(".", "/"))
            continue
        else:
            continue
        found |= _resolve(dotted)
        for alias in node.names:  # `from dspy.predict import avatar` names a module too
            found |= _resolve(f"{dotted}/{alias.name}")
    return found


def _resolve(dotted: str) -> set[str]:
    dotted = dotted.strip("/")
    for candidate in (f"{dotted}.py", f"{dotted}/__init__.py"):
        if (DSPY / candidate).exists():
            return {candidate}
    return set()


def _check_dead_consumer(module: str, entry: dict, table: dict) -> str | None:
    """`dead_consumer` claims nothing live imports the module. Read every import and check."""
    only_for = entry.get("only_for")
    if not only_for:
        return "dead_consumer needs `only_for` naming the dead modules that reach it"
    dead = {"upstream_dead", "dead_consumer"}
    for named in only_for:
        status = table.get(named, {}).get("status")
        if status not in dead:
            return f"only_for names {named}, which is {status or 'not classified'}, not dead"
    allowed = set(only_for) | {module}
    live = []
    for path in sorted(DSPY.rglob("*.py")):
        rel = str(path.relative_to(DSPY))
        if rel in allowed or module not in _imports_of(path):
            continue
        # A package `__init__` that only re-exports is not a consumer: importing it runs no code
        # that touches these names. Anything else importing them would be live use.
        if table.get(rel, {}).get("status") == "no_surface":
            continue
        live.append(rel)
    if live:
        return f"imported by live {', '.join(live)} — not dead after all"
    return None


def _pinned_modules() -> set[str]:
    return {str(p.relative_to(DSPY)) for p in DSPY.rglob("*.py")}


def _check_no_surface(module: str) -> str | None:
    """`no_surface` claims the module defines no public class or function. Re-walk and see."""
    api = surface_of(DSPY / module)
    found = sorted(list(api["classes"]) + list(api["functions"]))
    if found:
        return f"claims no public surface, but defines {', '.join(found)}"
    return None


def _check_dead(module: str, entry: dict, cache: dict) -> str | None:
    """`upstream_dead` claims the module's entry point raises. The golden constructed it."""
    name = entry.get("evidence")
    if not name:
        return "upstream_dead needs `evidence` naming the golden that constructed it"
    if name not in cache:
        path = GOLDENS / name
        if not path.exists():
            return f"evidence golden {name} does not exist"
        cache[name] = json.loads(path.read_text())
    rows = [e for e in cache[name]["entries"] if e["module"] == module]
    if not rows:
        return f"{name} records nothing for this module — regenerate it"
    alive = [e["name"] for e in rows if e["raises"] is None]
    if alive:
        return (
            f"{', '.join(alive)} constructs now — upstream repaired it, so this is backlog again"
        )
    return None


def main() -> int:
    table = tomllib.loads(TABLE.read_text())["modules"]
    pinned = _pinned_modules()
    ported = set(PORTED_MODULES)
    problems: list[str] = []

    for module in sorted(pinned - ported - set(table)):
        problems.append(f"{module}: in no list — port it, or classify it in unported_modules.toml")
    for module in sorted(set(table) - pinned):
        problems.append(f"{module}: classified but no longer in the pinned tree — delete the row")
    for module in sorted(set(table) & ported):
        problems.append(f"{module}: in PORTED_MODULES *and* unported_modules.toml")

    cache: dict[str, dict] = {}
    for module, entry in sorted(table.items()):
        if module not in pinned:
            continue
        status = entry.get("status")
        if status not in STATUSES:
            problems.append(f"{module}: unknown status {status!r}")
            continue
        if not str(entry.get("reason", "")).strip():
            problems.append(f"{module}: no reason")
        problem = None
        if status == "no_surface":
            problem = _check_no_surface(module)
        elif status == "upstream_dead":
            problem = _check_dead(module, entry, cache)
        elif status == "dead_consumer":
            problem = _check_dead_consumer(module, entry, table)
        if problem:
            problems.append(f"{module}: {problem}")

    counts = {s: sum(1 for e in table.values() if e.get("status") == s) for s in sorted(STATUSES)}
    print(f"unported modules: {len(table)} classified, {len(ported)} ported")
    for status, n in counts.items():
        print(f"  {status:15s} {n}")

    backlog = counts["backlog"]
    if backlog > BACKLOG:
        problems.append(f"backlog is {backlog}, floor {BACKLOG} — a new module cannot land as one")
    if backlog < BACKLOG:
        print(f"  backlog below the floor — lower BACKLOG to {backlog}")

    if problems:
        print(f"\nUnported-modules gate FAILED: {len(problems)} problem(s):")
        for p in problems:
            print(f"    {p}")
        return 1
    print("\nUnported-modules gate: OK (every pinned module is claimed once, and the skips hold)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
