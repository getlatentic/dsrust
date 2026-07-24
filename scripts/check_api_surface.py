"""Hold the API-surface ledger to the real dspy surface and the real Rust tree.

`api_surface.py` says what dspy publicly defines in the ported modules. `api_ledger.toml` says
what this crate did about each symbol. This binds the two so neither can drift:

  * a dspy symbol absent from the ledger fails the run — triage it (map, justify, or todo);
  * a ledger entry naming a symbol dspy no longer defines fails — delete it;
  * a `mapped` entry whose Rust identifier is defined nowhere in the tree fails — the mapping rotted.

`todo` entries do not fail: they are the acknowledged API backlog, and the run prints them so the
gap is visible rather than implied. Method-level coverage of mapped classes is reported too, as the
next thing to tighten.

Run by the upstream runner, so an API claim and its evidence cannot part company.
"""

from __future__ import annotations

import pathlib
import re
import sys
import tomllib

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from api_surface import PORTED_MODULES, full_surface

ROOT = pathlib.Path(__file__).parent.parent
LEDGER = ROOT / "scripts" / "api_ledger.toml"
#: Where a Rust counterpart may live — the crate plus its workspace members.
RUST_TREES = ["src", "derive/src", "tpe/src", "gepa/src", "pyrng/src"]
#: What counts as *defining* an identifier, so a passing reference to a common word (`Type`, `Code`)
#: is not mistaken for the thing existing.
DEFINES = ("struct", "enum", "trait", "type", "fn", "mod", "const", "static")


def rust_source() -> str:
    parts = []
    for tree in RUST_TREES:
        for path in (ROOT / tree).rglob("*.rs"):
            parts.append(path.read_text())
    return "\n".join(parts)


def is_defined(identifier: str, source: str) -> bool:
    """The identifier appears as a definition or a re-export, not merely in passing."""
    keyword = r"(?:" + "|".join(DEFINES) + r")\s+" + re.escape(identifier) + r"\b"
    reexport = r"pub use [^\n]*\b" + re.escape(identifier) + r"\b"
    return re.search(keyword, source) is not None or re.search(reexport, source) is not None


def top_level_keys(surface: dict) -> set[str]:
    keys = set()
    for module, api in surface.items():
        keys.update(f"{module}::{cls}" for cls in api["classes"])
        keys.update(f"{module}::{fn}" for fn in api["functions"])
    return keys


def method_coverage(surface: dict, ledger: dict, source: str) -> tuple[int, int, list[str]]:
    """Same-name Rust coverage of the public methods of mapped classes — informational."""
    have = total = 0
    thin = []
    for module, api in surface.items():
        for cls, methods in api["classes"].items():
            entry = ledger.get(f"{module}::{cls}", {})
            if entry.get("status") != "mapped":
                continue
            named = [m for m in methods if m != "__call__"]
            if not named:
                continue
            hit = sum(1 for m in named if re.search(r"fn\s+" + re.escape(m) + r"\b", source))
            have += hit
            total += len(named)
            if hit < len(named):
                thin.append(f"{module}::{cls}  {hit}/{len(named)}")
    return have, total, thin


def main() -> None:
    surface = full_surface()
    ledger = tomllib.loads(LEDGER.read_text())["symbols"]
    defined = top_level_keys(surface)
    source = rust_source()

    unclassified = sorted(defined - set(ledger))
    stale = sorted(set(ledger) - defined)
    broken = sorted(
        key
        for key, entry in ledger.items()
        if key in defined
        and entry.get("status") == "mapped"
        and not is_defined(entry["rust"], source)
    )

    failures = []
    if unclassified:
        failures.append(f"{len(unclassified)} dspy symbol(s) not in the ledger:")
        failures += [f"    + {k}" for k in unclassified]
    if stale:
        failures.append(f"{len(stale)} ledger entr(ies) dspy no longer defines:")
        failures += [f"    - {k}" for k in stale]
    if broken:
        failures.append(f"{len(broken)} mapped entr(ies) whose Rust identifier is undefined:")
        failures += [f"    ? {k} -> {ledger[k]['rust']}" for k in broken]

    counts = {"mapped": 0, "divergence": 0, "deferred": 0, "todo": 0}
    for key in defined:
        entry = ledger.get(key)
        if entry:
            counts[entry["status"]] = counts.get(entry["status"], 0) + 1
    total = len(defined)
    resolved = counts["mapped"] + counts["divergence"] + counts["deferred"]

    print(f"API surface (top-level, {len(PORTED_MODULES)} ported modules): {total} symbols")
    print(f"  mapped     : {counts['mapped']}")
    print(f"  divergence : {counts['divergence']}")
    print(f"  deferred   : {counts['deferred']} (out of 1.0 scope)")
    print(f"  todo       : {counts['todo']} (1.0 backlog)")
    if total:
        print(f"  resolved   : {resolved}/{total} ({100 * resolved // total}%)")

    todos = sorted(k for k in defined if ledger.get(k, {}).get("status") == "todo")
    if todos:
        print("  API backlog (todo):")
        for key in todos:
            print(f"    · {key} — {ledger[key]['reason']}")

    have, mtotal, thin = method_coverage(surface, ledger, source)
    if mtotal:
        print(f"method same-name coverage (informational): {have}/{mtotal} across mapped classes")
        for line in thin:
            print(f"    {line}")

    if failures:
        print("\nAPI-surface gate FAILED:")
        for line in failures:
            print(f"  {line}")
        sys.exit(1)
    print("\nAPI-surface gate: OK (every dspy symbol mapped, justified, or tracked as todo)")


if __name__ == "__main__":
    main()
