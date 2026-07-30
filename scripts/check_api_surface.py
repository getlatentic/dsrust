"""Hold the API-surface ledger to the real dspy surface and the real Rust tree.

`api_surface.py` says what dspy publicly defines in the ported modules. `api_ledger.toml` says
what this crate did about each symbol. This binds the two so neither can drift:

  * a dspy symbol absent from the ledger fails the run — triage it (map, justify, or todo);
  * a ledger entry naming a symbol dspy no longer defines fails — delete it;
  * a `mapped` entry whose Rust identifier is defined nowhere in the tree fails — the mapping rotted.

The three checks run over dspy's top-level symbols *and* over every public method of a class the
ledger maps, so a class cannot pass while missing half of what its Python counterpart does.

`todo` entries do not fail: they are the acknowledged API backlog, and the run prints them so the
gap is visible rather than implied.

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
RUST_TREES = [
    "crates/dsrust/src",
    "crates/dsrust-derive/src",
    "crates/dsrust-tpe/src",
    "crates/dsrust-gepa/src",
    "crates/pyrng/src",
]
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
    """The identifier appears as a definition or a re-export, not merely in passing.

    A public struct field counts. Several of dspy's constructor parameters map to one here —
    `ChatAdapter(use_json_adapter_fallback=...)` is `ChatAdapter { use_json_adapter_fallback }` —
    and a field is as much API as a method is.

    `Type::method` is checked as both halves. Without that this only accepted a bare name, which is
    what pushed entries toward generic ones: `LMRequest.from_call` was mapped to `new` because
    writing `LmRequest::from_items` would have failed the gate. A bare `new` proves only that the
    word appears somewhere in 160 files, so being specific has to be the thing that passes.
    """
    if "::" in identifier:
        owner, member = identifier.rsplit("::", 1)
        return is_defined(owner, source) and is_defined(member, source)
    keyword = r"(?:" + "|".join(DEFINES) + r")\s+" + re.escape(identifier) + r"\b"
    reexport = r"pub use [^\n]*\b" + re.escape(identifier) + r"\b"
    field = r"pub\s+" + re.escape(identifier) + r"\s*:"
    return any(
        re.search(pattern, source) is not None for pattern in (keyword, reexport, field)
    )


def top_level_keys(surface: dict) -> set[str]:
    keys = set()
    for module, api in surface.items():
        keys.update(f"{module}::{cls}" for cls in api["classes"])
        keys.update(f"{module}::{fn}" for fn in api["functions"])
    return keys


def method_keys(surface: dict, ledger: dict) -> set[str]:
    """Every public method of a mapped class, as the `module::Class.method` key the ledger uses.

    `__call__` is skipped: invoking a module is `forward` here, which is already classified.
    """
    keys = set()
    for module, api in surface.items():
        for cls, methods in api["classes"].items():
            if ledger.get(f"{module}::{cls}", {}).get("status") != "mapped":
                continue
            keys.update(f"{module}::{cls}.{m}" for m in methods if m != "__call__")
    return keys


def constructor_keys(surface: dict, ledger: dict) -> set[str]:
    """Every parameter each mapped class's `__init__` accepts.

    Gated for the same reason methods are: a name that quietly disappears from a constructor is a
    caller who can no longer say something dspy lets them say. Only mapped classes, since an
    unported class's parameters are the class's own gap.
    """
    keys = set()
    for module, api in surface.items():
        for cls, params in api.get("constructors", {}).items():
            if ledger.get(f"{module}::{cls}", {}).get("status") != "mapped":
                continue
            keys.update(f"{module}::{cls}.{name}" for name in params)
    return keys


def main() -> None:
    surface = full_surface()
    ledger_file = tomllib.loads(LEDGER.read_text())
    ledger = ledger_file["symbols"]
    methods = ledger_file["methods"]
    constructors = ledger_file["constructors"]
    defined = top_level_keys(surface)
    source = rust_source()

    # The same three checks over the top-level symbols and over the methods of every mapped class,
    # so a method that quietly went missing fails the run exactly as a symbol does.
    defined_methods = method_keys(surface, ledger)
    defined_params = constructor_keys(surface, ledger)
    entries = {**ledger, **methods, **constructors}
    unclassified = (
        sorted(defined - set(ledger))
        + sorted(defined_methods - set(methods))
        + sorted(defined_params - set(constructors))
    )
    stale = (
        sorted(set(ledger) - defined)
        + sorted(set(methods) - defined_methods)
        + sorted(set(constructors) - defined_params)
    )
    broken = sorted(
        key
        for key, entry in entries.items()
        if key in defined | defined_methods | defined_params
        and entry.get("status") == "mapped"
        and not is_defined(entry["rust"], source)
    )

    failures = []
    if unclassified:
        failures.append(f"{len(unclassified)} dspy symbol(s)/method(s) not in the ledger:")
        failures += [f"    + {k}" for k in unclassified]
    if stale:
        failures.append(f"{len(stale)} ledger entr(ies) dspy no longer defines:")
        failures += [f"    - {k}" for k in stale]
    if broken:
        failures.append(f"{len(broken)} mapped entr(ies) whose Rust identifier is undefined:")
        failures += [f"    ? {k} -> {entries[k]['rust']}" for k in broken]

    counts = {"mapped": 0, "divergence": 0, "deferred": 0, "todo": 0}
    for key in defined:
        entry = ledger.get(key)
        if entry:
            counts[entry["status"]] = counts.get(entry["status"], 0) + 1
    method_counts = {"mapped": 0, "divergence": 0, "deferred": 0, "todo": 0}
    for key in defined_methods:
        entry = methods.get(key)
        if entry:
            method_counts[entry["status"]] = method_counts.get(entry["status"], 0) + 1
    total = len(defined)
    resolved = counts["mapped"] + counts["divergence"] + counts["deferred"]

    print(f"API surface (top-level, {len(PORTED_MODULES)} ported modules): {total} symbols")
    print(f"  mapped     : {counts['mapped']}")
    print(f"  divergence : {counts['divergence']}")
    print(f"  deferred   : {counts['deferred']} (out of 1.0 scope)")
    print(f"  todo       : {counts['todo']} (1.0 backlog)")
    if total:
        print(f"  resolved   : {resolved}/{total} ({100 * resolved // total}%)")

    method_total = len(defined_methods)
    method_resolved = method_counts["mapped"] + method_counts["divergence"] + method_counts["deferred"]
    print(f"methods of mapped classes: {method_total}")
    print(f"  mapped     : {method_counts['mapped']}")
    print(f"  divergence : {method_counts['divergence']}")
    print(f"  deferred   : {method_counts['deferred']} (out of 1.0 scope)")
    print(f"  todo       : {method_counts['todo']} (1.0 backlog)")
    if method_total:
        print(f"  resolved   : {method_resolved}/{method_total} ({100 * method_resolved // method_total}%)")

    param_counts = {"mapped": 0, "divergence": 0, "deferred": 0, "todo": 0}
    for key in defined_params:
        entry = constructors.get(key)
        if entry:
            param_counts[entry["status"]] = param_counts.get(entry["status"], 0) + 1
    param_total = len(defined_params)
    param_resolved = (
        param_counts["mapped"] + param_counts["divergence"] + param_counts["deferred"]
    )
    print(f"constructor parameters of mapped classes: {param_total}")
    print(f"  mapped     : {param_counts['mapped']}")
    print(f"  divergence : {param_counts['divergence']}")
    print(f"  deferred   : {param_counts['deferred']} (out of 1.0 scope)")
    print(f"  todo       : {param_counts['todo']} (1.0 backlog)")
    if param_total:
        share = 100 * param_resolved // param_total
        print(f"  resolved   : {param_resolved}/{param_total} ({share}%)")

    todos = sorted(k for k in defined if ledger.get(k, {}).get("status") == "todo")
    todos += sorted(k for k in defined_methods if methods.get(k, {}).get("status") == "todo")
    if todos:
        print("  API backlog (todo):")
        for key in todos:
            entry = ledger.get(key) or methods[key]
            print(f"    · {key} — {entry['reason']}")

    if failures:
        print("\nAPI-surface gate FAILED:")
        for line in failures:
            print(f"  {line}")
        sys.exit(1)
    print(
        "\nAPI-surface gate: OK (every dspy symbol, method and constructor parameter mapped, "
        "justified, or tracked)"
    )


if __name__ == "__main__":
    main()
