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
    "crates/dsrust-json-repair/src",
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


def parameter_keys(surface: dict, ledger: dict, methods: dict) -> set[str]:
    """Every parameter each mapped method and mapped free function accepts.

    The `constructors` table opened this and stopped one method short. `__init__` is not the only
    callable whose parameters are API: `MIPROv2.compile` takes eighteen arguments, as much
    configuration as its constructor, and a gate that checks only that a method named `compile`
    exists sees none of them. Gated on the owner being mapped, for the reason the other tables are —
    an unported method's parameters are the method's own gap, counted once.
    """
    keys = set()
    for module, api in surface.items():
        for cls, taken in api.get("method_params", {}).items():
            if ledger.get(f"{module}::{cls}", {}).get("status") != "mapped":
                continue
            for method, params in taken.items():
                if methods.get(f"{module}::{cls}.{method}", {}).get("status") != "mapped":
                    continue
                keys.update(f"{module}::{cls}.{method}.{name}" for name in params)
        for fn, params in api.get("function_params", {}).items():
            if ledger.get(f"{module}::{fn}", {}).get("status") != "mapped":
                continue
            keys.update(f"{module}::{fn}.{name}" for name in params)
    return keys


def report(label: str, defined: set[str], entries: dict) -> None:
    """One table's tally, in the shape every table reports."""
    tally = {"mapped": 0, "divergence": 0, "deferred": 0, "todo": 0}
    for key in defined:
        entry = entries.get(key)
        if entry:
            tally[entry["status"]] = tally.get(entry["status"], 0) + 1
    total = len(defined)
    resolved = tally["mapped"] + tally["divergence"] + tally["deferred"]
    print(f"{label}: {total}")
    print(f"  mapped     : {tally['mapped']}")
    print(f"  divergence : {tally['divergence']}")
    print(f"  deferred   : {tally['deferred']} (out of 1.0 scope)")
    print(f"  todo       : {tally['todo']} (1.0 backlog)")
    if total:
        print(f"  resolved   : {resolved}/{total} ({100 * resolved // total}%)")


def main() -> None:
    surface = full_surface()
    ledger_file = tomllib.loads(LEDGER.read_text())
    ledger = ledger_file["symbols"]
    methods = ledger_file["methods"]
    constructors = ledger_file["constructors"]
    parameters = ledger_file["parameters"]
    defined = top_level_keys(surface)
    source = rust_source()

    # The same three checks over the top-level symbols and over the methods of every mapped class,
    # so a method that quietly went missing fails the run exactly as a symbol does.
    defined_methods = method_keys(surface, ledger)
    defined_params = constructor_keys(surface, ledger)
    defined_arguments = parameter_keys(surface, ledger, methods)
    entries = {**ledger, **methods, **constructors, **parameters}
    unclassified = (
        sorted(defined - set(ledger))
        + sorted(defined_methods - set(methods))
        + sorted(defined_params - set(constructors))
        + sorted(defined_arguments - set(parameters))
    )
    stale = (
        sorted(set(ledger) - defined)
        + sorted(set(methods) - defined_methods)
        + sorted(set(constructors) - defined_params)
        + sorted(set(parameters) - defined_arguments)
    )
    broken = sorted(
        key
        for key, entry in entries.items()
        if key in defined | defined_methods | defined_params | defined_arguments
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

    report(f"API surface (top-level, {len(PORTED_MODULES)} ported modules)", defined, ledger)
    report("methods of mapped classes", defined_methods, methods)
    report("constructor parameters of mapped classes", defined_params, constructors)
    report("parameters of mapped methods and functions", defined_arguments, parameters)

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
        "\nAPI-surface gate: OK (every dspy symbol, method, constructor parameter and method "
        "parameter mapped, justified, or tracked)"
    )


if __name__ == "__main__":
    main()
