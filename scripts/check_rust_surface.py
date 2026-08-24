#!/usr/bin/env python3
"""Gate the surface this crate invented, the way `check_api_surface.py` gates the one it ports.

That script walks **dspy to Rust** and is at 247/247 across four tables. It says nothing about the
other direction, so anything this crate added on its own has never been checked by anything — which
is how `LM::with_capabilities` survived with no callers at all, not in `src`, not in tests, not in
the docs, not in the README.

Measured 2026-08-01: **781 public items, 445 of which no ledger entry mentioned.** That is a floor to
work down rather than a failure to fix in one go — an item is fine to have invented, and the point
is that having invented it is a claim somebody wrote down.

A **ratchet**, not a target of zero, for the same reason `run_mutants.sh` is one: a gate that starts
six hundred red either blocks every run or gets switched off. The number may fall and never rise. A
new unclassified public item is a new claim nobody has made.

Classify one by adding it to `[rust_only]` in `scripts/api_ledger.toml`:

    "dsrust::optimize::Scoring" = { status = "divergence", reason = "..." }

`mapped` means it has a dspy counterpart under another name; `divergence` means this crate invented
it and the reason says why; `deferred` means it is out of 1.0 scope.

    ./scripts/check_rust_surface.py
"""

from __future__ import annotations

import pathlib
import sys
import tomllib

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from rust_surface import surface

LEDGER = pathlib.Path(__file__).parent / "api_ledger.toml"

#: The measured floor. Lower it when the count falls; never raise it to make a run pass.
BASELINE = 85


def named_in_ledger(ledger: dict) -> set[str]:
    """Every Rust identifier any ledger entry points at, however it spells the path.

    An entry's `rust` is written for a reader — `LmRequest::from_items`, `Parallel`, `Predict::demos`
    — so the parts are what match, not the whole string.
    """
    names = set()
    for table in ledger.values():
        if not isinstance(table, dict):
            continue
        for entry in table.values():
            if not isinstance(entry, dict):
                continue
            for part in str(entry.get("rust", "")).replace("::", " ").replace(",", " ").split():
                part = part.strip("`(),.")
                if part and (part[0].isalpha() or part[0] == "_"):
                    names.add(part)
    return names


def main() -> None:
    ledger = tomllib.loads(LEDGER.read_text())
    accounted = named_in_ledger(ledger)
    classified = set(ledger.get("rust_only", {}))

    found = surface()
    unclassified: list[str] = []
    total = 0
    for crate, items in sorted(found.items()):
        for item in sorted(items):
            total += 1
            if item in classified or item.rsplit("::", 1)[-1] in accounted:
                continue
            unclassified.append(item)

    print(f"public Rust surface: {total} items")
    print(f"  accounted for by a dspy mapping : {total - len(unclassified) - len(classified)}")
    print(f"  classified in [rust_only]       : {len(classified)}")
    print(f"  unclassified                    : {len(unclassified)} (baseline {BASELINE})")

    stale = sorted(classified - {item for items in found.values() for item in items})
    if stale:
        print(f"\n{len(stale)} [rust_only] entr(ies) name an item this crate no longer exposes:")
        for item in stale:
            print(f"    - {item}")
        sys.exit(1)

    if len(unclassified) > BASELINE:
        print(f"\nRust-surface gate FAILED: {len(unclassified)} unclassified, baseline {BASELINE}")
        for item in unclassified[:40]:
            print(f"    + {item}")
        if len(unclassified) > 40:
            print(f"    … and {len(unclassified) - 40} more")
        sys.exit(1)
    if len(unclassified) < BASELINE:
        print(f"\n{BASELINE - len(unclassified)} below the baseline — lower BASELINE to {len(unclassified)}")
    print("\nRust-surface gate: OK")


if __name__ == "__main__":
    main()
