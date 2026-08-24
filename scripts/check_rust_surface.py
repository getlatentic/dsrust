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

#: **Zero, as of 2026-08-24.** Every public item this crate exposes has a ledger answer, so this
#: has stopped being a ratchet and become an ordinary gate: a new `pub` without an entry fails the
#: run, exactly as an unmapped dspy symbol fails the three tables walking the other way.
#:
#: **A bare name only counts where it identifies one thing.** The check used to match by identifier
#: part alone, so a `new` written in any entry's `rust` accounted for every `::new` in every crate —
#: 35 of them. Measured at the zero: 408 items rested on a name rather than a key, and 117 of those
#: were genuinely ambiguous (`new`, `is_empty`, `len`, `get`, `format`, `set_lm`…) while 291 were one
#: type re-exported at several paths, where the name really does identify it.
#:
#: So `accounted_for` now asks the surface itself: a bare name is evidence only when exactly one
#: public item ends in it. Anything ambiguous has to be named `Owner::method` or keyed outright,
#: which is `api-5`'s own prescription — it turns a reading list into a gate.
#:
#: At zero under that rule: **526 keyed by their own path, 291 by a name that identifies exactly one
#: item, 13 by a qualified `Owner::method`, and none by a bare ambiguous word.** Turning the rule on
#: reopened 104 entries that had read as classified an hour earlier.
#:
#: **And the zero is over an incomplete surface.** `rust_surface.py` descends only into `pub mod`,
#: but a `pub fn` on a *public type* is public API wherever it is written — and 208 of them live in
#: private modules, which is a quarter as many again as the 830 counted here. `MIPROv2` has 19
#: builder methods that way, `GEPA` 14, `Predict` 10. Filed as `surface-private-modules`; until it
#: lands, this gate is a floor on a subset rather than on the whole.
BASELINE = 0


def named_in_ledger(ledger: dict) -> tuple[set[str], set[str]]:
    """What every ledger entry's `rust` points at: the bare names, and the qualified pairs.

    An entry's `rust` is written for a reader — `LmRequest::from_items`, `Parallel`, `Predict::demos`
    — so both halves are kept. The bare names are what a re-exported type matches on; the pairs are
    what a method has to match on when its own name is ambiguous.
    """
    names: set[str] = set()
    pairs: set[str] = set()
    for table in ledger.values():
        if not isinstance(table, dict):
            continue
        for entry in table.values():
            if not isinstance(entry, dict):
                continue
            for token in str(entry.get("rust", "")).replace(",", " ").split():
                token = token.strip("`(),.")
                if "::" in token:
                    pairs.add(token.rsplit("::", 2)[-2] + "::" + token.rsplit("::", 1)[-1])
                for part in token.split("::"):
                    part = part.strip("`(),.")
                    if part and (part[0].isalpha() or part[0] == "_"):
                        names.add(part)
    return names, pairs


def accounted_for(item: str, names: set[str], pairs: set[str], ambiguous: set[str]) -> bool:
    """Whether an entry's `rust` field is evidence *for this item* rather than for the word.

    A bare name counts only where exactly one public item ends in it. Where several do — 35 types
    have a `new` — the ledger has to say which, as `Owner::method`.
    """
    segment = item.rsplit("::", 1)[-1]
    if segment not in ambiguous:
        return segment in names
    parts = item.split("::")
    return len(parts) >= 2 and f"{parts[-2]}::{segment}" in pairs


def main() -> None:
    ledger = tomllib.loads(LEDGER.read_text())
    names, pairs = named_in_ledger(ledger)
    classified = set(ledger.get("rust_only", {}))

    found = surface()
    every = [item for items in found.values() for item in items]
    # Which last segments name more than one *distinct* item. A type re-exported at four paths is
    # not ambiguous — the name identifies it — so the owner is what decides, not the count.
    # The owner is the *type* a method hangs off, and nothing for a module-level item: a type
    # re-exported at four module paths has one owner (none) and stays unambiguous, while thirty-five
    # types each having a `new` gives that segment thirty-five owners.
    owners: dict[str, set[str]] = {}
    for item in every:
        parts = item.split("::")
        holder = parts[-2] if len(parts) >= 2 and parts[-2][:1].isupper() else ""
        owners.setdefault(parts[-1], set()).add(holder)
    ambiguous = {segment for segment, seen in owners.items() if len(seen) > 1}

    unclassified: list[str] = []
    total = 0
    for crate, items in sorted(found.items()):
        for item in sorted(items):
            total += 1
            if item in classified or accounted_for(item, names, pairs, ambiguous):
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
