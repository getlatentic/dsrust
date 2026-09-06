#!/usr/bin/env python3
"""Require a qualified ledger or module decision for every public dspy binding."""

from __future__ import annotations

import tomllib
from pathlib import Path

from check_pinned_all import pinned_tag, pinned_tree
from dspy_exports import ExportError, PinnedModules, PublicExports

ROOT = Path(__file__).resolve().parent.parent
DSPY = ROOT / "third_party" / "dspy"
LEDGER = ROOT / "scripts" / "api_ledger.toml"
UNPORTED = ROOT / "scripts" / "unported_modules.toml"


def missing_exports(
    exports: dict[str, str], ledger: dict, unported: dict
) -> dict[str, str]:
    decided_symbols = set().union(
        *(ledger.get(table, {}) for table in ("symbols", "methods", "constructors"))
    )
    return {
        name: symbol
        for name, symbol in exports.items()
        if symbol not in decided_symbols and symbol.partition("::")[0] not in unported
    }


def main() -> int:
    tag = pinned_tag()
    print(f"==> Every public dspy binding ({tag})")
    try:
        modules = PinnedModules(DSPY, pinned_tree(tag))
        exports = PublicExports(modules).exports()
        ledger = tomllib.loads(LEDGER.read_text())
        unported = tomllib.loads(UNPORTED.read_text())["modules"]
        missing = missing_exports(exports, ledger, unported)
    except (ExportError, OSError, tomllib.TOMLDecodeError) as error:
        print(f"\ntop-level gate FAILED: {error}")
        return 1

    print(f"    {len(exports)} public names resolved to their defining symbols")
    if missing:
        print("\ntop-level gate FAILED: exports without a ledger or module decision:")
        for name, symbol in sorted(missing.items()):
            print(f"    dspy.{name} -> {symbol}")
        return 1
    print("    every export has a qualified ledger or module decision")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
