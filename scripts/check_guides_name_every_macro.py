#!/usr/bin/env python3
"""Every exported macro is named in a guide.

`check_docs.py` compiles the blocks the guides *do* contain, which says nothing about a macro no
block ever uses. Two were public, in the ledger, and mentioned nowhere a reader would look:
`make_signature!` — the compile-checked way to build a `Signature`, and the only way to hand one to
`ReActV2::new` from a literal — and `example!`, which is how a trainset is written at all. Both had
been exported for months.

A macro is the shape of the crate a caller types. One nobody wrote down is a promise nobody made.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GUIDES = ("README.md", "docs/usage.md")

#: Exported for the other macros to expand into, never for a caller to type. Each one here is a
#: deliberate exception, not a backlog: it takes a name only because `macro_rules!` has no way to
#: be visible across modules without one.
PLUMBING = {"asks_with_a_prediction"}


def exported() -> set[str]:
    """Every macro a caller can reach: `#[macro_export]` here, plus the derive crate's three kinds.

    Each marker is counted and then resolved, and a marker that resolves to nothing is an error
    rather than a silent absence. The first version of this walked from `#[proc_macro]` over
    attribute lines only, so a doc comment between the marker and its `pub fn` hid three macros —
    and the gate reported OK on a list missing the very one that prompted it.
    """
    names: set[str] = set()
    markers = 0
    for source in (ROOT / "crates/dsrust/src").rglob("*.rs"):
        text = source.read_text()
        markers += text.count("#[macro_export]")
        for found in re.finditer(r"#\[macro_export\][^!]*?macro_rules!\s+(\w+)", text, re.S):
            names.add(found.group(1))
    derive = (ROOT / "crates/dsrust-derive/src/lib.rs").read_text()
    for kind in ("proc_macro_derive", "proc_macro", "proc_macro_attribute"):
        for marker in re.finditer(rf"#\[{kind}[(\]]", derive):
            markers += 1
            named = re.search(r"\bpub fn (\w+)", derive[marker.end() :])
            if named is None:
                raise SystemExit(f"a {kind} marker in dsrust-derive names no function")
            names.add(named.group(1))
    # `proc_macro_derive` is spelled `#[proc_macro_derive(Signature, …)]`, so the derive's own
    # name is the attribute's argument rather than the function it sits on.
    names |= set(re.findall(r"proc_macro_derive\((\w+)", derive))
    names -= {"derive_signature", "derive_module"}
    if len(names) + len(PLUMBING) < markers:
        raise SystemExit(f"{markers} macro markers resolved to only {len(names)} names")
    return names - PLUMBING


def named_in_the_guides() -> str:
    return "\n".join((ROOT / guide).read_text() for guide in GUIDES)


def main() -> int:
    print("==> Every exported macro is named in a guide")
    guides = named_in_the_guides()
    missing = sorted(
        name
        for name in exported()
        # The four spellings a guide can use: `name!`, `#[derive(Name)]`, `#[name]`, `Name(…)`.
        if not re.search(rf"\b{name}!|derive\([^)]*\b{name}\b|#\[{name}\]|\b{name}\(", guides)
    )
    print(f"    exported macros: {len(exported())}, named in a guide: {len(exported()) - len(missing)}")
    if missing:
        print("\nGuide-macro gate FAILED: exported but named in no guide:")
        for name in missing:
            print(f"    + {name}")
        print("\n  Write it into README.md or docs/usage.md, or make it `pub(crate)`.")
        return 1
    print("Guide-macro gate: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
