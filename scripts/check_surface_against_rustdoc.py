#!/usr/bin/env python3
"""Hold `rust_surface.py`'s line-reading walk to rustdoc's own idea of the public API.

The walk is regexes over source lines. That is cheap, needs no nightly, and runs in the gate — and
it is wrong in ways that are invisible from the inside. Three classes were found by asking rustdoc
instead, each of which had been silently missing for as long as the walk existed:

  * **Exported macros.** `macro_rules!` carries no `pub`, and `#[macro_export]` puts the name at the
    crate root wherever it is written. `example!` and `input!` — the most caller-facing things this
    crate has — were not in the surface at all.
  * **Public struct fields.** `pub score: f64` is API a caller reads and writes.
  * **A type behind nested generics.** `impl<E: Iterator<Item = LmStreamEvent>> LmStream<E>` defeats
    a `<[^>]*>` pattern, because that class stops at the *inner* `>`.

It also found an export that was simply missing: `MIPROv2::compile_traced` answers with
`Vec<Trial>`, and `Trial` was `pub` inside a private module — a value a caller could hold and not
name. rustdoc lists an item by where it is *defined*, which is what made that visible.

Not part of the gate: it needs a nightly toolchain, and a gate every contributor cannot run is one
that rots. Run it when the walk changes.

    ./scripts/check_surface_against_rustdoc.py
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from rust_surface import surface  # noqa: E402

#: Enum variants. rustdoc names each one; the walk records the enum. A variant is nameable API, but
#: what the ledger asks — has somebody justified this existing — is answered by the enum's entry, and
#: 96 lines saying "one member of a closed set" would bury the entries that carry information.
#: Recorded as a decision rather than left as a silent difference.
IGNORED_KINDS = {"variant"}


def rustdoc_api() -> tuple[dict[str, str], set[str]]:
    """Named items and public inherent methods, as rustdoc sees them."""
    built = subprocess.run(
        ["cargo", "+nightly", "doc", "--no-deps", "-p", "dsrust"],
        cwd=ROOT,
        env={**__import__("os").environ, "RUSTDOCFLAGS": "-Z unstable-options --output-format json"},
        capture_output=True,
        text=True,
    )
    if built.returncode != 0:
        raise SystemExit(f"rustdoc JSON needs a nightly toolchain:\n{built.stderr[-600:]}")
    doc = json.loads((ROOT / "target" / "doc" / "dsrust.json").read_text())
    index = {int(k): v for k, v in doc["index"].items()}
    paths = {int(k): v for k, v in doc["paths"].items()}
    local = index[doc["root"]]["crate_id"]

    named = {
        "::".join(info["path"]): info["kind"]
        for info in paths.values()
        if info.get("crate_id") == local
        and info["kind"] not in ({"module", "primitive"} | IGNORED_KINDS)
    }
    methods = set()
    for item in index.values():
        block = item.get("inner", {}).get("impl")
        if not block or block.get("trait"):
            continue
        holder = (block.get("for") or {}).get("resolved_path", {}).get("path")
        if not holder:
            continue
        for member in block.get("items", []):
            found = index.get(member)
            if found and found.get("name") and found.get("visibility") == "public":
                methods.add(f"{holder.split('::')[-1]}::{found['name']}")
    return named, methods


def renamed_exports() -> set[str]:
    """Definition names that reach a caller under a different one.

    `pub use legacy_request::sanitized as sanitized_legacy_message` is one item with two names:
    rustdoc lists where it was defined, the walk lists what a caller writes. Neither is wrong, and a
    diff that does not know about the rename reports a gap that is not there.
    """
    import re

    alias = re.compile(r"^\s*pub\s+use\s+[^;]*?::([A-Za-z_][A-Za-z0-9_]*)\s+as\s+[A-Za-z_]", re.M)
    found: set[str] = set()
    for path in (ROOT / "crates").rglob("*.rs"):
        found |= set(alias.findall(path.read_text(errors="ignore")))
    return found


def main() -> int:
    named, methods = rustdoc_api()
    aliased = renamed_exports()
    mine = {i for items in surface().values() for i in items if i.startswith("dsrust::")}
    my_names = {i.split("::")[-1] for i in mine}
    my_pairs = {"::".join(i.split("::")[-2:]) for i in mine if i.count("::") >= 2}

    missing_items = sorted(
        n for n in named if n.split("::")[-1] not in my_names and n.split("::")[-1] not in aliased
    )
    missing_methods = sorted(methods - my_pairs)

    print(f"rustdoc: {len(named)} named items, {len(methods)} inherent methods")
    print(f"walk:    {len(mine)} items for dsrust")
    if not missing_items and not missing_methods:
        print("\nSurface-vs-rustdoc: OK (the walk sees everything rustdoc names)")
        return 0
    print(f"\n{len(missing_items)} named item(s) and {len(missing_methods)} method(s) rustdoc has "
          "and the walk does not:")
    for name in missing_items[:40]:
        print(f"    + {named[name]:12} {name}")
    for name in missing_methods[:40]:
        print(f"    + method       {name}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
