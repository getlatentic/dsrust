"""The crate's public API, enumerated the way `api_surface.py` enumerates dspy's.

`check_api_surface.py` walks **dspy to Rust**: everything upstream defines must be mapped, justified
or tracked. Nothing walked the other way, so anything this crate invented was invisible to every
gate here — which is how `LM::with_capabilities` survived with zero callers anywhere, not in `src`,
not in tests, not in the docs.

Reachability, not a grep for `pub`. An item is public only if every module between it and the crate
root is public too, so `pub fn` inside a private module is not API and `pub use` of one is. A raw
scan counts about sixteen hundred `pub` tokens; almost none of that is surface.

Deliberately text over rustdoc JSON: that needs a nightly toolchain this machine does not carry, and
a check nobody can run is not a check. The cost is that a re-export renaming with `as` is recorded
under the name it is exported as, which is the name a caller writes anyway.
"""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).parent.parent

#: Crates whose public surface is API this project promises. `dsrs-bridge` is the PyO3 shim the
#: upstream suite runs through and is not something a caller depends on.
CRATES = {
    "dsrust": ROOT / "crates" / "dsrust" / "src",
    "gepa": ROOT / "crates" / "dsrust-gepa" / "src",
    "tpe": ROOT / "crates" / "dsrust-tpe" / "src",
    "pyrng": ROOT / "crates" / "pyrng" / "src",
}

#: `pub` items that are surface. A `pub mod` is followed rather than recorded; a `pub use` is
#: recorded under the name it re-exports.
ITEM = re.compile(
    r"^\s*pub\s+"
    r"(?:async\s+|unsafe\s+|const\s+|default\s+)*"
    r"(fn|struct|enum|trait|type|const|static|union)\s+"
    r"([A-Za-z_][A-Za-z0-9_]*)",
    re.M,
)
PUB_MOD = re.compile(r"^\s*pub\s+mod\s+(\w+)\s*;", re.M)
PUB_USE = re.compile(r"^\s*pub\s+use\s+([^;]+);", re.M)

#: An inherent `impl Type` or a `trait Type`, at column zero — where rustfmt puts a top-level item,
#: and the block ends at the matching `}` in column zero. Brace counting would have to survive a `{`
#: inside a string literal or a comment; this does not.
#:
#: `impl Trait for Type` is deliberately not matched. Its methods are the trait's, the trait is
#: recorded where it is declared, and counting them again would tally every impl of `Display`.
OWNER = re.compile(r"^(?:impl(?:<[^>]*>)?\s+(?!.*\bfor\b)|pub\s+trait\s+)([A-Za-z_][A-Za-z0-9_]*)")

#: Only bare `pub` is surface: `pub(crate)` and `pub(super)` cannot be named by a caller, and the
#: patterns above require a space after `pub` so a restricted item never matches in the first place.


def module_file(directory: pathlib.Path, name: str) -> pathlib.Path | None:
    """Where a `mod name;` lives: `name.rs` beside it, or `name/mod.rs` under it."""
    for candidate in (directory / f"{name}.rs", directory / name / "mod.rs"):
        if candidate.exists():
            return candidate
    return None


def reexported(clause: str) -> list[str]:
    """The names a `pub use` clause makes reachable, as a caller would write them."""
    body = clause.strip()
    if body.endswith("*"):
        # A glob re-export: the names come from the module it points at, which the walk reaches on
        # its own if that module is public. Nothing to record here.
        return []
    inner = body[body.index("{") + 1 : body.rindex("}")] if "{" in body else body.rsplit("::", 1)[-1]
    names = []
    for part in inner.split(","):
        part = part.strip()
        if not part or part == "self":
            continue
        # `Thing as Other` is reachable as `Other`.
        names.append(part.split(" as ")[-1].strip().rsplit("::", 1)[-1])
    return [name for name in names if name and name[0].isalpha() or name.startswith("_")]


def walk(crate: str, path: pathlib.Path, prefix: str, seen: set[pathlib.Path]) -> set[str]:
    """Every public item reachable through this module, keyed `crate::path::Name`."""
    if path in seen or not path.exists():
        return set()
    seen.add(path)
    source = path.read_text()
    found = set()

    owner: str | None = None
    for line in source.splitlines():
        if owner and line == "}":
            owner = None
        # The item first, then the block it opens: `pub trait Foo` matches both patterns, and it is
        # declared at the module — only what follows it belongs to `Foo`.
        if match := ITEM.match(line):
            where = f"{prefix}::{owner}" if owner else prefix
            found.add(f"{where}::{match.group(2)}")
        if not line[:1].isspace() and (start := OWNER.match(line)):
            owner = start.group(1)

    for clause in PUB_USE.findall(source):
        for name in reexported(clause):
            found.add(f"{prefix}::{name}")

    directory = path.parent if path.name == "mod.rs" else path.with_suffix("")
    for name in PUB_MOD.findall(source):
        child = module_file(directory, name) or module_file(path.parent, name)
        if child:
            found |= walk(crate, child, f"{prefix}::{name}", seen)
    return found


def surface() -> dict[str, set[str]]:
    """Each crate's public items, keyed by crate."""
    out = {}
    for crate, src in CRATES.items():
        out[crate] = walk(crate, src / "lib.rs", crate, set())
    return out


def main() -> None:
    found = surface()
    total = 0
    for crate, items in sorted(found.items()):
        print(f"{crate}: {len(items)} public items")
        total += len(items)
    print(f"\ntotal: {total}")
    if "--list" in sys.argv:
        for crate, items in sorted(found.items()):
            for item in sorted(items):
                print(f"  {item}")


if __name__ == "__main__":
    main()
