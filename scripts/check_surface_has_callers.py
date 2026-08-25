"""A public item this crate invented must be exercised somewhere, or nothing proves a caller can.

`check_rust_surface.py` asks whether every public item is *classified*. It does not ask whether any
of them is *used*, and the two are different questions: an item can carry a paragraph explaining why
a caller needs it and have no caller anywhere — no test, no example, no line of documentation.

That gap is not small. 344 of the workspace's public items are types or free functions rather than
members, and 148 of them are named nowhere outside the module that declares them. Narrowing to the
ones that also answer for no dspy symbol — so no upstream API promise keeps them public — and to
`dsrust` itself leaves the number below.

**A ratchet, not a target of zero.** Some of these are honestly public and simply undemonstrated: a
`DEFAULT_OPENAI_BASE_URL` is a constant a caller reads and this repo never needs to. What the ratchet
stops is the surface growing without anything showing the growth works. Lower the floor by giving an
item a caller — a test is the good outcome — or by making it `pub(crate)`, which is the other one.

    ./scripts/check_surface_has_callers.py
"""

from __future__ import annotations

import pathlib
import re
import sys
import tomllib

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from rust_surface import surface  # noqa: E402

ROOT = pathlib.Path(__file__).parent.parent
LEDGER = ROOT / "scripts" / "api_ledger.toml"

#: Public dsrust inventions with no in-repo caller. Measured 2026-08-25; may fall and never rise.
#:
#: `dsrust` only. The sibling crates — `gepa`, `tpe`, `pyrng` — are byte-compliant reproductions of
#: Python packages, so their public surface mirrors *those* libraries and having no dspy symbol
#: behind it is the expected state rather than a defect. Counting them put the first reading of this
#: at 107; 24 of those were theirs.
#:
#: 83 -> 80 when `optimize::minibatch` became `pub(crate)`: MIPROv2's schedule arithmetic, which
#: nothing outside `optimize/mipro/` calls and no caller of this crate would. Thirteen items left the
#: public surface and three of them were counted here. That is the shape of a reduction — find a
#: module whose reach is one directory, not an item at a time.
FLOOR = 45


def doc_examples() -> str:
    """Every fenced block inside a doc comment, which is a compiled caller like any other.

    A `/// ```…``` ` example is a doctest: it is built and run, it uses the item the way a caller
    would, and it is the demonstration a reader actually meets first. Missing them made this gate
    overcount by five — `Streamed`, `LmItem`, `track_usage`, `ReflectiveDataset` and
    `asks_with_a_prediction` each have one.
    """
    blocks: list[str] = []
    for path in (ROOT / "crates").rglob("*.rs"):
        inside = False
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if not stripped.startswith("///"):
                continue
            body = stripped.removeprefix("///").strip()
            if body.startswith("```"):
                inside = not inside
                continue
            if inside:
                blocks.append(body)
    return "\n".join(blocks)


def exercised_text() -> str:
    """Everywhere a caller of this crate would show up: its tests, its examples, its prose.

    "Its examples" includes the ones in doc comments, not only `examples/` — see [`doc_examples`].
    """
    places = [
        *(ROOT / "crates" / "dsrust" / "tests").rglob("*.rs"),
        *(ROOT / "crates").glob("*/examples/*.rs"),
        # What the derive *emits* is a caller in every crate that uses the macro, and the only place
        # it appears here is inside a `quote!`. `signature::json_field_reflection` is named nowhere
        # else in this repo, and making it `pub(crate)` would compile clean here and break every
        # user of `#[derive(Signature)]` with a JSON field.
        *(ROOT / "crates" / "dsrust-derive" / "src").rglob("*.rs"),
        # The PyO3 bridge is this crate's most exercised consumer — it drives upstream's own test
        # suite against the Rust API from outside. It was missing here, and two items were made
        # `pub(crate)` on the strength of that omission before the workspace build caught it.
        *(ROOT / "crates" / "dsrs-bridge" / "src").rglob("*.rs"),
        ROOT / "README.md",
        ROOT / "docs" / "usage.md",
        ROOT / "HANDOFF.md",
    ]
    written = "\n".join(path.read_text() for path in places if path.exists())
    return written + "\n" + doc_examples()


def undemonstrated() -> list[str]:
    """Public types and free items this crate invented that nothing outside their module names."""
    ledger = tomllib.loads(LEDGER.read_text())
    invented = set(ledger["rust_only"])
    promised = {
        (entry.get("rust") or "").split("::")[-1]
        for table in ("symbols", "methods", "constructors")
        for key, entry in ledger[table].items()
        if not key.startswith("dsrust::") and entry.get("rust")
    }
    text = exercised_text()

    leaves: dict[str, str] = {}
    for item in sorted({name for names in surface().values() for name in names}):
        parts = [part for part in item.split("::")[1:] if part]
        # A member is reachable exactly when its owner is, so the owner carries the question.
        if len(parts) >= 2 and parts[-2][:1].isupper():
            continue
        leaves.setdefault(parts[-1], item)

    return sorted(
        item
        for name, item in leaves.items()
        if item.startswith("dsrust::")
        and item in invented
        and name not in promised
        and not re.search(rf"\b{re.escape(name)}\b", text)
    )


def main() -> int:
    found = undemonstrated()
    print(f"public crate inventions with no in-repo caller: {len(found)} (floor {FLOOR})")
    if len(found) > FLOOR:
        print(f"\nSurface-callers gate FAILED: rose to {len(found)} from a floor of {FLOOR}.")
        print("  The whole set, not the additions — this counts rather than remembering:")
        for item in found:
            print(f"    · {item}")
        print(
            "\n  Give the new one a caller — a test is the good outcome — or make it `pub(crate)`,\n"
            "  which is the other one. A public item nothing exercises is a promise nobody checked."
        )
        return 1
    if len(found) < FLOOR:
        print(f"  down from {FLOOR}; lower FLOOR to {len(found)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
