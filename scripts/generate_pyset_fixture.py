"""Record CPython's `set`-of-ints iteration order, which GEPA's merge selection draws against.

Merge picks candidates with `rng.sample(list(set(...)), 2)` and a common ancestor with
`rng.choices(list(set_a & set_b), weights=...)`, so *which* element the shared generator selects
depends on the order `list(set(...))` yields. That order matches `sorted` for indices 0-7 and
diverges above — reproducing it is the difference between a merge that conforms and one that only
looks like it does on a small run.

The cases sweep well past index 7, and past the 8->32 table resize, so a port that leaned on
`sorted` fails here rather than on some later real run. Intersections are included because merge's
common-ancestor set is one, and CPython builds it by iterating the smaller operand (the right one
on a size tie).

    .dspy-venv/bin/python scripts/generate_pyset_fixture.py
"""

from __future__ import annotations

import json
import pathlib
import sys

OUT = pathlib.Path(__file__).parent.parent / "crates" / "dsrust-gepa" / "tests" / "conformance"

# Deterministic without an RNG: a fixed spread of sequences chosen to cross the boundaries a
# faithful port has to clear — a table that stays at 8 slots, one that resizes to 32 and 128, keys
# in and out of the 0-7 range where set order stops matching sorted, and duplicate keys the table
# must fold. `range(...)` cases pin the resize points exactly.
BUILD_CASES = [
    [],
    [0],
    [3, 1, 2, 0],
    [7, 6, 5, 4, 3, 2, 1, 0],
    [8, 13, 5],
    [9, 6],
    [0, 1, 3, 8],
    [11, 4, 5],
    [6, 9, 14],
    [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
    list(range(20)),
    list(range(32)),
    list(range(50)),
    list(range(100)),
    [40, 8, 72, 8, 40, 1, 99, 50, 17, 17, 3],
    [i * 8 for i in range(12)],  # every key collides on the low 3 bits, forcing the probe chain
    [i * 7 for i in range(20)],
    [100, 50, 25, 12, 6, 3, 1, 0, 200, 150],
]

# (a, b) pairs for `a & b`, spanning: disjoint, nested, a tie (which operand is iterated matters),
# and both above the divergence boundary.
INTERSECTION_CASES = [
    ([1, 2, 3], [2, 3, 4]),
    ([10, 11, 12, 13], [12, 13, 14, 15]),
    ([5, 8, 13, 21], [8, 21, 34]),
    ([0, 1, 2, 3, 4, 5], [3, 4, 5]),
    ([9, 14, 6, 2], [14, 2, 40, 9]),  # equal size, but `a & b` and `b & a` come out the same
    # Equal size *and* discriminating: `a & b` is [143, 87] where `b & a` is [87, 143]. The tie rule
    # — which operand CPython iterates when the two are the same length — is unobservable on most
    # pairs, including the one above, because the result is re-inserted into a fresh table that
    # usually lands on the same order either way. Found by search after a mutation run showed the
    # `>` in the size comparison could be `<`, `==` or `>=` with every test still passing.
    ([87, 143, 102, 289], [143, 78, 87, 297]),
    ([143, 78, 87, 297], [87, 143, 102, 289]),
    # Unequal size, and it shows *which* operand is iterated: taking the smaller gives [183, 31],
    # taking the larger gives [31, 183]. Every other unequal case here comes out the same either
    # way, so the size comparison could have been reversed with the suite still green.
    ([183, 31, 12], [91, 31, 65, 203, 183, 78]),
    ([91, 31, 65, 203, 183, 78], [183, 31, 12]),
    (list(range(30)), list(range(15, 45))),
    ([i * 8 for i in range(10)], [i * 8 for i in range(5, 15)]),
]


def main() -> None:
    fixture = {
        "source": f"CPython {sys.version_info.major}.{sys.version_info.minor} "
        "via scripts/generate_pyset_fixture.py",
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "build": [
            {"input": case, "order": list(set(case))} for case in BUILD_CASES
        ],
        "intersection": [
            {"a": a, "b": b, "order": list(set(a) & set(b))}
            for a, b in INTERSECTION_CASES
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "pyset.json"
    path.write_text(json.dumps(fixture, indent=1) + "\n")
    print(
        f"  wrote {path.relative_to(OUT.parent.parent.parent)} "
        f"({len(fixture['build'])} builds, {len(fixture['intersection'])} intersections)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
