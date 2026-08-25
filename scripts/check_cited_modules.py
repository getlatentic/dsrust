"""A module this crate explains itself in terms of is ported in effect, and belongs on the list.

`check_coverage.py` guarantees every upstream *test file* is running or excused. Nothing made that
guarantee for *source* modules, and the cost is on the record twice. Its own docstring: "the usage
tracker was not in PORTED_MODULES at all despite the crate implementing it. Two live bugs were
sitting behind that miss." And an audit in August found nine more — including
`clients/openai_format.py`, the module the OpenAI body is byte-verified against, cited twenty-six
times and holding not one ledger entry.

The tell is cheap. This crate names upstream constantly to say what it is doing and what it is not,
so a backtick-quoted name whose only home is a module on no list is a module ported without anyone
saying so. A name on that list has its symbols held to `api_ledger.toml`; a name off it has nothing.

**An excuse table rather than a threshold**, exactly as `check_coverage.py` has and for the same
reason: a citation is prose. `download` and `set` are ordinary English that happen to name dspy
functions, and `DummyVectorizer` is cited *beside the sentence saying it is out of scope*. Those
must be excusable by name. A count with a floor would let the next real one hide behind a number.
"""

from __future__ import annotations

import collections
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from api_surface import DSPY, PORTED_MODULES  # noqa: E402

ROOT = pathlib.Path(__file__).parent.parent
RUST = ROOT / "crates" / "dsrust" / "src"

#: Names this crate cites that are *not* evidence of a ported module, and why.
#:
#: Two kinds. Ordinary English that happens to name a dspy function, and upstream machinery named
#: in the sentence explaining that it has no counterpart here — the second is the more interesting,
#: because citing something is exactly how a divergence gets written down.
EXCUSED: dict[str, str] = {
    # Read from the citation, not from what the name suggests. Six of these were first written from
    # a mental model of what the crate was probably saying, and six were wrong: `set` is the Python
    # *type* rather than `magicattr.set`, `delete` and `load` are intra-doc links to this crate's
    # own methods, `lookup` is a tool name inside a test assertion. A reason nobody checked is the
    # thing this whole gate exists to stop, so these were checked.
    "set": "`optimize/earned.rs` on `_input_keys` being a Python `set` — the built-in type, not `magicattr.set`",
    "delete": "an intra-doc link to `Signature::delete` in `signature.rs`; the dspy function of that name is `magicattr.delete`",
    "lookup": "a tool named `lookup` inside a ReAct prompt assertion in `react/v2.rs`",
    "load": "an intra-doc link to `Module::load` in `module.rs`. `saving.load` is separately classified",
    "download": "`adapters/types/image.py`'s refused mapping key, quoted in `image.rs` where 3.3.0's rule is reproduced — not `utils.download`",
    "BootstrapFinetune": "named in `better_together.rs` as the upstream default this crate does not have, because finetuning is out of 1.0 scope",
    "bootstrap_trace_data": "named in `optimize/gepa/metric.rs` describing how upstream calls a metric while scoring, beside `Evaluate` — it is the calling convention being explained, not the helper being ported",
}

#: Modules known to be ported and not yet listed, each with the story that will list them.
#:
#: A holding pen, not an excuse: every name here is a module whose symbols nobody has classified,
#: and the entry says so. It exists because listing a module means classifying everything it
#: declares, which is a day's work per module and not a reason to leave the check unbuilt.
PENDING: dict[str, str] = {}
"""Modules known to be ported and not yet listed, each with the story that would list them.

Empty, and worth keeping so it can be filled again rather than argued about. It held six between
the gate landing and `unlisted-2` closing them: listing a module means classifying everything it
declares, which is real work and not a reason to leave the check unbuilt in the meantime.
"""



def defined_by() -> dict[str, set[str]]:
    """Every public name dspy defines, and the modules that define it."""
    homes: dict[str, set[str]] = {}
    for path in DSPY.rglob("*.py"):
        rel = str(path.relative_to(DSPY))
        for name in re.findall(r"^(?:class|def)\s+(\w+)", path.read_text(), re.M):
            if not name.startswith("_"):
                homes.setdefault(name, set()).add(rel)
    return homes


def citations() -> tuple[collections.Counter[tuple[str, tuple[str, ...]]], set[str]]:
    """Backtick-quoted names whose every home is unlisted, and which excuses were used.

    The second half matters as much as the first. An excuse for a name nothing cites any more is a
    declaration that stopped being true — the same shape as a `[rust_only]` entry naming an item
    the crate no longer exposes, which its own gate reports. Two were already dead when this check
    was written: `DummyVectorizer` and `dummy_rm`, excused for citations that were never there.
    """
    homes = defined_by()
    found: collections.Counter[tuple[str, tuple[str, ...]]] = collections.Counter()
    used: set[str] = set()
    for path in RUST.rglob("*.rs"):
        for name in re.findall(r"`(\w+)`", path.read_text()):
            if name in EXCUSED:
                used.add(name)
                continue
            where = homes.get(name)
            if where and not any(home in PORTED_MODULES for home in where):
                found[(name, tuple(sorted(where)))] += 1
    return found, used


def main() -> int:
    found, used = citations()
    dead = sorted(set(EXCUSED) - used)
    unexcused = [
        (name, homes, count)
        for (name, homes), count in found.items()
        if not all(home in PENDING for home in homes)
    ]
    waiting = {home for (_, homes), _ in found.items() for home in homes if home in PENDING}

    print(f"cited-module gate: {len(found)} name(s) cited from unlisted modules")
    print(f"  waiting on a story : {len(waiting)} module(s)")
    for home in sorted(waiting):
        print(f"      · {home} — {PENDING[home]}")

    if dead:
        print(f"\nCited-module gate FAILED: {len(dead)} excuse(s) for a name nothing cites:")
        for name in dead:
            print(f"    {name} — {EXCUSED[name]}")
        print("\n  The citation went away. Drop the excuse rather than leaving it to be trusted.")
        return 1

    if unexcused:
        print(f"\nCited-module gate FAILED: {len(unexcused)} name(s) with no answer:")
        for name, homes, count in sorted(unexcused, key=lambda row: -row[2]):
            print(f"    {name} ({count}x) — defined in {', '.join(homes)}")
        print(
            "\n  Either the module is ported and belongs in PORTED_MODULES with its symbols\n"
            "  classified, or the citation is prose and belongs in EXCUSED with the reason."
        )
        return 1

    print("\nCited-module gate: OK (every cited module is listed, excused, or filed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
