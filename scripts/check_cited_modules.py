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
    "set": "`magicattr.set`; the word appears in prose about setting anything",
    "delete": "`magicattr.delete`; likewise",
    "lookup": "`magicattr.lookup`; likewise",
    "download": "`utils.download`; the word appears in prose about fetching a model or a file",
    "load": "`saving.load` is classified; the bare word also appears in prose about loading anything",
    "DummyVectorizer": "cited in the sentence saying embeddings are out of scope",
    "dummy_rm": "cited in the sentence saying retrieval is out of scope",
    "BootstrapFinetune": "cited where finetuning is deferred past 1.0",
    "bootstrap_trace_data": "cited where the finetuning data helpers are deferred",
}

#: Modules known to be ported and not yet listed, each with the story that will list them.
#:
#: A holding pen, not an excuse: every name here is a module whose symbols nobody has classified,
#: and the entry says so. It exists because listing a module means classifying everything it
#: declares, which is a day's work per module and not a reason to leave the check unbuilt.
PENDING: dict[str, str] = {
    "propose/grounded_proposer.py": "MIPROv2's proposer — `optimize/mipro/grounded.rs` (#unlisted-2)",
    "propose/dataset_summary_generator.py": "the dataset descriptor — `optimize/mipro/dataset_summary.rs`, which has its own golden (#unlisted-2)",
    "utils/usage_tracker.py": "usage tracking — `lm/usage.rs`; its *tests* were added when this gap was first found and the module never was (#unlisted-2)",
    "utils/parallelizer.py": "`ParallelExecutor` — `predict/parallel.rs` (#unlisted-2)",
    "teleprompt/utils.py": "the optimizer helpers MIPROv2 uses — minibatch evaluation and demo-set building, both ported (#unlisted-2)",
    "propose/utils.py": "`strip_prefix` and `create_example_string`, both ported into `optimize/mipro/` and both cited by the functions that reproduce them (#unlisted-2)",
}


def defined_by() -> dict[str, set[str]]:
    """Every public name dspy defines, and the modules that define it."""
    homes: dict[str, set[str]] = {}
    for path in DSPY.rglob("*.py"):
        rel = str(path.relative_to(DSPY))
        for name in re.findall(r"^(?:class|def)\s+(\w+)", path.read_text(), re.M):
            if not name.startswith("_"):
                homes.setdefault(name, set()).add(rel)
    return homes


def citations() -> collections.Counter[tuple[str, tuple[str, ...]]]:
    """Backtick-quoted names in the Rust source whose every home is unlisted."""
    homes = defined_by()
    found: collections.Counter[tuple[str, tuple[str, ...]]] = collections.Counter()
    for path in RUST.rglob("*.rs"):
        for name in re.findall(r"`(\w+)`", path.read_text()):
            if name in EXCUSED:
                continue
            where = homes.get(name)
            if where and not any(home in PORTED_MODULES for home in where):
                found[(name, tuple(sorted(where)))] += 1
    return found


def main() -> int:
    found = citations()
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
