"""Public dspy API that cannot be constructed, recorded by trying to construct it.

Three names dspy 3.3.0b1 still ships raise before doing anything, and two of them are exported:
`dspy.teleprompt.__all__` lists `AvatarOptimizer`, whose `__init__` reaches `dspy.TypedPredictor`,
removed in 2.6. `Avatar` — the module `AvatarOptimizer` exists to optimize — reaches it too.
`SignatureOptimizer` breaks differently: `COPRO.__init__` dropped its `verbose` parameter and the
subclass still forwards seven positional arguments to a constructor taking six.

This matters to a port because a module that cannot run has no behaviour to be faithful to. Writing
Rust for it would mean inventing the semantics of a removed class and then testing the invention
against itself, so these stay unported — and `unported_modules.toml` cites this file for the claim
rather than asserting it in prose. Recorded by *calling* each entry point, so if upstream repairs
one the gate turns red and says so.

    .venv/bin/python scripts/generate_upstream_dead_api_fixture.py
"""

from __future__ import annotations

import json
import logging
import pathlib
import warnings

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

from pins import require

PINNED = require("dspy")
OUT = (
    pathlib.Path(__file__).parent.parent
    / "crates"
    / "dsrust"
    / "tests"
    / "conformance"
    / "upstream_dead_api.json"
)


def _avatar():
    from dspy.predict.avatar import Avatar
    from dspy.predict.avatar.models import Tool

    return Avatar("question -> answer", tools=[Tool(tool=None, name="Search", desc="a tool")])


def _avatar_optimizer():
    from dspy.teleprompt import AvatarOptimizer

    return AvatarOptimizer(metric=lambda example, pred, trace=None: 1.0)


def _signature_optimizer():
    from dspy.teleprompt.signature_opt import SignatureOptimizer

    return SignatureOptimizer(metric=lambda example, pred, trace=None: 1.0)


#: Every entry point named by a module `unported_modules.toml` marks `upstream_dead`, with the call
#: a caller would actually make. `exported` records whether the name is in a package `__all__`,
#: which is the difference between shipped-and-broken and merely reachable.
ENTRY_POINTS = [
    ("predict/avatar/avatar.py", "Avatar", False, _avatar),
    ("teleprompt/avatar_optimizer.py", "AvatarOptimizer", True, _avatar_optimizer),
    ("teleprompt/signature_opt.py", "SignatureOptimizer", False, _signature_optimizer),
]


def _construct(build) -> dict:
    try:
        build()
    except Exception as e:  # noqa: BLE001 — the exception *is* the measurement
        return {"raises": type(e).__name__, "message": str(e)}
    return {"raises": None, "message": None}


def main() -> None:
    import dspy.teleprompt as tp

    entries = []
    for module, name, expected_export, build in ENTRY_POINTS:
        outcome = _construct(build)
        entries.append(
            {
                "module": module,
                "name": name,
                "exported_from_dspy_teleprompt": name in getattr(tp, "__all__", []),
                "expected_exported": expected_export,
                **outcome,
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "source": f"generated from dspy=={PINNED} via scripts/generate_upstream_dead_api_fixture.py",
                "note": (
                    "Each entry was constructed, not read. A null `raises` means upstream repaired "
                    "the name and the module is a port backlog item again."
                ),
                "entries": entries,
            },
            indent=2,
        )
        + "\n"
    )
    for e in entries:
        print(f"  {e['name']:20s} {e['raises'] or 'CONSTRUCTS'}: {e['message']}")
    print(f"wrote {OUT.relative_to(pathlib.Path(__file__).parent.parent)}")


if __name__ == "__main__":
    main()
