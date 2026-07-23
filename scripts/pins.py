"""The library versions the fixtures are generated from, and the check that they were.

A fixture's `source` line is its provenance: it says which library produced the bytes a Rust test
is held to. Written as a literal it is only a claim, and a claim drifts — `gepa.json` carried
`gepa==0.0.27` while 0.1.1 was the installed package. So nothing here lets a generator name a
version it did not run: `require` reads the version out of the imported module, refuses to go on
when it is not the pinned one, and hands back the string it read for the stamp.

    from pins import require
    version = require("gepa")   # raises unless the installed gepa is the pin
"""

from __future__ import annotations

import pathlib

HERE = pathlib.Path(__file__).parent

#: The pinned version of each library a fixture is generated from. dspy's lives in its own file
#: because the upstream-test runner reads it from shell too.
PINS = {
    "dspy": (HERE / "DSPY_VERSION").read_text().strip(),
    # dspy 3.3.0b1 requires `gepa[dspy]==0.1.1`, so this pin follows dspy's.
    "gepa": "0.1.1",
}


def require(name: str) -> str:
    """The installed version of `name`, having checked it against the pin.

    Read from the installed distribution rather than a `__version__` attribute: that is what pip
    actually put there, and it exists for every package — `gepa` exposes no such attribute, which
    is how its stamp came to be a hand-written literal in the first place.
    """
    from importlib.metadata import PackageNotFoundError, version

    pinned = PINS[name]
    try:
        found = version(name)
    except PackageNotFoundError:
        raise SystemExit(f"{name} is not installed; expected {pinned}") from None
    if found != pinned:
        raise SystemExit(
            f"expected {name} {pinned}, found {found} — regenerating against another version "
            f"would stamp a provenance that is not true. Install the pin, or move it in "
            f"{__file__} and regenerate every fixture that names it."
        )
    return found
