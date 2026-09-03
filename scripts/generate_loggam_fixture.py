"""CPython's `math.lgamma` as a second oracle for numpy's `loggam`.

`pyrng` reproduces numpy's `random_loggam` — its own Stirling series, transcribed rather than taken
from a crate, because PTRS compares against it and the last digit decides a rejection. The poisson
streams hold it to numpy exactly, which is the fidelity claim and is the one that matters.

They do not, however, hold its *coefficients* to much. `loggam` reaches the acceptance test through
an inequality whose two sides are usually far apart: the smallest margin over 66,427 comparisons
swept for this is `4.3e-06`, so a coefficient contributing less than that moves no draw. Only the
second coefficient does — `1.6e-05`, and the corpus now contains the one stream in 350,000 where it
decides — while the fourth contributes `1.4e-09` and the sixth `1.9e-12`. Reaching those through
poisson would take a corpus of some `10^8` draws.

Comparing against `lgamma` instead asks a different and much sharper question: is this ln Γ at all?
Over `x = 1..400` numpy's series and CPython's implementation agree to a relative `8.9e-16` — four
ULPs, two independent implementations — so a bound of `1e-13` has a hundredfold headroom and still
fails if the fourth or sixth coefficient is wrong. It is deliberately not tighter: the two differ by
a few ULPs already, and `ln` is a libm call whose last bit is not portable.

    .venv/bin/python scripts/generate_loggam_fixture.py
"""

from __future__ import annotations

import json
import math
import pathlib
import platform

OUT = (
    pathlib.Path(__file__).parent.parent
    / "crates" / "pyrng" / "tests" / "conformance" / "loggam.json"
)

#: `loggam` is only ever called as `loggam(k + 1)` for an integer `k`, so integers are the whole
#: reachable domain. Everything at or below seven takes the shift-and-recur branch, and the small
#: values are where a wrong coefficient shows most, `x0` bottoming out at seven.
XS = [float(x) for x in range(1, 101)] + [float(x) for x in range(150, 2001, 50)]

#: Measured, not chosen: the largest relative gap between the two implementations over `XS` is
#: `8.9e-16`. This is that with room for a different libm, and it is still four orders below the
#: `1.4e-09` that the fourth coefficient is worth.
TOLERANCE = 1e-13


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "source": (
                    f"CPython {platform.python_version()} math.lgamma, via "
                    "scripts/generate_loggam_fixture.py"
                ),
                "python_version": platform.python_version(),
                "note": (
                    "A second implementation of ln gamma, independent of numpy's series. Holds "
                    "loggam's coefficients to a bound the poisson streams cannot reach."
                ),
                "relative_tolerance": TOLERANCE,
                "values": [{"x": x, "lgamma": math.lgamma(x)} for x in XS],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"  wrote {OUT.name}: {len(XS)} values, relative tolerance {TOLERANCE:g}")


if __name__ == "__main__":
    main()
