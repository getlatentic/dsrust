"""Which annotations accept a bare, non-JSON string, asked of pydantic rather than of a type list.

An adapter reading a reply gets text. For most annotations that text must be JSON — a `list[str]`
answered as `hello` is malformed. For some it *is* the value: a `datetime` is `2024-01-01T12:00:00`
and a `dspy.Code` is the code. Casting the second kind as JSON rejects a reply dspy accepts, and the
crate's list of them was one short — `dspy.Code`, which took two upstream tests red for as long as
nobody ran that suite.

So the set is measured. Each candidate annotation is validated twice: once with a string that is its
own form, and once with a string that is nothing's form. A type that takes the second takes anything,
and it is the *anything* column that decides whether a bare string may be left alone — a `datetime`
that rejects `print('hi')` still needs its own well-formed string, and the crate's rule for it is
"leave it for the caller's typing", which only holds because the typing will refuse a bad one.

    .venv/bin/python scripts/generate_string_form_fixture.py
"""

from __future__ import annotations

import datetime
import json
import logging
import pathlib
import sys
import warnings

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

import dspy
import pydantic
from dspy.adapters.types.base_type import Type
from dspy.adapters.utils import get_annotation_name

from pins import require

OUT = (
    pathlib.Path(__file__).parent.parent / "crates" / "dsrust" / "tests" / "conformance" / "adapter"
)
PINNED = require("dspy")

#: A string that is nothing's own form. Not JSON, not a date, not a URL — so a type that validates
#: it validates any string, which is the property the crate's rule turns on.
ARBITRARY = "print('hi')"

#: A string that *is* each temporal type's own form, to tell "accepts anything" apart from "accepts
#: its own spelling". Per type rather than one probe for all four: `date` rejects a datetime's
#: spelling and `time` rejects a date's, so a single probe would report three of them as accepting
#: no string at all and would argue for dropping them from a list they belong in.
WELL_FORMED = {
    "datetime": "2024-01-01T12:00:00",
    "date": "2024-01-01",
    "time": "12:00:00",
    "timedelta": "P1DT2H",
}

#: What a type with no spelling of its own is probed with — it should reject this, which is what
#: puts it in neither column.
DEFAULT_WELL_FORMED = "2024-01-01T12:00:00"


def candidates() -> dict[str, type]:
    """Every `dspy.Type` subclass dspy exports, plus the temporal types a signature can name."""
    found: dict[str, type] = {
        "datetime": datetime.datetime,
        "date": datetime.date,
        "time": datetime.time,
        "timedelta": datetime.timedelta,
    }
    for name in dir(dspy):
        obj = getattr(dspy, name)
        if isinstance(obj, type) and issubclass(obj, Type) and obj is not Type:
            found[name] = obj
    # The subscripted spelling builds a class of its own, and its annotation name differs.
    found["Code[java]"] = dspy.Code["java"]
    return found


def accepts(annotation: type, probe: str) -> bool:
    try:
        pydantic.TypeAdapter(annotation).validate_python(probe)
        return True
    except Exception:
        return False


def main() -> None:
    rows = []
    for name, annotation in sorted(candidates().items()):
        rows.append(
            {
                "type": name,
                "annotation_name": get_annotation_name(annotation),
                "accepts_arbitrary": accepts(annotation, ARBITRARY),
                "well_formed_probe": WELL_FORMED.get(name, DEFAULT_WELL_FORMED),
                "accepts_well_formed": accepts(annotation, WELL_FORMED.get(name, DEFAULT_WELL_FORMED)),
            }
        )

    fixture = {
        "source": f"generated from dspy=={PINNED} via scripts/generate_string_form_fixture.py",
        "dspy_version": PINNED,
        "note": (
            "Whether pydantic validates a bare string for each annotation a signature can name. "
            "`annotation_name` is what get_annotation_name gives, which is the string the crate "
            "matches on."
        ),
        "arbitrary_probe": ARBITRARY,
        "well_formed_probes": WELL_FORMED,
        "rows": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "string_form.json"
    path.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"  wrote {path.relative_to(OUT.parent.parent.parent)}", file=sys.stderr)

    anything = [r["annotation_name"] for r in rows if r["accepts_arbitrary"]]
    dated = [r["annotation_name"] for r in rows if r["accepts_well_formed"] and not r["accepts_arbitrary"]]
    print(f"    accepts any string: {sorted(set(anything))}", file=sys.stderr)
    print(f"    accepts its own form only: {sorted(set(dated))}", file=sys.stderr)

    # Both columns must have members or the corpus cannot tell the two rules apart, and a corpus
    # where everything accepts everything would pass against a crate that skipped casting entirely.
    if not anything:
        raise SystemExit("nothing accepts an arbitrary string — the probe is wrong")
    if not dated:
        raise SystemExit("no type accepts only its own form — the two columns are not distinct")
    if any(not r["accepts_arbitrary"] and not r["accepts_well_formed"] for r in rows) is False:
        raise SystemExit("every type accepts a string — nothing exercises the JSON path")


if __name__ == "__main__":
    main()
