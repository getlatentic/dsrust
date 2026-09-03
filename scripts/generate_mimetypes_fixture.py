"""Record what CPython's `mimetypes` calls each suffix this port meets.

`Image.from_path`, `Audio.from_path` and `File.from_path` all pick a media type off the file's
suffix, and that type reaches the model — inside a `data:` URI, or as an `input_audio` block's
`format`. So the mapping is a byte a prompt carries and belongs in a golden rather than in a
transcription.

**The control, and the reason it exists.** `mimetypes.guess_type` is *not* deterministic across
machines: `mimetypes.init()` reads system files — `/etc/mime.types` and friends — and merges them
over the module's built-in table. On the machine this was first run on, six of the suffixes below
came back different from the built-in answer (`.ogg`, `.flac`, `.m4a` and `.opus` are absent from
the built-in map or spelled differently there; `.aac` and `.xml` disagree outright).

Recording `guess_type`'s answer would therefore bake one machine's `/etc/mime.types` into a golden
every other machine would fail. So this records the **built-in** answer — `MimeTypes(filenames=())`,
which is the table CPython ships — and *asserts* it, so a suffix whose two answers disagree is
written down as disagreeing rather than silently taking one side.

A caller on a machine whose system files say otherwise will see dspy answer differently from this
crate for those suffixes. That is upstream's non-determinism rather than a divergence to close, and
it is recorded in the golden so nobody has to rediscover it.

    .venv/bin/python scripts/generate_mimetypes_fixture.py
"""

from __future__ import annotations

import json
import platform
import mimetypes
import pathlib

ROOT = pathlib.Path(__file__).parent.parent
OUT = ROOT / "crates/dsrust/tests/conformance/constants/mimetypes.json"

#: Every suffix the port's own factories and readers may meet: image and audio for the media types,
#: the document set `legacy.rs` already carried, and the handful a `File` realistically holds.
SUFFIXES = [
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".svg",
    ".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac", ".opus",
    ".webm", ".mp4",
    ".pdf", ".txt", ".html", ".htm", ".json", ".csv", ".md", ".xml",
    ".zip", ".bin", ".wasm",
]


def main() -> int:
    mimetypes.init()
    builtin = mimetypes.MimeTypes(filenames=(), strict=True)

    entries = {}
    machine_specific = {}
    for suffix in SUFFIXES:
        shipped = builtin.guess_type("x" + suffix)[0]
        here = mimetypes.guess_type("x" + suffix)[0]
        if shipped is not None:
            entries[suffix] = shipped
        if here != shipped:
            machine_specific[suffix] = {"builtin": shipped, "this_machine": here}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "source": f"CPython {platform.python_version()} mimetypes, via scripts/generate_mimetypes_fixture.py",
                "python_version": platform.python_version(),
                "note": (
                    "CPython's built-in mimetypes table. `mimetypes.guess_type` merges system "
                    "files over this, so the entries under `machine_specific` are ones where "
                    "dspy's answer depends on the host."
                ),
                "types": entries,
                "machine_specific": machine_specific,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"  {len(entries)} suffixes -> {OUT.relative_to(ROOT)}")
    if machine_specific:
        print(f"  {len(machine_specific)} answer differently on this machine than in CPython's table:")
        for suffix, both in sorted(machine_specific.items()):
            print(f"    {suffix:8} builtin={both['builtin']}  here={both['this_machine']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
