"""Record what `soundfile` writes for `Audio.from_array`, so the Rust WAV writer is held to it.

dspy hands a numpy array to `soundfile.write(..., format="WAV", subtype="PCM_16")` and base64s the
result. That is a byte a prompt carries, so this crate reproduces it from primitives rather than
binding a codec — libsndfile's default WAV is a canonical 44-byte RIFF header and nothing else, with
no vendor chunks and no padding, which is the reason reproducing it is reasonable at all.

Reproducing it is only reasonable if it is *measured*, hence this. Two things get pinned: the whole
file for a small array, and the sample conversion at the edges — `1.0` must land on 32767 rather
than wrapping to -32768, which is exactly what a naive cast gets wrong.

`soundfile` and `numpy` are declared in pyproject for this script. Upstream treats both as optional
(`SF_AVAILABLE`), and no upstream test in the harness needs them; they are here so this golden can
be regenerated rather than trusted.

    .venv/bin/python scripts/generate_wav_fixture.py
"""

from __future__ import annotations

import base64
import io
import json
import platform
import pathlib

import numpy as np
import soundfile as sf

ROOT = pathlib.Path(__file__).parent.parent
OUT = ROOT / "crates/dsrust/tests/conformance/constants/wav_pcm16.json"

#: Each case is (name, samples, sampling rate). The edges matter more than the middle: `1.0` and
#: `-1.0` are where a scale-and-cast disagrees with libsndfile, and an empty array is where a header
#: with no payload has to still be a valid file.
CASES = [
    ("a_few_samples", [0.0, 0.5, -0.5, 1.0], 8000),
    ("full_scale_edges", [1.0, -1.0, 0.999969482421875, -0.99998], 44100),
    ("empty", [], 16000),
    ("one_sample_at_a_high_rate", [0.25], 48000),
    # The sweep that settled the conversion rule. libsndfile floors and clamps; truncation and
    # round-to-nearest each disagree on three of these, so the rule is derived here rather than
    # assumed, and stays derived if libsndfile ever changes it.
    (
        "conversion_sweep",
        [
            -1.0, -0.99998, -0.75, -0.5000153, -0.5, -0.25, -1e-5,
            0.0,
            1e-5, 0.25, 0.5, 0.5000153, 0.75, 0.999969482421875, 0.99998, 1.0,
        ],
        8000,
    ),
]


def main() -> int:
    cases = {}
    for name, samples, rate in CASES:
        buffer = io.BytesIO()
        sf.write(
            buffer,
            np.array(samples, dtype="float32"),
            rate,
            format="WAV",
            subtype="PCM_16",
        )
        raw = buffer.getvalue()
        cases[name] = {
            "samples": samples,
            "sampling_rate": rate,
            "bytes": len(raw),
            "base64": base64.b64encode(raw).decode(),
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "source": f"CPython {platform.python_version()} wave + soundfile, via scripts/generate_wav_fixture.py",
                "python_version": platform.python_version(),
                "note": (
                    "soundfile.write(..., format='WAV', subtype='PCM_16'), which is what "
                    "dspy's Audio.from_array emits. libsndfile writes a canonical 44-byte RIFF "
                    "header and no vendor chunks, which is why this is reproducible."
                ),
                "libsndfile": sf.__libsndfile_version__,
                "cases": cases,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"  {len(cases)} cases -> {OUT.relative_to(ROOT)}  (libsndfile {sf.__libsndfile_version__})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
