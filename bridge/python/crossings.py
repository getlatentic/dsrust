"""How far into the crate a test reached.

A test that passes without reaching it never exercised Rust, whatever its name says, so
`conftest.py` refuses to count it as conformance.

Two counts, because they answer different questions. `RENDERED` is the bytes a model would read:
a test that moves it asserts on something this crate wrote. `SIGNATURE` is the layer beneath,
where a field's prefix and a signature's shape are decided; it moves during the construction of
almost any signature, including in tests about something else entirely, so folding it into the
first would report coverage that no assertion backs.
"""

from __future__ import annotations

RENDERED = 0
SIGNATURE = 0


def record_render() -> None:
    global RENDERED
    RENDERED += 1


def record_signature() -> None:
    global SIGNATURE
    SIGNATURE += 1
