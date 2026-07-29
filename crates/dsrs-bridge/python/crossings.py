"""How far into the crate a test reached.

A test that passes without reaching it never exercised Rust, whatever its name says, so
`conftest.py` refuses to count it as conformance.

Two counts, because they answer different questions. `RENDERED` is the bytes a model would read:
a test that moves it asserts on something this crate wrote. `SIGNATURE` is the layer beneath,
where a field's prefix and a signature's shape are decided; it moves during the construction of
almost any signature, including in tests about something else entirely, so folding it into the
first would report coverage that no assertion backs.

**Attributed, not sampled.** These were bare globals read before and after each test, so a
crossing made on a *background* thread — diskcache's fanout writers outliving the test that
started them — was credited to whichever test happened to be running when it landed. That made the
guard flaky in both directions, and the second direction is the dangerous one: a
declared-non-crossing test failing for a crossing it did not make is noisy and obvious, while a
real crossing landing on a neighbour leaves the test that earned it reading as dead.

The rule is the test's own thread, plus any thread that did not exist when it started. A test that
fans work out — `dspy.Parallel`, an async run, `asyncify` — crosses on threads it created, and
those count. A thread already alive at that moment belongs to something else, and does not.
"""

from __future__ import annotations

import threading

RENDERED = 0
SIGNATURE = 0

#: The test currently running, the thread it runs on, and the threads that already existed when
#: it began — set by `_require_a_crossing`.
_CURRENT: tuple[str, int, frozenset[int]] | None = None

#: Per-test counts, credited only for calls made on that test's own thread.
_BY_TEST: dict[str, list[int]] = {}


def begin(nodeid: str) -> None:
    """Start attributing to this test, on the thread that calls this."""
    global _CURRENT
    already = frozenset(thread.ident for thread in threading.enumerate() if thread.ident)
    _CURRENT = (nodeid, threading.get_ident(), already - {threading.get_ident()})
    _BY_TEST[nodeid] = [0, 0]


def end() -> tuple[int, int]:
    """Stop attributing, and hand back what the test itself reached: (rendered, signature)."""
    global _CURRENT
    if _CURRENT is None:
        return (0, 0)
    nodeid = _CURRENT[0]
    _CURRENT = None
    rendered, signature = _BY_TEST.pop(nodeid, [0, 0])
    return (rendered, signature)


def _credit(slot: int) -> None:
    if _CURRENT is None:
        return
    nodeid, own, inherited = _CURRENT
    here = threading.get_ident()
    # The test's own thread, or one it started; never one that was already running.
    if here != own and here in inherited:
        return
    counts = _BY_TEST.get(nodeid)
    if counts is not None:
        counts[slot] += 1


def record_render() -> None:
    global RENDERED
    RENDERED += 1
    _credit(0)


def record_signature() -> None:
    global SIGNATURE
    SIGNATURE += 1
    _credit(1)
