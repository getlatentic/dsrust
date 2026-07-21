"""dspy's signature layer, decided by this crate, for running upstream's own tests.

`signatures/test_signature.py` asserts on behaviour no adapter renders, so running it through
the bridge would say nothing about this port until the crate owns the decisions it makes. What
crosses here is the part that is pure logic over names and structure. What stays in Python is
the part that is Python: resolving a type annotation needs the interpreter's own type system,
and `reflect.py` already draws that line for adapters.
"""

from __future__ import annotations

import dsrs_bridge

import crossings


def infer_prefix(attribute_name: str) -> str:
    """dspy's `infer_prefix`, decided in Rust.

    A field name becomes the label printed in front of it. Upstream reaches the answer through
    four regular expressions whose character classes disagree about ASCII, which is the sort of
    detail a port gets wrong quietly, so the whole decision crosses rather than its result.
    """
    crossings.record_signature()
    return dsrs_bridge.infer_prefix(attribute_name)
