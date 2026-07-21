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


def split_example(example):
    """dspy `Example.inputs()`, with the split decided by the crate.

    Which fields are inputs and which are labels is the record's one real decision, and it is
    the crate's. Python keeps the values and rebuilds the record around the answer, the same
    division `reflect.py` draws for adapters.
    """
    crossings.record_signature()
    names = list(example._store)
    declared = None if example._input_keys is None else list(example._input_keys)
    return dsrs_bridge.split_example(names, declared)


def inputs(self):
    """Upstream's `Example.inputs`, answering from the crate's split."""
    names, _ = split_example(self)
    kept = type(self)(base={name: self._store[name] for name in names})
    kept._input_keys = self._input_keys
    return kept


def labels(self):
    """Upstream's `Example.labels`, which drops the declaration rather than carrying it."""
    _, names = split_example(self)
    return type(self)(base={name: self._store[name] for name in names})
