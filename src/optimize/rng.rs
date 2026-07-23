//! CPython's `random.Random`, reused from the [`pyrng`] crate.
//!
//! Which examples an optimizer keeps *is* its output, and here that output is whatever this
//! generator draws — the reproduction and its conformance against CPython live in [`pyrng::cpython`],
//! shared with the `tpe` and (later) `gepa` crates rather than reimplemented per consumer.

pub(super) use pyrng::Random as Rng;
