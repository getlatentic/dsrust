//! Byte-faithful Rust reproductions of Python's Mersenne-Twister random number generators.
//!
//! CPython's `random.Random` and numpy's legacy `RandomState` are the two Python RNGs a faithful
//! DSPy port has to reproduce: an optimizer's demo selection *is* whatever CPython's generator draws
//! ([`cpython`]), and optuna's TPE sampler draws through numpy's ([`numpy`]). Both are the same
//! MT19937 — the shared [`mt19937::Mt19937`] core — differing only in seeding and what they read
//! out, so they live in one crate rather than being reimplemented per consumer.

pub mod cpython;
pub mod mt19937;
pub mod numpy;
pub mod pcg64;

pub use cpython::Random;
pub use mt19937::Mt19937;
pub use numpy::RandomState;
