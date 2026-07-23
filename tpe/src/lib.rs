//! A faithful Rust reproduction of the slice of [optuna](https://optuna.org) that dspy's MIPROv2
//! drives: a seeded `TPESampler` over categorical distributions.
//!
//! dspy keeps optuna as a separate optional dependency it wraps; this crate is the same seam in
//! Rust, so `dsrs`'s MIPROv2 can wrap it the way `mipro_optimizer_v2.py` wraps optuna. Faithfulness
//! is the point: seeded the same way, over the same categorical distributions and the same observed
//! scores, the sampler proposes the same trials optuna does — verified against optuna itself.
//!
//! The foundation is [`mt19937`], numpy's `RandomState`; [`TpeSampler`] is built on top of it.

pub mod mt19937;
mod parzen;
mod sampler;

pub use mt19937::Mt19937;
pub use sampler::TpeSampler;
