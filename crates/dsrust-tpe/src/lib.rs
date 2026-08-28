//! A faithful Rust reproduction of the slice of [optuna](https://optuna.org) that dspy's MIPROv2
//! drives: a seeded `TPESampler` over categorical distributions.
//!
//! dspy keeps optuna as a separate optional dependency it wraps; this crate is the same seam in
//! Rust, so `dsrs`'s MIPROv2 can wrap it the way `mipro_optimizer_v2.py` wraps optuna. Faithfulness
//! is the point: seeded the same way, over the same categorical distributions and the same observed
//! scores, the sampler proposes the same trials optuna does — verified against optuna itself.
//!
//! The generator is [`pyrng::RandomState`], numpy's legacy MT19937; [`TpeSampler`] is built on it.

mod argsort;
mod int_sampler;
mod numerical;
mod parzen;
mod sampler;
pub mod truncnorm;

pub use int_sampler::IntTpeSampler;
pub use sampler::TpeSampler;
