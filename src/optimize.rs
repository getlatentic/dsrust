//! Compilers: they read a program and write back a better one.
//!
//! This is the layer DSPy is named for. A signature says what the task is, a module says how to
//! ask, and an optimizer decides what the prompt should actually contain — by choosing demos,
//! or rewriting instructions — measured against a metric rather than guessed at.
//!
//! Every optimizer here works through [`Module::named_predictors`](crate::Module), the same seam
//! dspy's do: walk the program, read each predictor, write improved demos back. That is why
//! `Predict` implementing `Module` mattered — without it there is nothing for a compiler to
//! reach into.
//!
//! The split mirrors dspy's own: `vanilla.py` holds the labelled baseline, `bootstrap.py` holds
//! the optimizer that runs a program to earn its demos and imports the baseline to prime its
//! teacher.

mod bootstrap;
mod labeled;
mod rng;

#[cfg(test)]
mod conformance;
#[cfg(test)]
mod scripted;

pub use bootstrap::BootstrapFewShot;
pub use labeled::LabeledFewShot;
