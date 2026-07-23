//! A Rust reproduction of GEPA's reflective prompt-evolution engine — the `gepa` package dspy wraps.
//!
//! The engine is LLM-agnostic (it drives evaluation and reflection through an adapter); this crate
//! reproduces its numerical and structural core, over `pyrng`'s CPython RNG and held to the real
//! `gepa` package. It begins with Pareto-front candidate selection, the choice the evolution loop
//! makes each iteration.

pub mod batch;
pub mod pareto;
