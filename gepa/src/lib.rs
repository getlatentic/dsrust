//! A Rust reproduction of GEPA's reflective prompt-evolution engine — the `gepa` package dspy wraps.
//!
//! The engine is LLM-agnostic (it drives evaluation and reflection through an adapter); this crate
//! reproduces its numerical and structural core, over `pyrng`'s CPython RNG and held to the real
//! `gepa` package. [`engine::GepaEngine`] runs the reflective-mutation loop; [`pareto`] selects the
//! candidate to evolve each iteration and [`batch`] samples its minibatch, both off one shared RNG.

pub mod adapter;
pub mod batch;
pub mod engine;
pub mod instruction_proposal;
pub mod pareto;
pub mod pyset;
pub mod state;

pub use adapter::{Candidate, EvalBatch, GepaAdapter};
pub use engine::{GepaEngine, GepaOutcome};
pub use instruction_proposal::{Reflective, ReflectiveSample, extract_new_instruction, render_prompt};
