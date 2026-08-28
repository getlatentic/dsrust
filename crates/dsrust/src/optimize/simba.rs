//! SIMBA — stochastic introspective mini-batch ascent.
//!
//! Being ported. What is here is the arithmetic layer the search sits on, held to numpy and
//! CPython directly: see [`arithmetic`].

pub mod arithmetic;
mod compile;
pub mod feedback;
pub mod search;
mod strategies;
