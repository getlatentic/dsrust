//! Building a dsrust program from a graph a user wired at run time.
//!
//! `#[derive(Module)]` writes a program from a Rust struct: its fields are the steps, and the
//! derive reads that field list to write the walk an optimizer works through. That list is fixed
//! when the crate compiles.
//!
//! A program assembled from a document — nodes and edges a user drew in an editor — has no such
//! list. The steps are a `Vec` whose length is known when the document loads. So the derive cannot
//! help, and `Module` is written by hand.
//!
//! **That is the one case where hand-writing `impl Module` is right rather than a mistake, and it
//! is also where the cost of hand-writing bites hardest.** `Module::named_predictors` defaults to
//! answering with nothing, so a graph module that does not write it is one an optimizer walks
//! empty: it rewrites nothing, changes nothing, and **returns `Ok`**. `tests/graph.rs` holds both
//! halves — that the graph runs, and that a compile actually changed it — and the second assertion
//! is on the demos rather than on `compile` succeeding, because succeeding proves nothing.
//!
//! **`impl Module` has two halves to write, not one.** `named_predictors` is the half this doc
//! used to name; the other is the observability point — `#[derive(Module)]` is dspy's
//! `Module.__call__` decorator, opening `on_module_start` before the body and closing
//! `on_module_end` after. A hand-written `forward` *is* that entry, and omitting it makes the
//! outermost program of a run silent while the `Predict`s inside it still report. This crate
//! shipped without it, and the app that took it as a reference inherited the hole.
//!
//! It began as a worked reference to copy from. It is now a dependency: Calibrate consumes it as a
//! path dep, having deleted its vendored copies after they needed hand-syncing twice. So an API
//! change here breaks a real consumer at compile time, which is the better arrangement — a
//! divergence that used to need noticing now needs fixing.

pub mod calibrate;
pub mod document;
pub mod graph;

pub use calibrate::CalibrateGraph;
pub use document::{Answer, Declared, Field, GraphDocument, Node, Source, Wire};
pub use graph::Graph;
