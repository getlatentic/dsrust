//! The embeddable public surface, pinned so a root re-export cannot silently drop.
//!
//! A host that embeds DsRust (mapping its own graph to modules, optimizers, and the trace stream)
//! reaches everything from the crate root. `dsrust::GEPA` was missing from that root while every
//! other optimizer was present; this test names the whole surface so that gap cannot reopen.

#![allow(unused_imports)]

// Modules and how a program is asked.
use dsrust::{
    Ask, ChainOfThought, Forward, Module, NamedPredictor, Predict, Prediction, ReAct, ReActV2,
};
use dsrust::{Example, FnTool, History, ReasoningEffort, Signature, Steering, Tool};

// Every optimizer, all reachable from the root (this is the line GEPA was missing from).
use dsrust::{BootstrapFewShot, COPRO, DynOptimizer, GEPA, LabeledFewShot, MIPROv2, Optimizer};
use dsrust::{Feedback, GepaOutcome};

// Evaluation.
use dsrust::{Evaluate, Evaluation, Scored, exact_match};

// Provider config and keyless testing.
use dsrust::{Capabilities, DummyLM, configure_model};

// The trace stream and the saved-program state a host renders as a before/after diff.
use dsrust::{PredictorState, ProgramState, TraceStep};

/// The `use` statements above are the real assertion: if any symbol were not re-exported at the
/// crate root, this file would fail to compile. The body just keeps the intent legible.
#[test]
fn the_embeddable_surface_resolves_from_the_crate_root() {
    fn _a_program_is_a_module(_: &dyn Module) {} // Module is the object-safe seam
    fn _erased_optimizer(_: &dyn DynOptimizer) {} // DynOptimizer is the dyn form of Optimizer
    let _ = exact_match;
    let _: Option<TraceStep> = None;
    let _: Option<ProgramState> = None;
    let _: Option<Capabilities> = None;
}
