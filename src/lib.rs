//! A DSPy-style layer in Rust: declare a task as a [`struct@signature::Signature`], drive it
//! through a module in [`mod@predict`], and let the [`adapter`]s carry it over the wire to the
//! provider behind [`lm::LM`].

// `#[derive(Signature)]` expands to `::dsrs::...` paths; the alias keeps those
// paths valid when the derive is used inside this crate itself.
extern crate self as dsrs;

pub mod adapter;
pub mod evaluate;
pub mod example;
pub mod lm;
pub mod module;
pub mod optimize;
pub mod predict;
pub mod react;
pub mod signature;

pub use adapter::baml::BamlAdapter;
pub use adapter::{Adapter, ChatAdapter, Extraction, JsonAdapter, TwoStepAdapter};
pub use evaluate::{Evaluate, Evaluation, Scored, exact_match};
pub use example::{Example, Prediction};
pub use module::{Ask, Module, NamedPredictor, TraceStep};
pub use optimize::{BootstrapFewShot, DynOptimizer, LabeledFewShot, Optimizer};
pub use react::{FnTool, ReAct, Tool, Trajectory};

/// Re-exports the macros need so a caller does not have to depend on them directly.
#[doc(hidden)]
pub mod __macro_support {
    pub use serde_json::json;
}
pub use lm::dummy::DummyLM;
pub use lm::global::configure_model;
pub use lm::{
    ChatModel, ChatTurn, LM, ModelRef, OutputMode, Provider, Role, configure, configure_with_client,
};
pub use predict::{ChainOfThought, Predict, TypedChainOfThought, TypedPredict};
pub use signature::{
    FieldKind, InField, LiteralValue, OutField, Signature, SignatureSpec, chain_of_thought,
    json_field_schema, predict, signature,
};
