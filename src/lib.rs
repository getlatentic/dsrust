//! A DSPy-style layer in Rust: declare a task as a [`signature::Signature`], drive it
//! through a module in [`predict`], and let the [`adapter`]s carry it over the wire to the
//! provider behind [`lm::LM`].

// `#[derive(Signature)]` expands to `::dsrs::...` paths; the alias keeps those
// paths valid when the derive is used inside this crate itself.
extern crate self as dsrs;

pub mod adapter;
pub mod evaluate;
pub mod example;
pub mod lm;
pub mod predict;
pub mod signature;

pub use adapter::{Adapter, ChatAdapter, JsonAdapter};
pub use evaluate::{Evaluate, Evaluation, Scored, exact_match};
pub use example::{Example, Prediction};

/// Re-exports the macros need so a caller does not have to depend on them directly.
#[doc(hidden)]
pub mod __macro_support {
    pub use serde_json::json;
}
pub use lm::{
    ChatModel, ChatTurn, LM, ModelRef, OutputMode, Provider, Role, configure, configure_with_client,
};
pub use predict::{ChainOfThought, Predict, TypedChainOfThought, TypedPredict};
pub use signature::{
    FieldKind, InField, OutField, Signature, SignatureSpec, chain_of_thought, json_field_schema,
    predict,
};
