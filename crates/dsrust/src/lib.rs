//! A DSPy-style layer in Rust: declare a task as a [`struct@signature::Signature`], drive it
//! through a module in [`mod@predict`], and let the [`adapter`]s carry it over the wire to the
//! provider behind [`lm::LM`].

// `#[derive(Signature)]` expands to `::dsrust::...` paths; the alias keeps those
// paths valid when the derive is used inside this crate itself.
extern crate self as dsrust;

/// The HTTP client this crate uses, re-exported because [`ChatModel::forward`] takes one.
///
/// A provider of your own is the `ChatModel` trait, and its signature names a `reqwest::Client`. A
/// caller who depends only on `dsrust` therefore could not write that `impl` at all — and one who
/// added `reqwest` of their own would have to keep its major version matched to this crate's by
/// hand. Reach it as `dsrust::reqwest::Client` and neither is a problem.
pub use reqwest;

pub mod adapter;
pub mod evaluate;
pub mod example;
pub mod interpreter;
pub mod lm;
pub mod module;
pub mod optimize;
pub mod predict;
pub mod react;
pub mod signature;

pub use adapter::baml::BamlAdapter;
pub use adapter::xml::XmlAdapter;
pub use adapter::{
    Adapter, ChatAdapter, Extraction, JsonAdapter, NativeFunctionCalling, Reasoning,
    ReasoningEffort, TwoStepAdapter,
};
pub use adapter::{
    Audio, Citation, Citations, Code, Document, File, Formatted, History, Image, MediaType,
    ToolCall, ToolCallResult, ToolCallResults, ToolCalls, Type,
};
pub use dsrust_derive::Module;
pub use evaluate::{Evaluate, Evaluation, Scored, exact_match};
pub use example::{Completions, Example, Prediction};
pub use lm::Capabilities;
pub use module::{Ask, Forward, Module, NamedPredictor, PredictorState, ProgramState, TraceStep};
pub use optimize::{
    Attempt, BootstrapFewShot, BootstrapRandomSearch, COPRO, DynOptimizer, Ensemble, Ensembled,
    Feedback, GEPA, GepaOutcome, LabeledFewShot, MIPROv2, Optimizer,
};
pub use react::{
    FnTool, ReAct, ReActV2, Tool, Trajectory, mcp_tool, mcp_tool_args, mcp_tool_result,
};

/// Items the macros expand into so a caller does not have to depend on them directly.
#[doc(hidden)]
#[path = "macro_support.rs"]
pub mod __macro_support;
pub use interpreter::{
    CodeInterpreter, Executed, ReplEntry, ReplHistory, ReplVariable, SandboxSerializable,
    build_repl_variable,
};
pub use lm::dummy::DummyLM;
pub use lm::global::configure_model;
pub use lm::{
    ChatModel, ChatTurn, DEFAULT_PROVIDER_TIMEOUT, LM, ModelRef, OutputMode, Provider, Role,
    configure, configure_with_client,
};
pub use predict::{
    Answered, BestOfN, ChainOfThought, CodeAct, MultiChainComparison, Parallel, Predict,
    ProgramOfThought, Refine, Rlm, Steering, TypedChainOfThought, TypedPredict,
};
pub use signature::{
    ChainOfThought, FieldEdit, FieldKind, InField, LiteralValue, OutField, Predict, Side,
    Signature, SignatureSpec, json_field_schema, make_signature,
};
