//! A DSPy-style layer in Rust: declare a task as a [`struct@signature::Signature`], drive it
//! through a module in [`mod@predict`], and let the [`adapter`]s carry it over the wire to the
//! provider behind [`lm::LM`].

// `#[derive(Signature)]` expands to `::dsrust::...` paths; the alias keeps those
// paths valid when the derive is used inside this crate itself.
extern crate self as dsrust;

pub mod adapter;
pub mod evaluate;
pub mod example;
pub mod lm;
pub mod module;
pub mod interpreter;
pub mod optimize;
pub mod predict;
pub mod react;
pub mod signature;

pub use adapter::baml::BamlAdapter;
pub use adapter::xml::XmlAdapter;
pub use adapter::{Adapter, ChatAdapter, Extraction, JsonAdapter, NativeFunctionCalling, Reasoning, ReasoningEffort, TwoStepAdapter};
pub use adapter::{
    Audio, Citation, Citations, Code, Document, File, Formatted, History, Image, MediaType,
    ToolCall, ToolCallResult, ToolCallResults, ToolCalls, Type,
};
pub use lm::Capabilities;
pub use dsrust_derive::Module;
pub use evaluate::{Evaluate, Evaluation, Scored, exact_match};
pub use example::{Completions, Example, Prediction};
pub use module::{Ask, Forward, Module, NamedPredictor, PredictorState, ProgramState, TraceStep};
pub use optimize::{
    BootstrapFewShot, COPRO, DynOptimizer, Ensemble, Ensembled, Feedback, GEPA, GepaOutcome,
    LabeledFewShot, MIPROv2, Optimizer,
};
pub use react::{FnTool, ReAct, ReActV2, Tool, Trajectory, mcp_tool, mcp_tool_args, mcp_tool_result};

/// Items the macros expand into so a caller does not have to depend on them directly.
#[doc(hidden)]
#[path = "macro_support.rs"]
pub mod __macro_support;
pub use lm::dummy::DummyLM;
pub use lm::global::configure_model;
pub use lm::{
    ChatModel, ChatTurn, DEFAULT_PROVIDER_TIMEOUT, LM, ModelRef, OutputMode, Provider, Role,
    configure, configure_with_client,
};
pub use interpreter::{
    CodeInterpreter, Executed, ReplEntry, ReplHistory, ReplVariable, SandboxSerializable,
    build_repl_variable,
};
pub use predict::{
    Answered, BestOfN, ChainOfThought, CodeAct, MultiChainComparison, Parallel, Predict,
    ProgramOfThought, Refine, Rlm, Steering, TypedChainOfThought, TypedPredict,
};
pub use signature::{
    FieldEdit, FieldKind, InField, LiteralValue, OutField, Side, Signature, SignatureSpec,
    chain_of_thought, json_field_schema, predict, signature,
};
