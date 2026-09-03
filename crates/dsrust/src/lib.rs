//! A DSPy-style layer in Rust: declare a task as a [`struct@signature::Signature`], drive it
//! through a module in [`mod@predict`], and let the [`adapter`]s carry it over the wire to the
//! provider behind [`lm::LM`].

// `#[derive(Signature)]` expands to `::dsrust::...` paths; the alias keeps those
// paths valid when the derive is used inside this crate itself.
extern crate self as dsrust;

/// The HTTP client this crate uses, re-exported for the two places its type is still named.
///
/// [`ChatModel`] no longer mentions it — implementing a provider of your own needs nothing from
/// here. What remains is [`configure_with_client`], for a caller supplying a pooled client of their
/// own, and `LM::forward_stream`, whose returned stream has to borrow one that outlives the call.
pub use reqwest;

/// `serde_json`, because this crate's surface is made of its types and a caller has to be able to
/// *build* one.
///
/// Reading an answer needs nothing — the methods on [`Value`](serde_json::Value) are inherent. But
/// a runtime-shaped signature, a tool's argument schema, or an [`Input`] is a `Value` a caller
/// constructs, and constructing one means `json!`. Without this that is a `cargo add serde_json`
/// the README never mentions, discovered as a compile error — the same shape as the `serde` and
/// `serde_json` the derive needed, which an outside project reported.
///
/// `json!` resolves through here because a `macro_rules!` macro refers to its own crate as
/// `$crate`, which is bound at definition and does not consult the caller's extern prelude:
///
/// ```
/// use dsrust::serde_json::json;
/// let schema = json!({ "type": "string" });
/// assert_eq!(schema["type"], "string");
/// ```
///
/// [`Input`]: crate::adapter::Input
pub use serde_json;

/// The error crate `Tool::call` answers with, re-exported so a tool can be written by a crate that
/// depends on nothing but this one. Naming the return type is unavoidable — `#[tool]` reads the
/// function you wrote rather than rewriting it — and `anyhow::Result<String>` is what it has to be.
///
/// ```
/// # use dsrust::{Tool, tool};
/// #[tool]
/// /// Greet one person by name.
/// fn greet(name: String) -> dsrust::anyhow::Result<String> {
///     Ok(format!("Hello, {name}."))
/// }
/// assert_eq!(Greet.name(), "greet");
/// ```
pub use anyhow;

pub mod adapter;
pub mod callback;
mod error;
pub mod evaluate;
pub mod example;
pub mod hasher;
pub mod interpreter;
pub mod lm;
mod mimetypes;
pub mod module;
pub mod numpy;
pub mod observe;
pub mod optimize;
pub mod predict;
pub mod python;
pub mod react;
mod resource;
pub mod retrievers;
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
pub use callback::{CallId, Callback, Rendered, configure_callbacks};
pub use dsrust_derive::Module;
pub use evaluate::{Evaluate, Evaluation, Pass, Scored, exact_match};
pub use example::{Completions, Example, Prediction};
pub use hasher::Hasher;
pub use lm::Capabilities;
pub use lm::dummy_vectorizer::DummyVectorizer;
pub use lm::embedding::{EmbedCall, Embedder, EmbedderModel};
pub use module::{
    Ask, FailedPrediction, Forward, Module, NamedPredictor, PredictorState, ProgramState,
    StepOutputs, TraceStep, Typed,
};
pub use optimize::knn_fewshot::{KnnFewShot, KnnFewShotProgram};
pub use optimize::{
    Attempt, BootstrapFewShot, BootstrapRandomSearch, COPRO, DynOptimizer, Ensemble, Ensembled,
    Feedback, GEPA, GepaOutcome, LabeledFewShot, MIPROv2, MetricContext, Optimizer,
};
pub use predict::knn::Knn;
pub use react::{
    AsyncFnTool, FnTool, ReAct, ReActV2, Tool, Trajectory, mcp_tool, mcp_tool_args,
    mcp_tool_result, typed_tool,
};
pub use retrievers::{Embeddings, EmbeddingsWithScores, Retrieved};

/// Items the macros expand into so a caller does not have to depend on them directly.
#[doc(hidden)]
#[path = "macro_support.rs"]
pub mod __macro_support;
pub use interpreter::{
    CodeInterpreter, Executed, ReplEntry, ReplHistory, ReplVariable, SandboxSerializable,
    build_repl_variable,
};
pub use lm::dummy::DummyLM;
pub use lm::global::{Scope, configure_model, context, context_model, context_with_client};
pub use lm::{
    Assistant, ChatModel, ChatTurn, DEFAULT_PROVIDER_TIMEOUT, Developer, LM, LmItem, LmMessage,
    LmPart, LmRequest, LmResponse, ModelRef, OutputMode, Provider, Role, System, User, configure,
    configure_with_client,
};
pub use predict::{
    Answered, BestOfN, ChainOfThought, CodeAct, MultiChainComparison, Parallel, Predict,
    ProgramOfThought, Refine, Rlm, Steering, TypedChainOfThought, TypedPredict,
};
pub use signature::{
    ChainOfThought, FieldEdit, FieldKind, InField, LiteralValue, OutField, Predict, Side,
    Signature, SignatureSpec, json_field_schema, make_signature, tool,
};
