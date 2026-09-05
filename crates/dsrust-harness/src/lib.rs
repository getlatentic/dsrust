//! Coding agents as dsrust models.
//!
//! [`HarnessModel`] puts any [`harness::Harness`] — Claude Code, Codex, an
//! ACP agent, the built-in OpenAI-compatible runtime — behind dsrust's
//! [`ChatModel`](dsrust::lm::ChatModel), so an ordinary `Predict`, a
//! `ChainOfThought`, or a `ReActV2` loop runs on an agent the user already has
//! signed in, on its own billing, with no API key in the program.
//!
//! ```no_run
//! use std::sync::Arc;
//! use dsrust_harness::harness::Claude;
//! use dsrust::lm::DynChatModel;
//! use dsrust::{Example, Module, Predict};
//! use dsrust_harness::HarnessModel;
//!
//! # async fn ask() -> anyhow::Result<()> {
//! let claude = Arc::new(HarnessModel::builder(Claude::new()).cwd(std::env::temp_dir()).build()?);
//! let qa = Predict::parse("question -> answer")?.set_lm(claude as Arc<dyn DynChatModel>);
//! let answer = qa.forward(Example::new([("question", serde_json::json!("What is 2+2?"))])).await?;
//! # Ok(())
//! # }
//! ```
//!
//! Every call runs the agent with its own tools withheld — the reasons are on
//! [`HarnessModel`] — and carries [`MARKER_DISCIPLINE`] as standing instructions,
//! because the fragile part of reading an agent's reply is not the JSON inside a
//! field but the markers around it.
//!
//! The other direction is [`tool_server`]: dsrust [`Tool`](dsrust::Tool)s offered
//! to the agent as an MCP server living in this process, with
//! `.tools(ToolAccess::Default)` on the [`HarnessModelBuilder`] letting the agent use them. A `Predict`
//! over that is the agent as a module — one answer per call, reached with your
//! tools, and still a predictor an optimizer can rewrite.

mod builder;
mod collect;
mod model;
mod prompt;
pub mod tools;

/// agent-harness, as this crate builds against it — reach `Claude`, `Codex`,
/// `OpenHarness`, `ToolServer` through here rather than depending on it too. A
/// second copy under a different version would be a different `Harness` trait,
/// and a `Claude` from it would not be one this crate's model accepts.
pub use harness;

pub use builder::HarnessModelBuilder;
pub use model::{HarnessModel, MARKER_DISCIPLINE};
pub use tools::{read_only_tool_server, tool_server};
