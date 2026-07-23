//! dspy `adapters/types/`: the types a signature field can be declared as, beyond the scalars.
//!
//! Upstream gives each its own module under `dspy/adapters/types/` — `reasoning.py`, `code.py`,
//! `tool.py` — and each states how it renders into a prompt and reads back out of a reply. The
//! same split lives here, so a type ported from upstream lands in the file that names it.

pub mod reasoning;
pub mod tool;

pub use reasoning::Reasoning;
pub use tool::{ToolCall, ToolCallResult, ToolCallResults, ToolCalls};
