//! dspy `adapters/types/`: the types a signature field can be declared as, beyond the scalars.
//!
//! Upstream gives each its own module under `dspy/adapters/types/` — `reasoning.py`, `code.py`,
//! `tool.py` — and each states how it renders into a prompt and reads back out of a reply. The
//! same split lives here, so a type ported from upstream lands in the file that names it.

pub mod audio;
pub mod base;
pub mod citation;
pub mod code;
pub mod document;
pub mod file;
pub mod history;
pub mod image;
pub mod reasoning;
pub mod tool;

pub use audio::{Audio, Container};
pub use base::{Formatted, Type, serialized, to_field_value};
pub use citation::{Citation, Citations};
pub use code::Code;
pub use document::{Document, MediaType};
pub use file::File;
pub use history::History;
pub use image::Image;
pub use reasoning::Reasoning;
pub use tool::{ToolCall, ToolCallResult, ToolCallResults, ToolCalls};
