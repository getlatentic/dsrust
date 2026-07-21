//! dspy 3.3's normalized LM API, from `dspy/core/types.py`.
//!
//! Added beside the existing call path rather than replacing it, which is upstream's own
//! migration shape: opt-in at 3.3, default at 3.5, the legacy types gone by 4.0.

mod config;
mod message;
mod part;
mod request;
mod response;
mod wire;

pub use config::{
    LmCacheConfig, LmConfig, LmPromptCacheConfig, LmReasoningConfig, LmToolChoice, Logprobs,
    RolloutId, ToolChoiceMode,
};
pub use message::{LmMessage, LmToolSpec};
pub use part::{Detail, DocumentSource, LEGACY_BLOCK, LmPart, LmSource, Metadata};
pub use request::LmRequest;
pub use response::{LmOutput, LmResponse};
pub use wire::{Content, blocks_of, content_of};
