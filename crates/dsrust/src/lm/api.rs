//! dspy 3.3's normalized LM API, from `dspy/core/types.py`.
//!
//! Added beside the existing call path rather than replacing it, which is upstream's own
//! migration shape: opt-in at 3.3, default at 3.5, the legacy types gone by 4.0.

mod builder;
mod config;
pub(crate) mod defaults;
mod delta;
mod event;
mod history;
pub(crate) mod interop;
pub use interop::wire_messages_of;
mod legacy;
mod message;
mod openai_shape;
mod part;
mod patch;
mod request;
mod response;
mod stream;
mod wire;

pub use builder::LmOutputBuilder;
pub use config::{
    LmCacheConfig, LmConfig, LmPromptCacheConfig, LmReasoningConfig, LmToolChoice, Logprobs,
    RolloutId, ToolChoiceMode,
};
pub use delta::LmDelta;
pub use event::LmStreamEvent;
pub use history::LmHistoryEntry;
pub use legacy::part_of_block;
pub use message::{LmMessage, LmToolSpec};
pub use part::{Detail, DocumentSource, LEGACY_BLOCK, LmPart, LmSource, Metadata};
pub use patch::LmRequestPatch;
pub use request::LmRequest;
pub use response::{LmOutput, LmResponse};
pub use stream::LmStream;
pub use wire::{Content, blocks_content, blocks_of, content_of};
