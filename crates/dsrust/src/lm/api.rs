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
mod inspect;
mod items;
mod legacy;
mod legacy_request;
mod message;
mod openai_shape;
mod part;
mod patch;
mod request;
mod response;
mod source;
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
pub use inspect::pretty_print_history;
pub use items::{LmItem, messages_from_items};
pub use legacy::part_of_block;
pub use legacy_request::sanitized as sanitized_legacy_message;
pub use message::roles::{Assistant, Developer, System, User};
pub use message::{LmMessage, LmToolSpec, after_system, system_of};
pub use part::{Detail, LEGACY_BLOCK, LmPart, Metadata};
pub use patch::LmRequestPatch;
pub use request::LmRequest;
pub(crate) use request::request_of;
pub use response::{LmOutput, LmResponse};
pub use source::{DocumentSource, LmSource};
pub use stream::LmStream;
pub use wire::{Content, blocks_content, blocks_of, content_of};
