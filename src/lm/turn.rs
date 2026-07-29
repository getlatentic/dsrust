//! The conversation as the adapters still write it, and what the reply should look like.
//!
//! `ChatTurn` predates the typed 3.3 boundary in [`api`](super::api): an adapter renders into
//! these and the request is raised from them at the edge. Replacing it with `LmMessage` is the
//! last slice of that migration (s13-3), and until then this is what `Adapter::format` answers
//! with.

use serde_json::Value;

use super::api::Content;

/// One side of the conversation; every provider speaks the user/assistant pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
    /// What a tool returned. dspy sends these as their own messages once a provider has called
    /// tools natively, each naming the call it answers.
    Tool,
}

impl Role {
    /// The name this role travels under on the wire, and the one dspy's message dicts use.
    pub fn as_str(self) -> &'static str {
        match self {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }
}

/// One turn of the conversation. Retries append the model's own previous reply as an
/// assistant turn followed by a corrective user turn, so the model sees what it wrote.
#[derive(Debug, Clone)]
pub struct ChatTurn {
    pub role: Role,
    pub content: Content,
}

impl ChatTurn {
    pub fn user(content: impl Into<Content>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<Content>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
        }
    }
}

/// What the reply should look like on the wire. `Json` engages the provider's native
/// structured output (Anthropic json_schema, OpenRouter/ollama JSON mode); `Text` leaves
/// the reply free-form for marker-based adapters, which need no provider support at all.
#[derive(Debug, Clone, Copy)]
pub enum OutputMode<'a> {
    Text,
    Json { schema: &'a Value },
}
