//! The conversation as an adapter builds it internally, and how it becomes the messages a
//! request carries.
//!
//! `Adapter::format` answers with [`LmMessage`](super::api::LmMessage), as dspy's does. `ChatTurn`
//! is the vocabulary the shipped adapters render *into* before that: `conversation` grows a demo
//! into a user/assistant pair, and `blocks::split_custom_types` splits a rendered field into
//! multimodal blocks. [`messages_of`] is where the two meet.

use serde_json::Value;

use super::api::{Content, LmMessage, LmPart, part_of_block};

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

/// The message list a render *is* — dspy's `format` returns exactly this, the system prompt as the
/// first message rather than as a value beside the turns.
///
/// The pair it replaces was the pre-3.3 shape: `(String, Vec<ChatTurn>)` made the system prompt a
/// different kind of thing from the turns, which is a distinction upstream's own type does not have
/// and the wire does not either.
pub(crate) fn messages_of(system: &str, turns: &[ChatTurn]) -> Vec<LmMessage> {
    let mut messages = vec![LmMessage::system(vec![LmPart::text(system)])];
    for turn in turns {
        messages.push(LmMessage::new(turn.role.as_str(), parts_of(&turn.content)));
    }
    messages
}

/// A turn's content as the parts it was collapsed from — the inverse of
/// [`content_of`](super::content_of), reading a block back to a part the same way
/// [`part_of_block`] does the multimodal path.
fn parts_of(content: &Content) -> Vec<LmPart> {
    match content {
        Content::Text(text) => vec![LmPart::text(text)],
        Content::Blocks(blocks) => blocks.iter().map(part_of_block).collect(),
        // Already parts: a turn that carries tool calls or a tool result built them directly,
        // because no content block spells either.
        Content::Parts(parts) => parts.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// The system prompt leads as its own message; every turn follows under the role it declared.
    #[test]
    fn a_render_becomes_a_system_message_then_its_turns() {
        let messages = messages_of(
            "Be concise.",
            &[ChatTurn::user("Why?"), ChatTurn::assistant("Because.")],
        );

        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].role, "system");
        assert_eq!(messages[0].text().as_deref(), Some("Be concise."));
        assert_eq!(messages[1].role, "user");
        assert_eq!(messages[2].role, "assistant");
    }

    /// The system message is always present, even empty, because the wire always carries one — so
    /// a provider lifting it into its own field reads an empty string, not a missing key.
    #[test]
    fn an_empty_system_still_leads_with_a_system_message() {
        let messages = messages_of("", &[]);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "system");
        assert_eq!(messages[0].text().as_deref(), Some(""));
    }

    /// A multimodal turn's blocks are read back to parts, so the messages render the same blocks
    /// the turn was built from.
    #[test]
    fn a_multimodal_turn_reads_its_blocks_back_to_parts() {
        let turns = [ChatTurn {
            role: Role::User,
            content: Content::Blocks(vec![
                json!({ "type": "text", "text": "look:" }),
                json!({ "type": "image_url", "image_url": { "url": "u" } }),
            ]),
        }];
        assert_eq!(messages_of("", &turns)[1].parts.len(), 2);
    }
}
