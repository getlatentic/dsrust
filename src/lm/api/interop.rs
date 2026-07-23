//! Raising the crate's legacy `(system, turns, mode, config)` render into a typed 3.3 request.
//!
//! The adapters still render the `(system, turns)` shape dspy 3.2.1 built, and
//! [`Predict`](crate::predict::Predict) hands the model an [`LMRequest`](ApiRequest). This is the
//! one seam between the two: `raise_request` builds the typed request from the rendered shape, its
//! messages collapsed to the exact blocks a provider reads, so the typed boundary sits in front of
//! the byte-exact providers without a rendered byte moving.

use super::{LmCacheConfig, LmConfig as ApiConfig, LmMessage, LmPart};
use super::{LmRequest as ApiRequest, RolloutId, part_of_block};
use crate::lm::{ChatTurn, Content, LmConfig as CallConfig, OutputMode};
use serde_json::Value;

/// A rendered `(system, turns, mode, config)` raised to the typed request — what an adapter that
/// renders the old shape hands the typed boundary, dspy's adapter building an `LMRequest`.
///
/// The first `system`-role message carries the system prompt, even when empty, because the legacy
/// wire always carries one; every turn becomes a message under the role it declared, its content
/// split back into the parts it was collapsed from. The model is left blank — this crate keeps it
/// on the [`LM`](crate::lm::LM) rather than in the request, so the provider fills it and the cache
/// key is taken against it separately.
pub(crate) fn raise_request(
    system: &str,
    turns: &[ChatTurn],
    mode: OutputMode,
    config: &CallConfig,
) -> ApiRequest {
    let mut messages = vec![LmMessage::system(vec![LmPart::text(system)])];
    for turn in turns {
        messages.push(LmMessage::new(turn.role.as_str(), parts_of(&turn.content)));
    }
    ApiRequest::new("", messages).configured(raise_config(mode, config))
}

/// The OpenAI-shaped messages these turns become — the list dspy's `format` returns.
///
/// A turn carrying tool calls or a tool result cannot be described by a role and a content alone:
/// the calls travel beside the content and a result names the call it answers. Rendering through
/// the request keeps that shape in one place, rather than rebuilt by every caller that needs it.
pub fn wire_messages_of(turns: &[ChatTurn]) -> Vec<Value> {
    let messages: Vec<LmMessage> = turns
        .iter()
        .map(|turn| LmMessage::new(turn.role.as_str(), parts_of(&turn.content)))
        .collect();
    ApiRequest::new("", messages).wire_messages()
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

/// The legacy config's four fields raised into the typed one, the rollout back under its nested
/// cache. The mode becomes the response format, the one place `OutputMode` lives in the typed
/// config rather than beside it.
fn raise_config(mode: OutputMode, config: &CallConfig) -> ApiConfig {
    ApiConfig {
        temperature: config.temperature,
        max_tokens: config.max_tokens,
        n: config.completions,
        response_format: match mode {
            OutputMode::Json { schema } => Some(schema.clone()),
            OutputMode::Text => None,
        },
        cache: config.rollout_id.map(|id| LmCacheConfig {
            rollout_id: Some(RolloutId::Number(id as i64)),
            ..LmCacheConfig::default()
        }),
        ..ApiConfig::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lm::Role;
    use serde_json::json;

    /// The system prompt leads as its own message; every turn follows under the role it declared.
    #[test]
    fn a_render_raises_to_a_system_message_then_turns() {
        let turns = [ChatTurn::user("Why?"), ChatTurn::assistant("Because.")];
        let raised = raise_request("Be concise.", &turns, OutputMode::Text, &CallConfig::default());

        assert_eq!(raised.messages.len(), 3);
        assert_eq!(raised.messages[0].role, "system");
        assert_eq!(raised.system(), "Be concise.");
        assert_eq!(raised.messages[1].role, "user");
        assert_eq!(raised.messages[2].role, "assistant");
    }

    /// The system message is always present, even empty, because the legacy wire always carries
    /// one — so a provider lifting it into its own field reads an empty string, not a missing key.
    #[test]
    fn an_empty_system_still_raises_to_a_system_message() {
        let raised = raise_request("", &[], OutputMode::Text, &CallConfig::default());
        assert_eq!(raised.messages.len(), 1);
        assert_eq!(raised.messages[0].role, "system");
        assert_eq!(raised.system(), "");
    }

    /// The mode and the four sampling fields cross into the typed config, a numeric rollout folded
    /// under the nested cache where the key reads it.
    #[test]
    fn the_mode_and_sampling_fields_cross_into_the_typed_config() {
        let schema = json!({ "type": "object" });
        let config = CallConfig {
            temperature: Some(0.7),
            max_tokens: Some(256),
            completions: Some(3),
            rollout_id: Some(5),
        };
        let raised = raise_request(
            "",
            &[ChatTurn::user("Why?")],
            OutputMode::Json { schema: &schema },
            &config,
        );

        assert_eq!(raised.config.temperature, Some(0.7));
        assert_eq!(raised.config.max_tokens, Some(256));
        assert_eq!(raised.config.n, Some(3));
        assert_eq!(raised.output_schema(), Some(&schema));
        assert_eq!(raised.config.rollout_id(), Some(&RolloutId::Number(5)));
    }

    /// A plain render carries no response format, so it raises to text mode — not an empty schema,
    /// which would change what every ordinary call asks for.
    #[test]
    fn a_render_with_no_schema_raises_to_no_response_format() {
        let raised = raise_request("", &[], OutputMode::Text, &CallConfig::default());
        assert_eq!(raised.output_schema(), None);
    }

    /// A multimodal turn's blocks are read back to parts, so the raised request renders the same
    /// blocks it was built from.
    #[test]
    fn a_multimodal_turn_raises_its_blocks_back_to_parts() {
        let blocks = Content::Blocks(vec![
            json!({ "type": "text", "text": "look:" }),
            json!({ "type": "image_url", "image_url": { "url": "u" } }),
        ]);
        let turns = [ChatTurn {
            role: Role::User,
            content: blocks,
        }];
        let raised = raise_request("", &turns, OutputMode::Text, &CallConfig::default());
        assert_eq!(raised.messages[1].parts.len(), 2);
    }
}
