//! dspy `LMMessage.normalize_parts`: reading a message written the way a provider writes one.
//!
//! A typed message is a role and a list of parts, but almost nothing hands one over in that shape.
//! A provider's reply, an `lm.history` entry, a saved conversation and every OpenAI-shaped fixture
//! are `{role, content}` — with `tool_calls` beside the content on an assistant turn, and
//! `tool_call_id` on a tool result. Upstream accepts all of those into `LMMessage` and normalises
//! them; without the same, a typed message can only be built by code that already holds parts.

use serde::Deserialize;
use serde_json::{Map, Value};

use super::legacy::part_of_block;
use super::message::LmMessage;
use super::part::{LmPart, Metadata};

/// A message as it may be written: parts, or the provider shape, or both.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct OpenAiShaped {
    role: String,
    #[serde(default)]
    parts: Option<Vec<LmPart>>,
    #[serde(default)]
    content: Option<Value>,
    #[serde(default)]
    tool_calls: Option<Vec<Value>>,
    #[serde(default)]
    tool_call_id: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    metadata: Metadata,
}

impl TryFrom<OpenAiShaped> for LmMessage {
    type Error = String;

    /// The two shapes do not mix. Upstream pops a provider key only in the branch that reads it and
    /// leaves `extra="forbid"` to reject whatever is left, so `parts` beside `content`, or
    /// `tool_call_id` on anything but a bare tool message, is refused rather than half-read. Only
    /// `tool_calls` is popped in every branch, and only it may sit beside `parts`.
    fn try_from(written: OpenAiShaped) -> Result<Self, Self::Error> {
        let tool_result = written.role == "tool" && written.parts.is_none();
        if written.parts.is_some() && written.content.is_some() {
            return Err("a message carries `parts` or `content`, not both".to_owned());
        }
        if written.tool_call_id.is_some() && !tool_result {
            return Err(
                "`tool_call_id` belongs to a tool message written with `content`, not `parts`"
                    .to_owned(),
            );
        }

        // A tool result written the provider's way carries the call it answers beside the content,
        // and its `name` belongs to the result rather than to the message.
        if tool_result {
            return Ok(LmMessage {
                role: written.role,
                parts: vec![LmPart::ToolResult {
                    call_id: written.tool_call_id,
                    name: written.name,
                    content: parts_of(written.content.as_ref()),
                    is_error: false,
                    provider_data: Map::new(),
                    metadata: Map::new(),
                }],
                name: None,
                metadata: written.metadata,
            });
        }

        // Parts where they were given, else the content read into them; either way the tool calls
        // an assistant turn carries alongside are appended.
        let mut parts = match written.parts {
            Some(parts) => parts,
            // dspy distinguishes an absent `content` from a null one: both give no parts here, but
            // only a message with neither key is empty by construction.
            None => parts_of(written.content.as_ref()),
        };
        parts.extend(written.tool_calls.into_iter().flatten().map(tool_call_of));

        Ok(LmMessage {
            role: written.role,
            parts,
            name: written.name,
            metadata: written.metadata,
        })
    }
}

/// dspy `_parts_from_openai_content`: nothing, one text part, or one part per block.
fn parts_of(content: Option<&Value>) -> Vec<LmPart> {
    match content {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::String(text)) => vec![LmPart::text(text)],
        Some(Value::Array(blocks)) => blocks.iter().map(part_of_block).collect(),
        Some(other) => vec![part_of_block(other)],
    }
}

/// dspy `_tool_call_from_openai`: the call under its OpenAI envelope, whose arguments arrive as
/// JSON *text* far more often than as an object.
fn tool_call_of(written: Value) -> LmPart {
    let function = written.get("function").unwrap_or(&Value::Null);
    let name = function
        .get("name")
        .or_else(|| written.get("name"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let args = match function.get("arguments").or_else(|| written.get("args")) {
        Some(Value::String(text)) => serde_json::from_str(text).unwrap_or_default(),
        Some(Value::Object(args)) => args.clone(),
        _ => Map::new(),
    };
    LmPart::ToolCall {
        id: written
            .get("id")
            .or_else(|| written.get("call_id"))
            .and_then(Value::as_str)
            .map(str::to_owned),
        name,
        args,
        provider_data: Map::new(),
        metadata: Map::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn read(written: Value) -> LmMessage {
        serde_json::from_value(written).expect("a message the provider could have written")
    }

    /// The shape every provider reply and history entry arrives in.
    #[test]
    fn a_role_and_a_string_content_become_one_text_part() {
        let message = read(json!({ "role": "user", "content": "Why?" }));
        assert_eq!(message.parts, vec![LmPart::text("Why?")]);
        assert_eq!(message.text().as_deref(), Some("Why?"));
    }

    /// A block list becomes a part each, through the same reader the legacy wire uses.
    #[test]
    fn a_block_list_becomes_a_part_each() {
        let message = read(json!({
            "role": "user",
            "content": [
                { "type": "text", "text": "look" },
                { "type": "image_url", "image_url": { "url": "https://example.com/a.png" } },
            ],
        }));
        assert_eq!(message.parts.len(), 2);
        assert_eq!(message.parts[0], LmPart::text("look"));
        assert!(matches!(message.parts[1], LmPart::Image { .. }));
    }

    /// An assistant turn carries its calls beside the content, and both end up as parts.
    #[test]
    fn tool_calls_join_the_content_as_parts() {
        let message = read(json!({
            "role": "assistant",
            "content": "on it",
            "tool_calls": [{
                "id": "call_1",
                "function": { "name": "search", "arguments": "{\"q\": \"rust\"}" },
            }],
        }));
        assert_eq!(message.parts.len(), 2);
        let LmPart::ToolCall { id, name, args, .. } = &message.parts[1] else {
            panic!("the call is a part: {:?}", message.parts[1]);
        };
        assert_eq!(id.as_deref(), Some("call_1"));
        assert_eq!(name, "search");
        // Arguments arrive as JSON text far more often than as an object, and are read either way.
        assert_eq!(args["q"], json!("rust"));
    }

    /// A turn that is only calls has no content at all, which is what a provider sends.
    #[test]
    fn an_assistant_turn_of_only_calls_has_no_text() {
        let message = read(json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{ "function": { "name": "search", "arguments": {} } }],
        }));
        assert_eq!(message.parts.len(), 1);
        assert_eq!(message.text(), None);
    }

    /// A tool result names the call it answers, and its `name` belongs to the result.
    #[test]
    fn a_tool_result_takes_the_call_id_and_the_name() {
        let message = read(json!({
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "search",
            "content": "found it",
        }));
        assert_eq!(message.name, None, "the name went to the result");
        let LmPart::ToolResult { call_id, name, content, .. } = &message.parts[0] else {
            panic!("a tool message is one result: {:?}", message.parts[0]);
        };
        assert_eq!(call_id.as_deref(), Some("call_1"));
        assert_eq!(name.as_deref(), Some("search"));
        assert_eq!(content, &vec![LmPart::text("found it")]);
    }

    /// A tool result with no content is a result with no parts, not a result with an empty one.
    #[test]
    fn a_tool_result_with_no_content_has_no_parts() {
        let message = read(json!({ "role": "tool", "tool_call_id": "call_1", "content": null }));
        let LmPart::ToolResult { content, .. } = &message.parts[0] else {
            panic!("a tool message is one result");
        };
        assert!(content.is_empty());
    }

    /// The typed form still reads as itself, and still round-trips.
    #[test]
    fn parts_given_explicitly_are_kept() {
        let typed = json!({ "role": "user", "parts": [{ "type": "text", "text": "Why?" }] });
        let message = read(typed.clone());
        assert_eq!(message.parts, vec![LmPart::text("Why?")]);
        assert_eq!(serde_json::to_value(&message).expect("serializes"), typed);
    }

    /// A key belonging to neither shape is still refused — the point is to accept what a provider
    /// writes, not to stop checking.
    #[test]
    fn an_unknown_key_is_still_refused() {
        let refused: Result<LmMessage, _> =
            serde_json::from_value(json!({ "role": "user", "content": "hi", "nonsense": 1 }));
        assert!(refused.is_err());
    }

    /// The two shapes do not mix, in exactly the combinations upstream refuses. Each is a message
    /// written half one way and half the other, and reading it either way would drop something.
    #[test]
    fn the_two_shapes_do_not_mix() {
        for written in [
            json!({ "role": "user", "parts": [], "content": "the old spelling" }),
            json!({ "role": "user", "parts": [], "tool_call_id": "call_1" }),
            json!({ "role": "tool", "parts": [], "tool_call_id": "call_1" }),
            json!({ "role": "user", "content": "x", "tool_call_id": "call_1" }),
        ] {
            let refused: Result<LmMessage, _> = serde_json::from_value(written.clone());
            assert!(refused.is_err(), "should be refused: {written}");
        }
        // `tool_calls` is the exception: upstream reads it in every branch, so it may sit beside
        // `parts`.
        let mixed = read(json!({
            "role": "assistant",
            "parts": [{ "type": "text", "text": "on it" }],
            "tool_calls": [{ "function": { "name": "search", "arguments": "{}" } }],
        }));
        assert_eq!(mixed.parts.len(), 2);
    }

    /// A message with no content key at all is a message with no parts.
    #[test]
    fn no_content_key_is_no_parts() {
        assert!(read(json!({ "role": "user" })).parts.is_empty());
    }
}
