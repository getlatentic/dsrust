//! The legacy door: a chat message dict, written by anything, made readable by the strict model.
//!
//! dspy `_sanitize_legacy_message` and `_normalize_legacy_content_block`. A caller holding messages
//! that came out of a provider SDK has more than a conversation: an assistant turn dumped from an
//! OpenAI response carries `refusal`, `annotations` and `function_call` beside its content, and a
//! Responses output item carries `id`, `type` and `status` as well. None of those are message
//! *inputs*, and the typed model forbids what it does not declare — so the door drops them here.
//!
//! It also rewrites the Responses API's own content blocks into their chat spellings, because the
//! same conversation replayed from a Responses reply comes back saying `input_text` and
//! `output_text` where a chat message says `text`. Reading direction off the block would let an
//! assistant's `output_text` re-emit as user input; rewriting it to `text` leaves the direction to
//! the role, which is the only thing that knows it.
//!
//! Upstream is explicit that this tolerance lives at one door and nowhere else — "the typed path
//! stays strict" — so nothing here is reachable from
//! [`LmMessage`](super::LmMessage)'s own deserializer, which still refuses an unknown field.

use serde_json::{Map, Value};

/// The keys a message may hand the typed model. Everything else a provider SDK dumped alongside is
/// output, not input, and is dropped rather than refused.
const MESSAGE_KEYS: [&str; 6] = [
    "role",
    "content",
    "name",
    "metadata",
    "tool_calls",
    "tool_call_id",
];

/// One legacy message, keeping only what a message carries and rewriting its content blocks.
///
/// Anything that is not an object is handed back untouched: upstream returns it unchanged and lets
/// the typed model fail on it, which reports the real shape rather than an empty message.
pub fn sanitized(message: &Value) -> Value {
    let Some(fields) = message.as_object() else {
        return message.clone();
    };
    let mut cleaned = Map::new();
    for key in MESSAGE_KEYS {
        if let Some(value) = fields.get(key) {
            cleaned.insert(key.to_owned(), value.clone());
        }
    }
    if let Some(Value::Array(blocks)) = cleaned.get("content") {
        let rewritten = blocks.iter().map(chat_block).collect();
        cleaned.insert("content".to_owned(), Value::Array(rewritten));
    }
    Value::Object(cleaned)
}

/// One Responses-native content block in its chat spelling, or the block itself when it already is
/// one — an unknown block passes through, as every unmodelled block does.
fn chat_block(block: &Value) -> Value {
    let Some(fields) = block.as_object() else {
        return block.clone();
    };
    match fields.get("type").and_then(Value::as_str) {
        // The direction is the role's to say, so both spellings collapse to the one that does not
        // claim one. A missing `text` becomes empty rather than absent, as upstream's default does.
        Some("input_text" | "output_text") => serde_json::json!({
            "type": "text",
            "text": fields.get("text").cloned().unwrap_or_else(|| Value::String(String::new())),
        }),
        // Only the flat spelling: `image_url` is a bare string here, where the chat block nests it
        // under a `url`. A block already carrying the nested object is left alone.
        Some("input_image") => match fields.get("image_url") {
            Some(Value::String(url)) => serde_json::json!({
                "type": "image_url",
                "image_url": { "url": url },
            }),
            _ => block.clone(),
        },
        Some("input_file") => {
            let mut file = Map::new();
            for key in ["file_data", "file_id", "filename"] {
                match fields.get(key) {
                    Some(Value::Null) | None => {}
                    Some(value) => {
                        file.insert(key.to_owned(), value.clone());
                    }
                }
            }
            serde_json::json!({ "type": "file", "file": file })
        }
        _ => block.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// An assistant turn as the OpenAI SDK dumps one: the content, and three output fields that
    /// are not inputs at all. The typed model forbids an unknown field, so without this door the
    /// whole conversation is refused over `refusal`.
    #[test]
    fn a_provider_sdk_dump_keeps_only_what_a_message_carries() {
        let cleaned = sanitized(&json!({
            "role": "assistant",
            "content": "hi",
            "refusal": null,
            "annotations": [],
            "function_call": null,
        }));
        assert_eq!(cleaned, json!({ "role": "assistant", "content": "hi" }));
    }

    /// A Responses *output item* replayed as a message: `id`, `type` and `status` go the same way.
    #[test]
    fn a_responses_output_item_loses_its_item_fields() {
        let cleaned = sanitized(&json!({
            "id": "msg_1",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{ "type": "output_text", "text": "hello", "annotations": [] }],
        }));
        assert_eq!(
            cleaned,
            json!({
                "role": "assistant",
                "content": [{ "type": "text", "text": "hello" }],
            })
        );
    }

    /// Both Responses spellings collapse to `text`, so the role stays the only thing saying which
    /// way a turn points. A user message really does arrive carrying `output_text`.
    #[test]
    fn both_directed_text_blocks_become_the_undirected_one() {
        for spelling in ["input_text", "output_text"] {
            let cleaned = sanitized(&json!({
                "role": "user",
                "content": [{ "type": spelling, "text": "Say hi." }],
            }));
            assert_eq!(
                cleaned["content"],
                json!([{ "type": "text", "text": "Say hi." }]),
                "for {spelling}"
            );
        }
    }

    /// The Responses image block puts the url where the chat block puts an object; only the flat
    /// spelling is rewritten, so a block that already nests one is left as it is.
    #[test]
    fn a_flat_image_url_is_nested_and_a_nested_one_is_left_alone() {
        let nested = json!({ "type": "input_image", "image_url": { "url": "u" } });
        let cleaned = sanitized(&json!({
            "role": "user",
            "content": [{ "type": "input_image", "image_url": "u" }, nested.clone()],
        }));
        assert_eq!(
            cleaned["content"],
            json!([{ "type": "image_url", "image_url": { "url": "u" } }, nested])
        );
    }

    /// A file block's three sources move under a `file` envelope, and a null one is not a source.
    #[test]
    fn a_responses_file_block_moves_its_sources_under_an_envelope() {
        let cleaned = sanitized(&json!({
            "role": "user",
            "content": [{
                "type": "input_file",
                "file_data": "data:application/pdf;base64,YQ==",
                "file_id": null,
                "filename": "a.pdf",
            }],
        }));
        assert_eq!(
            cleaned["content"],
            json!([{
                "type": "file",
                "file": { "file_data": "data:application/pdf;base64,YQ==", "filename": "a.pdf" },
            }])
        );
    }

    /// A block nobody modelled passes through, which is what keeps a provider's own shape alive.
    #[test]
    fn an_unknown_block_is_not_touched() {
        let block = json!({ "type": "wildcard_v9", "payload": { "k": 1 } });
        let cleaned = sanitized(&json!({ "role": "user", "content": [block.clone()] }));
        assert_eq!(cleaned["content"], json!([block]));
    }

    /// A message that is not an object at all is handed back, so the typed model reports the shape
    /// it really got rather than an empty message it was handed instead.
    #[test]
    fn a_message_that_is_not_an_object_is_left_for_the_model_to_refuse() {
        assert_eq!(sanitized(&json!("hi")), json!("hi"));
    }
}
