//! `LMRequest`.

use serde_json::{Value, json};

use super::config::LmConfig;
use super::message::{LmMessage, LmToolSpec};
use super::part::{LmPart, Metadata};
use super::wire::{Content, content_of};
use crate::adapter::python_json::json_dumps;

/// One call as a value. The model is part of it, which is what lets a request be routed and
/// cached without ambient state deciding who answers.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LmRequest {
    pub model: String,
    pub messages: Vec<LmMessage>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<LmToolSpec>,
    #[serde(default)]
    pub config: LmConfig,
    #[serde(default, skip_serializing_if = "Metadata::is_empty")]
    pub metadata: Metadata,
}

impl LmRequest {
    pub fn new(model: impl Into<String>, messages: Vec<LmMessage>) -> Self {
        Self {
            model: model.into(),
            messages,
            tools: Vec::new(),
            config: LmConfig::default(),
            metadata: Metadata::new(),
        }
    }

    pub fn configured(mut self, config: LmConfig) -> Self {
        self.config = config;
        self
    }

    pub fn with_tools(mut self, tools: Vec<LmToolSpec>) -> Self {
        self.tools = tools;
        self
    }

    /// The top-level system prompt: the text of the first `system`-role message, or `""` when
    /// there is none. Read as a bare string, so a provider that carries the system prompt in its
    /// own field — Anthropic — reads the same bytes a native call would have carried.
    pub fn system(&self) -> &str {
        self.messages
            .iter()
            .find(|message| message.role == "system")
            // Only a lone text part is prose; a multi-part system message collapses to blocks the
            // way [`content_of`] does, and reads as no system prompt, which is what the legacy path
            // did with it.
            .and_then(|message| match message.parts.as_slice() {
                [only] => only.as_text(),
                _ => None,
            })
            .unwrap_or("")
    }

    /// Every message as the `{role, content}` a provider reads, the system message already first.
    /// Each content is [`content_of`] the message's parts — a bare string for a text-only message,
    /// a block list for a multimodal one — which is the same collapse the golden path renders
    /// through, so a message here goes on the wire byte for byte as a native one would.
    pub fn wire_messages(&self) -> Vec<Value> {
        self.messages.iter().map(wire_message).collect()
    }

    /// The same as [`wire_messages`](Self::wire_messages), without the system message — what a
    /// provider that lifts the system prompt out into its own field sends as its conversation.
    pub fn user_messages(&self) -> Vec<Value> {
        self.messages
            .iter()
            .filter(|message| message.role != "system")
            .map(wire_message)
            .collect()
    }

    /// The schema a structured reply must fit, if one was asked for. dspy's `response_format`.
    pub fn output_schema(&self) -> Option<&Value> {
        self.config.response_format.as_ref()
    }

    /// What two identical calls share, and what [`rollout_id`](super::LmCacheConfig::rollout_id)
    /// exists to break. Everything the provider is sent is in here — `model` included, since the
    /// store is shared across every model in the process — so no call can be answered with
    /// another's reply. `rollout_id` is in here and is *not* sent: it changes this string and
    /// nothing else, which is the whole of what it does and the mechanism behind `BestOfN`.
    ///
    /// Hashed rather than kept whole, so an entry is nameable as a file and a long conversation is
    /// not held twice. Upstream hashes the same way, `sha256(orjson.dumps(params, sort_keys))`.
    pub fn cache_key(&self, model: &str) -> String {
        use sha2::Digest;
        let identity = json!({
            "model": model,
            "messages": self.wire_messages(),
            "schema": self.output_schema(),
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "n": self.config.n,
            "rollout_id": self.config.cache.as_ref().and_then(|cache| cache.rollout_id.as_ref()),
        })
        .to_string();
        let digest = sha2::Sha256::digest(identity.as_bytes());
        digest.iter().map(|byte| format!("{byte:02x}")).collect()
    }
}

/// One message as the Chat Completions message dspy's `message_to_openai_chat` builds. Most messages
/// are a `{role, content}` pair, the parts collapsed the way a provider reads them; the two that are
/// not are an assistant turn carrying tool calls — split into a `tool_calls` field beside a content
/// that is `null` when the turn is only calls — and a tool result, which names the call it answers.
fn wire_message(message: &LmMessage) -> Value {
    let mut output = serde_json::Map::new();
    output.insert("role".to_owned(), json!(message.role));
    if let Some(name) = &message.name {
        output.insert("name".to_owned(), json!(name));
    }

    if message.role == "assistant" {
        let tool_calls: Vec<Value> = message.parts.iter().filter_map(assistant_tool_call).collect();
        if !tool_calls.is_empty() {
            let content_parts: Vec<LmPart> = message
                .parts
                .iter()
                .filter(|part| !matches!(part, LmPart::ToolCall { .. }))
                .cloned()
                .collect();
            // dspy sends `null`, not an empty block list, when the turn carries only tool calls.
            let content = match content_parts.is_empty() {
                true => Value::Null,
                false => content_value(&content_parts),
            };
            output.insert("content".to_owned(), content);
            output.insert("tool_calls".to_owned(), Value::Array(tool_calls));
            return Value::Object(output);
        }
    }

    if message.role == "tool"
        && let [LmPart::ToolResult { call_id, name, content, .. }] = message.parts.as_slice()
    {
        output.insert("content".to_owned(), content_value(content));
        if let Some(call_id) = call_id {
            output.insert("tool_call_id".to_owned(), json!(call_id));
        }
        if let Some(name) = name {
            output.insert("name".to_owned(), json!(name));
        }
        return Value::Object(output);
    }

    output.insert("content".to_owned(), content_value(&message.parts));
    Value::Object(output)
}

/// A message's parts as OpenAI `content`: the bare string of a lone text part, or a block list.
fn content_value(parts: &[LmPart]) -> Value {
    serde_json::to_value(content_of(parts).unwrap_or_else(|_| Content::Text(String::new())))
        .unwrap_or(Value::Null)
}

/// dspy's `assistant_tool_call_to_openai`: a function call, its arguments the `json.dumps` text of
/// the args — spaced the way Python writes it, which [`json_dumps`] matches — the id kept when
/// present, the provider's own data merged on. `None` for any part that is not a tool call.
fn assistant_tool_call(part: &LmPart) -> Option<Value> {
    let LmPart::ToolCall { id, name, args, provider_data, .. } = part else {
        return None;
    };
    let mut call = json!({
        "type": "function",
        "function": { "name": name, "arguments": json_dumps(&Value::Object(args.clone())) },
    });
    if let Some(id) = id {
        call["id"] = json!(id);
    }
    for (key, value) in provider_data {
        call[key] = value.clone();
    }
    Some(call)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lm::api::{LmPart, RolloutId};
    use serde_json::json;

    fn request() -> LmRequest {
        LmRequest::new(
            "openai/gpt-4o",
            vec![LmMessage::user(vec![LmPart::text("Why?")])],
        )
    }

    #[test]
    fn a_request_round_trips_through_json() {
        let request = request().configured(
            LmConfig::from_kwargs([
                ("temperature".to_owned(), json!(0.7)),
                ("rollout_id".to_owned(), json!(3)),
            ])
            .expect("builds"),
        );
        let written = serde_json::to_value(&request).expect("serializes");
        assert_eq!(written["model"], json!("openai/gpt-4o"));
        assert_eq!(written["config"]["cache"]["rollout_id"], json!(3));
        assert_eq!(
            serde_json::from_value::<LmRequest>(written).expect("parses"),
            request
        );
    }

    #[test]
    fn the_rollout_id_reads_back_from_where_it_folded_to() {
        let request = request().configured(
            LmConfig::from_kwargs([("rollout_id".to_owned(), json!(9))]).expect("builds"),
        );
        assert_eq!(request.config.rollout_id(), Some(&RolloutId::Number(9)));
    }

    #[test]
    fn a_request_forbids_what_it_does_not_declare() {
        assert!(
            serde_json::from_value::<LmRequest>(json!({
                "model": "m",
                "messages": [],
                "prompt": "the old spelling",
            }))
            .is_err()
        );
    }
}
