//! `LMRequest`.

use serde_json::{Value, json};

use super::config::LmConfig;
use super::message::{LmMessage, LmToolSpec};
use super::part::Metadata;
use super::wire::{Content, content_of};

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

/// One message as its `{role, content}` pair, the parts collapsed the way a provider reads them.
fn wire_message(message: &LmMessage) -> Value {
    json!({
        "role": &message.role,
        "content": content_of(&message.parts).unwrap_or_else(|_| Content::Text(String::new())),
    })
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
