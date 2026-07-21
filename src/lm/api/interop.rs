//! Converting a typed 3.3 request and response to and from the legacy call shape.
//!
//! dspy's `BaseLM.__call__` is the compatibility boundary during the typed-LM migration: it
//! builds an `LMRequest`, lowers it to the OpenAI/LiteLLM kwargs the current provider path still
//! speaks, calls that path, then lifts the result back into an `LMResponse`
//! (`clients/base_lm.py`). This is that boundary — the one seam where the byte-faithful 3.3 types
//! meet the providers this crate already proves byte-exact, so the providers need not move first.

use serde_json::Value;

use super::wire::content_of;
use super::{LmConfig as ApiConfig, LmOutput, LmRequest as ApiRequest, LmResponse as ApiResponse};
use super::{Metadata, RolloutId};
use crate::lm::{
    ChatTurn, Content, LmConfig as CallConfig, LmRequest as CallRequest, LmResponse as CallResponse,
    OutputMode, Role,
};

/// A typed request lowered to the owned pieces the borrowing [`CallRequest`] needs.
///
/// The legacy request borrows its `system` and `turns`, so the conversion cannot hand one back
/// directly — it hands back this holder, and [`as_call`](Self::as_call) borrows it.
pub(crate) struct Lowered {
    system: String,
    turns: Vec<ChatTurn>,
    schema: Option<Value>,
    config: CallConfig,
}

impl Lowered {
    /// The legacy request, valid as long as this holder lives.
    pub(crate) fn as_call(&self) -> CallRequest<'_> {
        let mode = match &self.schema {
            Some(schema) => OutputMode::Json { schema },
            None => OutputMode::Text,
        };
        CallRequest::new(&self.system, &self.turns, mode).sampled(self.config.clone())
    }
}

/// A typed request as the `(system, turns, mode, config)` the providers already know how to send.
///
/// The first `system`-role message becomes the top-level system prompt; every other message
/// becomes a turn, its parts collapsed to [`Content`] by the same [`content_of`] the multimodal
/// path is golden-tested through, so a turn built here renders byte-identically to one built
/// natively.
pub(crate) fn lower_request(request: &ApiRequest) -> Lowered {
    let mut system = String::new();
    let mut turns = Vec::new();
    for message in &request.messages {
        let content = content_of(&message.parts).unwrap_or_else(|_| Content::Text(String::new()));
        match message.role.as_str() {
            "system" => system = content.text().unwrap_or_default().to_owned(),
            "assistant" => turns.push(ChatTurn {
                role: Role::Assistant,
                content,
            }),
            // user, and anything unrecognised, is sent as a user turn rather than dropped.
            _ => turns.push(ChatTurn {
                role: Role::User,
                content,
            }),
        }
    }
    Lowered {
        system,
        turns,
        schema: request.config.response_format.clone(),
        config: lower_config(&request.config),
    }
}

/// The four fields of the legacy config the providers read, drawn from the twelve of the typed
/// one. The rest — `top_p`, `stop`, `reasoning`, and the nested caches beyond the rollout — have
/// no legacy slot yet and wait for the trait itself to take the typed config.
fn lower_config(config: &ApiConfig) -> CallConfig {
    CallConfig {
        temperature: config.temperature,
        max_tokens: config.max_tokens,
        completions: config.n,
        rollout_id: config
            .cache
            .as_ref()
            .and_then(|cache| cache.rollout_id.as_ref())
            .and_then(|rollout| match rollout {
                RolloutId::Number(id) => u64::try_from(*id).ok(),
                // The legacy cache key numbers its rollouts; a textual one has no place in it yet.
                RolloutId::Text(_) => None,
            }),
    }
}

/// A legacy response lifted into the typed one — dspy's `_process_lm_response`.
///
/// Each completion string becomes a one-part text [`LmOutput`]; the usage carries across
/// unchanged, since both families already share [`LmUsage`](crate::lm::LmUsage). The provider's
/// own extra data, kept whole as one JSON value on the legacy side, becomes the typed side's
/// `provider_data` map when it is an object.
pub(crate) fn lift_response(response: CallResponse) -> ApiResponse {
    ApiResponse {
        outputs: response.outputs.into_iter().map(LmOutput::text).collect(),
        usage: response.usage,
        cache_hit: response.cache_hit,
        provider_data: match response.provider_data {
            Some(Value::Object(map)) => map,
            _ => Metadata::new(),
        },
        ..ApiResponse::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lm::LmUsage;
    use crate::lm::api::LmMessage;
    use crate::lm::api::part::LmPart;
    use serde_json::json;

    fn request(messages: Vec<LmMessage>, config: ApiConfig) -> ApiRequest {
        ApiRequest::new("openai/gpt-4o", messages).configured(config)
    }

    /// A system message becomes the top-level system prompt; the rest become turns under the
    /// roles they declared.
    #[test]
    fn messages_lower_to_a_system_prompt_and_turns() {
        let lowered = lower_request(&request(
            vec![
                LmMessage::system(vec![LmPart::text("Be concise.")]),
                LmMessage::user(vec![LmPart::text("Why?")]),
                LmMessage::assistant(vec![LmPart::text("Because.")]),
            ],
            ApiConfig::default(),
        ));
        let call = lowered.as_call();

        assert_eq!(call.system, "Be concise.");
        assert_eq!(call.turns.len(), 2);
        assert_eq!(call.turns[0].role, Role::User);
        assert_eq!(call.turns[0].content.text(), Some("Why?"));
        assert_eq!(call.turns[1].role, Role::Assistant);
        assert_eq!(call.turns[1].content.text(), Some("Because."));
    }

    /// The load-bearing property: a lowered request hashes identically to a natively-built one.
    ///
    /// The cache key is `sha256` over model, system, every turn, the schema and the sampling
    /// fields, so two requests with the same key are the same bytes to every provider. If this
    /// holds, wiring the typed boundary in front of `chat` cannot move a byte — which is the
    /// whole safety claim of the shim.
    #[test]
    fn a_lowered_request_hashes_identically_to_a_native_one() {
        let turns = [ChatTurn::user("Why?"), ChatTurn::assistant("Because.")];
        let native = CallRequest::new("Be concise.", &turns, OutputMode::Text);

        let typed = request(
            vec![
                LmMessage::system(vec![LmPart::text("Be concise.")]),
                LmMessage::user(vec![LmPart::text("Why?")]),
                LmMessage::assistant(vec![LmPart::text("Because.")]),
            ],
            ApiConfig::default(),
        );
        let lowered = lower_request(&typed);

        assert_eq!(
            lowered.as_call().cache_key("openai/gpt-4o"),
            native.cache_key("openai/gpt-4o"),
            "the lowered request is byte-identical on the wire to the native one"
        );
    }

    /// The same, with sampling and a schema set, since those feed the key too.
    #[test]
    fn a_lowered_request_hashes_identically_with_sampling_and_a_schema() {
        let schema = json!({ "type": "object" });
        let turns = [ChatTurn::user("Why?")];
        let native = CallRequest::new("", &turns, OutputMode::Json { schema: &schema }).sampled(
            CallConfig {
                temperature: Some(0.7),
                max_tokens: Some(256),
                completions: Some(2),
                rollout_id: Some(5),
            },
        );

        let config = ApiConfig::from_kwargs([
            ("temperature".to_owned(), json!(0.7)),
            ("max_tokens".to_owned(), json!(256)),
            ("n".to_owned(), json!(2)),
            ("rollout_id".to_owned(), json!(5)),
            ("response_format".to_owned(), schema.clone()),
        ])
        .expect("valid config");
        let lowered = lower_request(&request(
            vec![LmMessage::user(vec![LmPart::text("Why?")])],
            config,
        ));

        assert_eq!(
            lowered.as_call().cache_key("m"),
            native.cache_key("m"),
            "sampling and schema lower without moving the key"
        );
    }

    /// A plain request has no response format, so it lowers to text mode — not an empty JSON
    /// schema, which would change what every ordinary call asks for.
    #[test]
    fn a_request_with_no_response_format_is_text_mode() {
        let lowered = lower_request(&request(
            vec![LmMessage::user(vec![LmPart::text("Why?")])],
            ApiConfig::default(),
        ));
        assert!(matches!(lowered.as_call().mode, OutputMode::Text));
    }

    #[test]
    fn a_response_format_lowers_to_json_mode_carrying_the_schema() {
        let schema = json!({ "type": "object" });
        let config = ApiConfig {
            response_format: Some(schema.clone()),
            ..ApiConfig::default()
        };
        let lowered = lower_request(&request(
            vec![LmMessage::user(vec![LmPart::text("Why?")])],
            config,
        ));
        match lowered.as_call().mode {
            OutputMode::Json { schema: got } => assert_eq!(*got, schema),
            OutputMode::Text => panic!("expected json mode"),
        }
    }

    /// The four config fields the providers read cross over; a numeric rollout becomes the legacy
    /// key's number.
    #[test]
    fn the_sampling_fields_and_rollout_cross_over() {
        let config = ApiConfig::from_kwargs([
            ("temperature".to_owned(), json!(0.7)),
            ("max_tokens".to_owned(), json!(256)),
            ("n".to_owned(), json!(3)),
            ("rollout_id".to_owned(), json!(5)),
        ])
        .expect("valid config");
        let lowered = lower_request(&request(
            vec![LmMessage::user(vec![LmPart::text("Why?")])],
            config,
        ));
        let call = lowered.as_call();

        assert_eq!(call.config.temperature, Some(0.7));
        assert_eq!(call.config.max_tokens, Some(256));
        assert_eq!(call.config.completions, Some(3));
        assert_eq!(call.config.rollout_id, Some(5), "cache.rollout_id → rollout_id");
    }

    /// Each completion string becomes a text output, and the usage rides across untouched.
    #[test]
    fn a_legacy_response_lifts_to_text_outputs_keeping_usage() {
        let legacy = CallResponse::completions(["Paris".to_owned(), "Lyon".to_owned()])
            .with_usage(Some(LmUsage::counted(10, 4)));
        let lifted = lift_response(legacy);

        assert_eq!(lifted.outputs.len(), 2);
        assert_eq!(lifted.first_text(), "Paris");
        assert_eq!(lifted.outputs[1].as_text(), "Lyon");
        assert_eq!(lifted.usage.expect("usage carried").total(), Some(14));
        assert!(!lifted.cache_hit);
    }

    /// A cache replay stays a replay across the lift, so spend still reads as nothing.
    #[test]
    fn a_replayed_response_keeps_its_cache_hit() {
        let mut legacy = CallResponse::text("Paris").with_usage(Some(LmUsage::counted(10, 4)));
        legacy.cache_hit = true;
        let lifted = lift_response(legacy);

        assert!(lifted.cache_hit);
        assert_eq!(lifted.spend(), None, "a replay is not billed");
    }
}
