//! `LMRequest`.

use serde_json::{Value, json};

use super::config::{LmCacheConfig, LmConfig, RolloutId};
use super::message::{LmMessage, LmToolSpec};
use super::part::{LmPart, Metadata};
use super::wire::{Content, content_of};
use crate::adapter::python_json::json_dumps;
use crate::lm::{OutputMode, Sampling};

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
    /// dspy `LMRequest.from_call(model=…, messages=…)`: the door an adapter comes through.
    ///
    /// A signature renders to messages and enters here; a caller writing turns by hand enters
    /// through [`from_items`](Self::from_items). Upstream is one constructor that raises when both
    /// are passed, which two constructors make unwritable instead.
    pub fn from_messages(model: impl Into<String>, messages: Vec<LmMessage>) -> Self {
        Self::new(model, messages)
    }

    /// dspy `LMRequest.from_call(model=…, items=…)`: the direct-call door, for a caller who wants
    /// the model without a signature.
    ///
    /// ```
    /// # use dsrust::{LmMessage, LmRequest};
    /// LmRequest::from_items("openai/gpt-4o-mini", ["What is the capital of France?"]);
    /// LmRequest::from_items("openai/gpt-4o-mini", [LmMessage::user(["Hello?"])]);
    /// ```
    ///
    /// What a run of items means is [`messages_from_items`](super::messages_from_items).
    pub fn from_items(
        model: impl Into<String>,
        items: impl IntoIterator<Item = impl Into<super::LmItem>>,
    ) -> anyhow::Result<Self> {
        Ok(Self::new(model, super::messages_from_items(items)?))
    }

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

    pub fn tools(mut self, tools: Vec<LmToolSpec>) -> Self {
        self.tools = tools;
        self
    }

    /// The top-level system prompt: the text of the first `system`-role message, or `""` when
    /// there is none. Read as a bare string, so a provider that carries the system prompt in its
    /// own field — Anthropic — reads the same bytes a native call would have carried.
    pub fn system(&self) -> &str {
        super::system_of(&self.messages)
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

    /// This request as the `inputs` dspy's `on_lm_start` is handed: the messages, and the settings
    /// they are being asked under.
    ///
    /// The messages whole, because the prompt is the thing a reader opened a trace to see. Not the
    /// credentials — they are on the model rather than the request, so there is nothing here to
    /// redact, and that is worth knowing before anyone adds a field.
    pub fn watchable(&self) -> String {
        serde_json::json!({
            "messages": self.wire_messages(),
            "config": self.config,
        })
        .to_string()
    }

    /// What two identical calls share, and what [`rollout_id`](super::LmCacheConfig::rollout_id)
    /// exists to break. Everything the provider is sent is in here — `model` included, since the
    /// store is shared across every model in the process — so no call can be answered with
    /// another's reply. `rollout_id` is in here and is *not* sent: it changes this string and
    /// nothing else, which is the whole of what it does and the mechanism behind `BestOfN`.
    ///
    /// Hashed rather than kept whole, so an entry is nameable as a file and a long conversation is
    /// not held twice. Upstream hashes the same way, `sha256(orjson.dumps(params, sort_keys))`.
    ///
    /// The *whole* request goes in, as upstream's whole kwargs dict does. A key over a chosen
    /// subset is the dangerous kind of wrong: two calls differing only in a field the subset misses
    /// share an entry, and the second is answered with the first's reply. Tools were exactly that
    /// — add a tool to a program and it replayed an answer from before the tool existed.
    pub fn cache_key(&self, model: &str) -> String {
        let mut identity =
            serde_json::to_value(self).unwrap_or_else(|_| json!({ "unserializable": true }));
        // The store is shared across every model in the process, so the model this call is going to
        // is part of what makes it that call — and the caller's is authoritative over the field.
        identity["model"] = json!(model);
        crate::lm::cache::key_of(&identity)
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
        let tool_calls: Vec<Value> = message
            .parts
            .iter()
            .filter_map(assistant_tool_call)
            .collect();
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
        && let [
            LmPart::ToolResult {
                call_id,
                name,
                content,
                ..
            },
        ] = message.parts.as_slice()
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
    let LmPart::ToolCall {
        id,
        name,
        args,
        provider_data,
        ..
    } = part
    else {
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

/// A rendered conversation as the request that carries it. The model is left blank — this crate
/// keeps it on the [`LM`](crate::lm::LM) rather than in the request, so the provider fills it and
/// the cache key is taken against it separately.
pub(crate) fn request_of(
    messages: Vec<LmMessage>,
    mode: OutputMode,
    config: &Sampling,
) -> LmRequest {
    LmRequest::new("", messages).configured(config_of(mode, config))
}

/// The four sampling fields as the typed config, the rollout back under its nested cache. The mode
/// becomes the response format, the one place `OutputMode` lives in the config rather than beside
/// it.
fn config_of(mode: OutputMode, config: &Sampling) -> LmConfig {
    LmConfig {
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
        ..LmConfig::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lm::api::{LmPart, RolloutId};
    use serde_json::json;

    /// The mode and the four sampling fields cross into the typed config, a numeric rollout folded
    /// under the nested cache where the key reads it.
    #[test]
    fn the_mode_and_sampling_fields_cross_into_the_typed_config() {
        let schema = json!({ "type": "object" });
        let built = request_of(
            vec![LmMessage::user(["Why?"])],
            OutputMode::Json { schema: &schema },
            &Sampling {
                temperature: Some(0.7),
                max_tokens: Some(256),
                completions: Some(3),
                rollout_id: Some(5),
            },
        );

        assert_eq!(built.config.temperature, Some(0.7));
        assert_eq!(built.config.max_tokens, Some(256));
        assert_eq!(built.config.n, Some(3));
        assert_eq!(built.output_schema(), Some(&schema));
        assert_eq!(built.config.rollout_id(), Some(&RolloutId::Number(5)));
    }

    /// A plain render carries no response format, so it asks in text mode — not with an empty
    /// schema, which would change what every ordinary call asks for.
    #[test]
    fn a_render_with_no_schema_asks_for_no_response_format() {
        let built = request_of(Vec::new(), OutputMode::Text, &Sampling::default());
        assert_eq!(built.output_schema(), None);
    }

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

#[cfg(test)]
mod cache_key_tests {
    use super::*;
    use crate::lm::api::{LmPart, LmToolSpec, RolloutId};

    fn asking() -> LmRequest {
        LmRequest::new(
            "openai/gpt-4o",
            vec![LmMessage::user(vec![LmPart::text("Why?")])],
        )
    }

    fn key(request: &LmRequest) -> String {
        request.cache_key("openai/gpt-4o")
    }

    /// Every field of the request participates. A key over a chosen subset means two calls that
    /// differ in a missed field share an entry, and the second is answered with the first's reply —
    /// a wrong answer rather than a wasted one.
    #[test]
    fn every_field_that_changes_the_call_changes_the_key() {
        let plain = key(&asking());

        let mut with_tool = asking();
        let mut parameters = Metadata::new();
        parameters.insert("type".to_owned(), json!("object"));
        with_tool.tools = vec![LmToolSpec::new("search", parameters)];
        assert_ne!(key(&with_tool), plain, "a tool changes the call");

        /// A config knob's name and the change that sets it.
        type ConfigChange = (&'static str, fn(&mut LmConfig));

        let variants: [ConfigChange; 6] = [
            ("temperature", |config| config.temperature = Some(0.7)),
            ("top_p", |config| config.top_p = Some(0.9)),
            ("stop", |config| config.stop = Some(vec!["END".to_owned()])),
            ("max_tokens", |config| config.max_tokens = Some(64)),
            ("n", |config| config.n = Some(3)),
            ("response_format", |config| {
                config.response_format = Some(json!({ "type": "json_object" }))
            }),
        ];
        for (name, set) in variants {
            let mut request = asking();
            set(&mut request.config);
            assert_ne!(key(&request), plain, "{name} changes the call");
        }

        // A provider knob the crate does not model still reaches the wire, so it still counts.
        let mut extended = asking();
        extended
            .config
            .extensions
            .insert("seed".to_owned(), json!(11));
        assert_ne!(key(&extended), plain, "an extension changes the call");
    }

    /// dspy's `rollout_id` is the one field that is *not* sent and still changes the key — that is
    /// the whole of what it does, and the mechanism behind BestOfN and a bootstrap round after the
    /// first.
    #[test]
    fn a_rollout_changes_the_key_without_reaching_the_provider() {
        let plain = key(&asking());
        let mut rolled = asking();
        rolled.config.cache = Some(crate::lm::api::LmCacheConfig {
            rollout_id: Some(RolloutId::Number(2)),
            ..Default::default()
        });
        assert_ne!(key(&rolled), plain);
    }

    /// The model is part of the key, since one store serves every model in the process.
    #[test]
    fn two_models_do_not_share_an_entry() {
        let request = asking();
        assert_ne!(
            request.cache_key("openai/gpt-4o"),
            request.cache_key("openai/gpt-4o-mini")
        );
    }

    /// The crate builds serde_json with `preserve_order`, so equal requests assembled by different
    /// paths can hold the same keys in different orders. They are the same call and must not each
    /// pay for the answer.
    #[test]
    fn key_order_within_a_value_does_not_split_an_entry() {
        let mut one = asking();
        one.config.response_format = Some(json!({ "type": "json_object", "strict": true }));
        let mut other = asking();
        other.config.response_format = Some(json!({ "strict": true, "type": "json_object" }));
        assert_eq!(key(&one), key(&other));
    }

    /// The same call asked twice is one entry.
    #[test]
    fn the_same_call_keys_the_same() {
        assert_eq!(key(&asking()), key(&asking()));
    }
}
