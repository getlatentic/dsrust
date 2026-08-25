//! The OpenAI `/v1/chat/completions` wire format, which OpenAI, OpenRouter, Groq, Together,
//! vLLM and LM Studio all speak. They differ only in host, credential, and how much of
//! `response_format` they accept, so one request builder, one reply reader and one error
//! shape serve all of them rather than a copy per service.

use std::future::Future;

use anyhow::{Result, anyhow};
use serde_json::{Value, json};

use super::token_limit::TokenLimitRule;
use crate::error::Explained;
use std::time::Duration;

use super::{ChatModel, LmUsage, api, env_nonempty};

mod reasoning_temperature;
mod response;
pub mod responses;
mod tools;

pub(super) use tools::{apply_tool_choice, provider_extras, tool_json, unreadable_arguments};
mod stream;

/// OpenAI's own endpoint, and the value every other service replaces.
///
/// Named so a caller can tell "pointed somewhere else" from "left alone" without hardcoding the
/// URL — which is the question worth asking before sending a key anywhere:
///
/// ```
/// use dsrust::lm::{DEFAULT_OPENAI_BASE_URL, DEFAULT_OPENAI_KEY_VAR};
///
/// let base = std::env::var("OPENAI_BASE_URL")
///     .unwrap_or_else(|_| DEFAULT_OPENAI_BASE_URL.to_owned());
/// if base != DEFAULT_OPENAI_BASE_URL {
///     // A local vLLM or LM Studio, not OpenAI — worth knowing before reading a credential.
///     assert_ne!(DEFAULT_OPENAI_KEY_VAR, "");
/// }
/// ```
pub const DEFAULT_OPENAI_BASE_URL: &str = "https://api.openai.com/v1";

/// The credential variable the OpenAI SDKs themselves read, so a shell already set up for one
/// of them needs nothing further here.
pub const DEFAULT_OPENAI_KEY_VAR: &str = "OPENAI_API_KEY";

/// The base-URL variable the OpenAI SDKs read, which is how a local vLLM or LM Studio is
/// usually pointed at.
const BASE_URL_VAR: &str = "OPENAI_BASE_URL";

const OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api/v1";

const OPENROUTER_KEY_VAR: &str = "OPENROUTER_API_KEY";

/// Which `response_format` envelope an endpoint understands.
///
/// [`Object`](JsonFormat::Object) is the default because it is the whole of `response_format`
/// that OpenAI, Groq, Together, vLLM and LM Studio agree on: a server that does not know
/// `json_schema` rejects the entire request rather than ignoring the field, so the richer
/// envelope cannot be the portable one. The schema reaches the model either way —
/// [`crate::JsonAdapter`] writes it into the system prompt — so `Object` costs enforcement,
/// not information.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum JsonFormat {
    /// `{"type": "json_object"}`: the reply is valid JSON, shaped by the prompt alone.
    #[default]
    Object,
    /// `{"type": "json_schema", ...}` with `strict: true`, constraining decoding to the
    /// signature's schema.
    ///
    /// Strict mode makes OpenAI validate the schema before generating: every object must set
    /// `additionalProperties: false` and name all of its properties in `required`.
    /// [`Signature::schema`](crate::Signature::schema) does that at the top level, but the
    /// schema of a `json` field carrying a nested struct does not, and OpenAI rejects such a
    /// call outright — those signatures belong on [`Object`](JsonFormat::Object).
    Schema,
}

/// Which OpenAI wire an endpoint speaks: the chat-completions route every service shares, or the
/// Responses API OpenAI uses for reasoning models. dspy's `model_type`, narrowed to the two this
/// crate builds. Non-streaming only for now — a streamed call takes the chat route regardless.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OpenAiWire {
    #[default]
    Chat,
    Responses,
}

/// Which OpenAI-shaped service [`Provider::OpenAiCompatible`](super::Provider::OpenAiCompatible)
/// talks to.
#[derive(Debug, Clone)]
pub struct OpenAiConfig {
    pub base_url: String,
    pub api_key: Option<String>,
    /// The variable a missing key points the caller at, which stops being `OPENAI_API_KEY`
    /// as soon as the endpoint is pointed at Groq or Together.
    pub key_var: String,
    pub json_format: JsonFormat,
    /// Which generation-cap field this service takes. See [`TokenLimitRule`].
    pub token_limit_rule: TokenLimitRule,
    /// The chat route, or the Responses API. See [`OpenAiWire`].
    pub wire: OpenAiWire,
}

impl Default for OpenAiConfig {
    fn default() -> Self {
        Self {
            base_url: DEFAULT_OPENAI_BASE_URL.to_owned(),
            api_key: None,
            key_var: DEFAULT_OPENAI_KEY_VAR.to_owned(),
            json_format: JsonFormat::Object,
            // The stock endpoint is OpenAI's own, and pointing `base_url` elsewhere leaves
            // this rule in place on purpose: a service is chosen by the caller, not
            // inferred from a host that a proxy or gateway can freely change.
            token_limit_rule: TokenLimitRule::ByOpenAiModelFamily,
            wire: OpenAiWire::Chat,
        }
    }
}

impl OpenAiConfig {
    /// The stock endpoint with OPENAI_BASE_URL and OPENAI_API_KEY laid over it.
    pub(super) fn from_env() -> Self {
        let mut config = Self::default();
        if let Some(base_url) = env_nonempty(BASE_URL_VAR) {
            config.base_url = base_url;
        }
        config.api_key = env_nonempty(&config.key_var);
        config
    }
}

/// One OpenAI-shaped service, resolved for a single call.
pub(crate) struct Endpoint<'a> {
    model: &'a str,
    /// Leads every error with the provider prefix the model was named with.
    label: &'a str,
    base_url: &'a str,
    api_key: Option<&'a str>,
    key_var: &'a str,
    json_format: JsonFormat,
    token_limit_rule: TokenLimitRule,
    wire: OpenAiWire,
    timeout: Duration,
}

impl<'a> Endpoint<'a> {
    /// OpenRouter: its own host and credential, on the envelope it has always been sent.
    /// It accepts `max_tokens` for every model it hosts, OpenAI's reasoning models included,
    /// so the model name never moves the cap to another field here.
    pub(crate) fn openrouter(model: &'a str, api_key: Option<&'a str>, timeout: Duration) -> Self {
        Self {
            model,
            label: "openrouter",
            base_url: OPENROUTER_BASE_URL,
            api_key,
            key_var: OPENROUTER_KEY_VAR,
            json_format: JsonFormat::Object,
            token_limit_rule: TokenLimitRule::AlwaysMaxTokens,
            wire: OpenAiWire::Chat,
            timeout,
        }
    }

    /// Whatever the configuration names: OpenAI itself by default, or any other service
    /// exposing the same route.
    pub(crate) fn configured(model: &'a str, config: &'a OpenAiConfig, timeout: Duration) -> Self {
        Self {
            model,
            label: "openai",
            base_url: &config.base_url,
            api_key: config.api_key.as_deref(),
            key_var: &config.key_var,
            json_format: config.json_format,
            token_limit_rule: config.token_limit_rule,
            wire: config.wire,
            timeout,
        }
    }

    /// The streaming form of [`forward`](ChatModel::forward): the same body with the stream flags,
    /// read back as typed [`LmStreamEvent`](api::LmStreamEvent)s. Its owned inputs are lifted out
    /// of the endpoint here, so — `use<'h>` — it captures only the client and outlives this
    /// temporary endpoint the way a returned stream must.
    pub(crate) fn stream(
        &self,
        http: &reqwest::Client,
        call: &api::LmRequest,
    ) -> std::pin::Pin<
        Box<dyn futures_util::Stream<Item = Result<api::LmStreamEvent>> + Send + 'static>,
    > {
        // The body is built first because building it can refuse the call — an OpenAI reasoning
        // model asked to reason at a chosen temperature. A refusal arrives as the stream's first
        // and only item, which is where a streaming caller reads a failure from anyway.
        let built = match self.wire {
            OpenAiWire::Chat => {
                streaming_body(self.model, call, self.json_format, self.token_limit_rule)
            }
            OpenAiWire::Responses => responses::streaming_body(self.model, call, self.json_format),
        };
        let body = match built {
            Ok(body) => body,
            Err(refused) => {
                return Box::pin(futures_util::stream::once(std::future::ready(Err(refused))));
            }
        };
        match self.wire {
            OpenAiWire::Chat => Box::pin(stream::events(
                http,
                chat_completions_url(self.base_url),
                self.api_key.map(str::to_owned),
                self.label.to_owned(),
                self.model.to_owned(),
                body,
                self.timeout,
            )),
            OpenAiWire::Responses => Box::pin(responses::stream(
                http,
                responses_url(self.base_url),
                self.api_key.map(str::to_owned),
                self.label.to_owned(),
                self.model.to_owned(),
                body,
                self.timeout,
            )),
        }
    }
}

/// The request body with the streaming flags OpenAI reads: emit chunks, and put the usage in the
/// final one rather than omitting it as a streamed call otherwise would.
fn streaming_body(
    model: &str,
    call: &api::LmRequest,
    json_format: JsonFormat,
    token_limit_rule: TokenLimitRule,
) -> Result<Value> {
    let mut body = request(model, call, json_format, token_limit_rule)?;
    body["stream"] = json!(true);
    body["stream_options"] = json!({ "include_usage": true });
    Ok(body)
}

impl ChatModel for Endpoint<'_> {
    fn forward<'a>(
        &'a self,
        call: &'a api::LmRequest,
    ) -> impl Future<Output = Result<api::LmResponse>> + Send + 'a {
        async move {
            let http = &crate::lm::global::client();
            let key = self
                .api_key
                .ok_or_else(|| anyhow!("{} is not set", self.key_var))?;
            let (url, body) = match self.wire {
                OpenAiWire::Chat => (
                    chat_completions_url(self.base_url),
                    request(self.model, call, self.json_format, self.token_limit_rule)?,
                ),
                OpenAiWire::Responses => (
                    responses_url(self.base_url),
                    responses::request(self.model, call, self.json_format)?,
                ),
            };
            let response = http
                .post(url)
                .bearer_auth(key)
                .timeout(self.timeout)
                .json(&body)
                .send()
                .await
                .map_err(|error| {
                    crate::lm::LmFailure::from_transport(&error, self.model, self.label)
                })?;
            let status = response.status();
            // Taken before the body, which consumes the response — and needed for a failure, where
            // `retry-after` is what the retry waits for rather than guessing.
            let headers = response.headers().clone();
            let body: Value = response
                .json()
                .await
                .explain_with(|| format!("{} response was not JSON", self.label))?;
            match self.wire {
                OpenAiWire::Chat => {
                    response::reply(self.label, self.model, status, &headers, &body)
                }
                OpenAiWire::Responses => {
                    responses::reply(self.label, self.model, status, &headers, &body)
                }
            }
        }
    }
}

/// A base URL carrying a trailing slash names the same endpoint, and self-hosted setups are
/// routinely configured with one.
fn chat_completions_url(base_url: &str) -> String {
    format!("{}/chat/completions", base_url.trim_end_matches('/'))
}

/// The Responses API route, `/responses` off the same base URL.
fn responses_url(base_url: &str) -> String {
    format!("{}/responses", base_url.trim_end_matches('/'))
}

fn request(
    model: &str,
    call: &api::LmRequest,
    json_format: JsonFormat,
    token_limit_rule: TokenLimitRule,
) -> Result<Value> {
    // dspy validates before it builds — `common_config_kwargs` opens with this — so a refused
    // pairing never reaches a body at all.
    reasoning_temperature::checked(&call.config, model, "chat")?;
    let mut request = json!({
        "model": model,
        "messages": call.wire_messages(),
    });
    // dspy's `common_config_kwargs`: each field present only when set, in this order, so the
    // body a typed request produces is the one dspy 3.3's `to_openai_chat_request` produces.
    let config = &call.config;
    // Unknown kwargs pass straight through — dspy opens with `data = dict(config.extensions)`.
    for (key, value) in &config.extensions {
        request[key] = value.clone();
    }
    if let Some(temperature) = config.temperature {
        request["temperature"] = json!(temperature);
    }
    if let Some(top_p) = config.top_p {
        request["top_p"] = json!(top_p);
    }
    // Sent only when the caller named a cap, and under the key this model's family reads. dspy
    // omits it otherwise rather than defaulting one — a bare chat call carries no `max_tokens`.
    if let Some(max_tokens) = config.max_tokens {
        request[token_limit_rule.field_for(model).wire_name()] = json!(max_tokens);
    }
    if let Some(stop) = &config.stop
        && !stop.is_empty()
    {
        request["stop"] = json!(stop);
    }
    if let Some(logprobs) = &config.logprobs {
        request["logprobs"] = serde_json::to_value(logprobs).unwrap_or(Value::Null);
    }
    if let Some(n) = config.n {
        request["n"] = json!(n);
    }
    if let Some(schema) = call.output_schema() {
        request["response_format"] = response_format(schema, json_format);
    }
    // dspy's `reasoning_to_chat_kwargs` and `prompt_cache_to_kwargs` — only the effort and the
    // cache key/off reach the chat wire, the rest of those configs being for other endpoints.
    if let Some(effort) = config.reasoning.as_ref().and_then(|r| r.effort.as_ref()) {
        request["reasoning_effort"] = json!(effort);
    }
    if let Some(cache) = &config.prompt_cache {
        if let Some(key) = &cache.key {
            request["prompt_cache_key"] = json!(key);
        }
        if cache.enabled == Some(false) {
            request["prompt_cache"] = json!(false);
        }
    }
    // dspy adds these in `to_openai_chat_request`, after the shared config.
    if let Some(choice) = &config.tool_choice {
        apply_tool_choice(&mut request, choice);
    }
    if !call.tools.is_empty() {
        request["tools"] = Value::Array(call.tools.iter().map(tool_json).collect());
    }
    Ok(request)
}

/// See [`JsonFormat`] for why the weaker envelope is the default.
fn response_format(schema: &Value, json_format: JsonFormat) -> Value {
    match json_format {
        JsonFormat::Object => json!({ "type": "json_object" }),
        JsonFormat::Schema => json!({
            "type": "json_schema",
            "json_schema": { "name": "response", "strict": true, "schema": schema },
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::super::token_limit::TokenLimitField;
    use super::*;

    /// `from_env` reads the environment — replaced by `Default::default()` it compiled and every
    /// test passed, because every test injects its endpoint explicitly. One test, both variables,
    /// set and removed in the same function so parallel tests cannot interleave half a state.
    #[test]
    fn from_env_reads_the_base_url_and_key() {
        unsafe {
            std::env::set_var(BASE_URL_VAR, "http://env-test:1234/v1");
            std::env::set_var("OPENAI_API_KEY", "sk-env-test");
        }
        let config = OpenAiConfig::from_env();
        unsafe {
            std::env::remove_var(BASE_URL_VAR);
            std::env::remove_var("OPENAI_API_KEY");
        }
        assert_eq!(config.base_url, "http://env-test:1234/v1");
        assert_eq!(config.api_key.as_deref(), Some("sk-env-test"));
    }
    use crate::lm::api::{LmMessage, request_of};
    use crate::lm::{DEFAULT_PROVIDER_TIMEOUT, OutputMode, Sampling};

    fn schema() -> Value {
        json!({
            "type": "object",
            "properties": { "answer": { "type": "string" } },
            "required": ["answer"],
            "additionalProperties": false,
        })
    }

    fn json_request(json_format: JsonFormat) -> Value {
        let schema = schema();
        let call = request_of(
            vec![LmMessage::system(["be helpful"]), LmMessage::user(["hi"])],
            OutputMode::Json { schema: &schema },
            &Sampling::default(),
        );
        request(
            "gpt-4o-mini",
            &call,
            json_format,
            TokenLimitRule::ByOpenAiModelFamily,
        )
        .expect("the body builds")
    }

    /// The body a text-mode call to `model` produces on OpenAI's own endpoint.
    fn text_request(model: &str, token_limit_rule: TokenLimitRule) -> Value {
        sampled_request(model, token_limit_rule, Sampling::default())
    }

    /// The same body, with the caller naming how the reply should be sampled.
    fn sampled_request(model: &str, token_limit_rule: TokenLimitRule, config: Sampling) -> Value {
        let call = request_of(
            vec![LmMessage::system(["be helpful"]), LmMessage::user(["hi"])],
            OutputMode::Text,
            &config,
        );
        request(model, &call, JsonFormat::Object, token_limit_rule).expect("the body builds")
    }

    #[test]
    fn the_stock_endpoint_is_openai_on_the_portable_envelope() {
        let config = OpenAiConfig::default();
        assert_eq!(config.base_url, "https://api.openai.com/v1");
        assert_eq!(config.key_var, "OPENAI_API_KEY");
        assert_eq!(config.api_key, None);
        assert_eq!(config.json_format, JsonFormat::Object);
        assert_eq!(config.token_limit_rule, TokenLimitRule::ByOpenAiModelFamily);
    }

    /// The Responses API is opt-in and reached at its own route off the same base url; a chat
    /// endpoint is untouched by it.
    #[test]
    fn the_responses_wire_is_opt_in_and_has_its_own_route() {
        assert_eq!(
            responses_url("https://api.openai.com/v1"),
            "https://api.openai.com/v1/responses"
        );
        assert_eq!(OpenAiConfig::default().wire, OpenAiWire::Chat);
        let responses = OpenAiConfig {
            wire: OpenAiWire::Responses,
            ..OpenAiConfig::default()
        };
        assert_eq!(
            Endpoint::configured("gpt-5", &responses, DEFAULT_PROVIDER_TIMEOUT).wire,
            OpenAiWire::Responses
        );
    }

    #[test]
    fn a_trailing_slash_on_the_base_url_names_the_same_route() {
        assert_eq!(
            chat_completions_url("http://localhost:1234/v1/"),
            "http://localhost:1234/v1/chat/completions"
        );
        assert_eq!(
            chat_completions_url(DEFAULT_OPENAI_BASE_URL),
            "https://api.openai.com/v1/chat/completions"
        );
    }

    #[test]
    fn text_mode_asks_for_no_particular_response_format() {
        let request = text_request("gpt-4o-mini", TokenLimitRule::ByOpenAiModelFamily);
        assert_eq!(request.get("response_format"), None);
        assert_eq!(request["model"], "gpt-4o-mini");
        assert_eq!(request["messages"][0]["role"], "system");
        assert_eq!(request["messages"][1]["content"], "hi");
    }

    #[test]
    fn json_mode_asks_for_an_object_by_default() {
        assert_eq!(
            json_request(JsonFormat::Object)["response_format"],
            json!({ "type": "json_object" })
        );
    }

    #[test]
    fn the_schema_envelope_carries_the_schema_and_asks_for_strict_decoding() {
        let format = json_request(JsonFormat::Schema)["response_format"].clone();
        assert_eq!(format["type"], "json_schema");
        assert_eq!(format["json_schema"]["strict"], true);
        assert_eq!(format["json_schema"]["schema"], schema());
        assert!(
            format["json_schema"]["name"].is_string(),
            "openai requires a name alongside the schema, got: {format}"
        );
    }

    /// OpenRouter reaches the wire through the shared path now, so its host, credential and
    /// envelope are pinned here.
    #[test]
    fn openrouter_keeps_its_own_host_and_credential() {
        let endpoint = Endpoint::openrouter("probe", Some("key"), DEFAULT_PROVIDER_TIMEOUT);
        assert_eq!(
            chat_completions_url(endpoint.base_url),
            "https://openrouter.ai/api/v1/chat/completions"
        );
        assert_eq!(endpoint.key_var, "OPENROUTER_API_KEY");
        assert_eq!(endpoint.json_format, JsonFormat::Object);
    }

    #[test]
    fn a_configured_endpoint_reports_the_variable_it_was_told_to_read() {
        let config = OpenAiConfig {
            base_url: "https://api.groq.com/openai/v1".into(),
            api_key: None,
            key_var: "GROQ_API_KEY".into(),
            json_format: JsonFormat::Object,
            token_limit_rule: TokenLimitRule::AlwaysMaxTokens,
            wire: OpenAiWire::Chat,
        };
        let endpoint = Endpoint::configured("probe", &config, DEFAULT_PROVIDER_TIMEOUT);
        assert_eq!(endpoint.key_var, "GROQ_API_KEY");
        assert_eq!(
            chat_completions_url(endpoint.base_url),
            "https://api.groq.com/openai/v1/chat/completions"
        );
    }

    /// dspy 3.3 omits `max_tokens` when the caller named none — a bare chat call carries no cap
    /// under either key, rather than a default one this crate once invented. The routing between
    /// the two keys is what [`a_named_cap_replaces_the_default_on_whichever_key_carries_it`]
    /// covers, on the cap a caller does set.
    #[test]
    fn no_cap_sends_neither_token_key() {
        for model in ["gpt-4o-mini", "o3"] {
            let request = text_request(model, TokenLimitRule::ByOpenAiModelFamily);
            assert_eq!(request.get("max_tokens"), None, "{model}");
            assert_eq!(request.get("max_completion_tokens"), None, "{model}");
        }
    }

    /// Faithfulness to dspy 3.3's typed_lm boundary: our OpenAI body equals the one dspy 3.3's own
    /// `to_openai_chat_request` builds from the same typed request, case for case — the config
    /// mapping (which fields, present only when set, under which key) and the message rendering
    /// (a bare string, or content blocks for a multimodal turn) both. The fixture is generated by
    /// `scripts/generate_openai_wire_fixture.py` running dspy 3.3, so a divergence is dspy's word
    /// against ours, not a hand-written expectation.
    #[test]
    fn our_body_matches_dspy_33_to_openai_chat_request() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/conformance/lm_api/openai_chat.json");
        let fixture: Value =
            serde_json::from_str(&std::fs::read_to_string(path).expect("fixture is readable"))
                .expect("fixture is valid json");

        for case in fixture["cases"].as_array().expect("cases array") {
            let name = case["name"].as_str().expect("a case name");
            let call: api::LmRequest = serde_json::from_value(case["lm_request"].clone())
                .unwrap_or_else(|error| panic!("{name}: the typed request did not parse: {error}"));
            // OpenAI's own endpoint: the model-family token rule and the object envelope, which
            // are dspy's defaults for `to_openai_chat_request`.
            let body = request(
                &call.model,
                &call,
                JsonFormat::Object,
                TokenLimitRule::ByOpenAiModelFamily,
            )
            .expect("the body builds");
            assert_eq!(
                body, case["expected"],
                "{name}: our OpenAI body diverges from dspy 3.3's to_openai_chat_request"
            );
        }
    }

    /// A caller names a cap without knowing which key this endpoint and model put it under,
    /// so the override has to follow the same rule the default does rather than pick a key.
    #[test]
    fn a_named_cap_replaces_the_default_on_whichever_key_carries_it() {
        let capped = Sampling {
            max_tokens: Some(64),
            ..Sampling::default()
        };
        let rule = TokenLimitRule::ByOpenAiModelFamily;

        let chat = sampled_request("gpt-4o-mini", rule, capped.clone());
        assert_eq!(chat["max_tokens"], 64);
        assert_eq!(chat.get("max_completion_tokens"), None);

        let reasoning = sampled_request("o3", rule, capped);
        assert_eq!(reasoning["max_completion_tokens"], 64);
        assert_eq!(reasoning.get("max_tokens"), None);
    }

    /// An unset temperature has to leave the key off entirely rather than send a value this
    /// crate invented, because the provider's own default is what the caller asked for.
    #[test]
    fn temperature_is_sent_only_when_the_caller_names_one() {
        let rule = TokenLimitRule::ByOpenAiModelFamily;
        let default = text_request("gpt-4o-mini", rule);
        assert_eq!(default.get("temperature"), None);

        let warmed = sampled_request(
            "gpt-4o-mini",
            rule,
            Sampling {
                temperature: Some(1.0),
                ..Sampling::default()
            },
        );
        assert_eq!(warmed["temperature"], 1.0);
    }

    /// OpenRouter's body is pinned whole. It reaches the wire through the shared builder,
    /// so a model name that OpenAI treats specially must still leave these exact bytes.
    #[test]
    fn openrouter_sends_every_model_on_the_max_tokens_envelope() {
        let endpoint = Endpoint::openrouter("probe", Some("key"), DEFAULT_PROVIDER_TIMEOUT);
        // A named cap, so the routing this test is about has something to place; OpenRouter's
        // rule keeps every model on `max_tokens`, its reasoning-named ones included.
        let capped = Sampling {
            max_tokens: Some(1024),
            ..Sampling::default()
        };
        for model in ["openai/gpt-5", "openai/o3", "openai/gpt-oss-120b"] {
            let call = request_of(
                vec![LmMessage::system(["be helpful"]), LmMessage::user(["hi"])],
                OutputMode::Text,
                &capped,
            );
            let body = request(
                model,
                &call,
                endpoint.json_format,
                endpoint.token_limit_rule,
            )
            .expect("the body builds");
            assert_eq!(
                body.to_string(),
                format!(
                    r#"{{"model":"{model}","messages":[{{"role":"system","content":"be helpful"}},{{"role":"user","content":"hi"}}],"max_tokens":1024}}"#
                )
            );
        }
    }

    /// The rule is a property of the endpoint, and reading it back is how a caller checks
    /// which field a service will be sent.
    #[test]
    fn each_endpoint_reports_the_token_limit_rule_it_follows() {
        assert_eq!(
            Endpoint::openrouter("probe", Some("key"), DEFAULT_PROVIDER_TIMEOUT).token_limit_rule,
            TokenLimitRule::AlwaysMaxTokens
        );
        let config = OpenAiConfig::default();
        assert_eq!(
            Endpoint::configured("probe", &config, DEFAULT_PROVIDER_TIMEOUT)
                .token_limit_rule
                .field_for("o3"),
            TokenLimitField::MaxCompletionTokens
        );
    }

    /// Asking for several completions is one round trip where re-asking would be several, so `n`
    /// has to reach the wire. Reading every choice back is [`response`]'s job, verified there.
    #[test]
    fn asking_for_several_completions_sends_n() {
        let asked = sampled_request(
            "gpt-4o-mini",
            TokenLimitRule::ByOpenAiModelFamily,
            Sampling {
                completions: Some(3),
                ..Sampling::default()
            },
        );
        assert_eq!(asked["n"], 3);
    }

    /// Unset means the field is left off, so a service that rejects `n` is untouched by a
    /// caller who never asked for several.
    #[test]
    fn no_n_is_sent_when_none_was_asked_for() {
        let asked = text_request("gpt-4o-mini", TokenLimitRule::ByOpenAiModelFamily);
        assert_eq!(asked.get("n"), None);
    }
}
