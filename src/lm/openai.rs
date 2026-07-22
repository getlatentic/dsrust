//! The OpenAI `/v1/chat/completions` wire format, which OpenAI, OpenRouter, Groq, Together,
//! vLLM and LM Studio all speak. They differ only in host, credential, and how much of
//! `response_format` they accept, so one request builder, one reply reader and one error
//! shape serve all of them rather than a copy per service.

use anyhow::{Context, Result, anyhow};
use serde_json::{Value, json};

use super::token_limit::TokenLimitRule;
use super::{LmUsage, PROVIDER_TIMEOUT, api, env_nonempty};

/// OpenAI's own endpoint, and the value every other service replaces.
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
pub(super) struct Endpoint<'a> {
    /// Leads every error with the provider prefix the model was named with.
    label: &'a str,
    base_url: &'a str,
    api_key: Option<&'a str>,
    key_var: &'a str,
    json_format: JsonFormat,
    token_limit_rule: TokenLimitRule,
}

impl<'a> Endpoint<'a> {
    /// OpenRouter: its own host and credential, on the envelope it has always been sent.
    /// It accepts `max_tokens` for every model it hosts, OpenAI's reasoning models included,
    /// so the model name never moves the cap to another field here.
    pub(super) fn openrouter(api_key: Option<&'a str>) -> Self {
        Self {
            label: "openrouter",
            base_url: OPENROUTER_BASE_URL,
            api_key,
            key_var: OPENROUTER_KEY_VAR,
            json_format: JsonFormat::Object,
            token_limit_rule: TokenLimitRule::AlwaysMaxTokens,
        }
    }

    /// Whatever the configuration names: OpenAI itself by default, or any other service
    /// exposing the same route.
    pub(super) fn configured(config: &'a OpenAiConfig) -> Self {
        Self {
            label: "openai",
            base_url: &config.base_url,
            api_key: config.api_key.as_deref(),
            key_var: &config.key_var,
            json_format: config.json_format,
            token_limit_rule: config.token_limit_rule,
        }
    }

    pub(super) async fn chat(
        &self,
        http: &reqwest::Client,
        model: &str,
        call: &api::LmRequest,
    ) -> Result<api::LmResponse> {
        let key = self
            .api_key
            .ok_or_else(|| anyhow!("{} is not set", self.key_var))?;
        let response = http
            .post(chat_completions_url(self.base_url))
            .bearer_auth(key)
            .timeout(PROVIDER_TIMEOUT)
            .json(&request(
                model,
                call,
                self.json_format,
                self.token_limit_rule,
            ))
            .send()
            .await
            .with_context(|| format!("{} request failed", self.label))?;
        let status = response.status();
        let body: Value = response
            .json()
            .await
            .with_context(|| format!("{} response was not JSON", self.label))?;
        reply(self.label, model, status, &body)
    }
}

/// A base URL carrying a trailing slash names the same endpoint, and self-hosted setups are
/// routinely configured with one.
fn chat_completions_url(base_url: &str) -> String {
    format!("{}/chat/completions", base_url.trim_end_matches('/'))
}

fn request(
    model: &str,
    call: &api::LmRequest,
    json_format: JsonFormat,
    token_limit_rule: TokenLimitRule,
) -> Value {
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
    request
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

/// The reply text, or the message the service itself gave for refusing the call.
fn reply(
    label: &str,
    model: &str,
    status: reqwest::StatusCode,
    body: &Value,
) -> Result<api::LmResponse> {
    if !status.is_success() {
        let detail = body["error"]["message"].as_str().unwrap_or("unknown error");
        return Err(anyhow!("{label} {status}: {detail}"));
    }
    let outputs: Vec<String> = body["choices"]
        .as_array()
        .map(|choices| {
            choices
                .iter()
                .filter_map(|choice| choice["message"]["content"].as_str())
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();
    if outputs.is_empty() {
        return Err(anyhow!("{label} returned no content"));
    }
    Ok(api::LmResponse::completions(outputs)
        .with_usage(usage(&body["usage"]))
        .with_provider_response(provider_data(body))
        .with_model(model))
}

/// `prompt_tokens` already includes whatever was read from cache here, unlike Anthropic's split
/// counters, so the two fields are the whole of it.
fn usage(usage: &Value) -> Option<LmUsage> {
    let input = usage["prompt_tokens"].as_u64();
    let output = usage["completion_tokens"].as_u64();
    // A count the provider omitted stays unknown rather than becoming zero, which is what
    // optional counters buy: reporting one of the two is now sayable.
    (input.is_some() || output.is_some()).then(|| {
        LmUsage {
            input_tokens: input.map(|count| count as u32),
            output_tokens: output.map(|count| count as u32),
            ..LmUsage::default()
        }
        .fill_aliases()
    })
}

/// `finish_reason` is this format's name for why generation stopped — `length` where Anthropic
/// says `max_tokens`. It is left under the name the service used rather than translated, because
/// a caller reading it is already reading one provider's vocabulary.
fn provider_data(body: &Value) -> Option<Value> {
    let finish_reason = body["choices"][0]["finish_reason"].as_str()?;
    Some(json!({ "finish_reason": finish_reason }))
}

#[cfg(test)]
mod tests {
    use super::super::token_limit::TokenLimitField;
    use super::*;
    use crate::lm::api::interop::raise_request;
    use crate::lm::{ChatTurn, LmConfig, OutputMode};

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
        let call = raise_request(
            "be helpful",
            &[ChatTurn::user("hi")],
            OutputMode::Json { schema: &schema },
            &LmConfig::default(),
        );
        request(
            "gpt-4o-mini",
            &call,
            json_format,
            TokenLimitRule::ByOpenAiModelFamily,
        )
    }

    /// The body a text-mode call to `model` produces on OpenAI's own endpoint.
    fn text_request(model: &str, token_limit_rule: TokenLimitRule) -> Value {
        sampled_request(model, token_limit_rule, LmConfig::default())
    }

    /// The same body, with the caller naming how the reply should be sampled.
    fn sampled_request(model: &str, token_limit_rule: TokenLimitRule, config: LmConfig) -> Value {
        let call = raise_request("be helpful", &[ChatTurn::user("hi")], OutputMode::Text, &config);
        request(model, &call, JsonFormat::Object, token_limit_rule)
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
        let endpoint = Endpoint::openrouter(Some("key"));
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
        };
        let endpoint = Endpoint::configured(&config);
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
            );
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
        let capped = LmConfig {
            max_tokens: Some(64),
            ..LmConfig::default()
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
            LmConfig {
                temperature: Some(1.0),
                ..LmConfig::default()
            },
        );
        assert_eq!(warmed["temperature"], 1.0);
    }

    /// OpenRouter's body is pinned whole. It reaches the wire through the shared builder,
    /// so a model name that OpenAI treats specially must still leave these exact bytes.
    #[test]
    fn openrouter_sends_every_model_on_the_max_tokens_envelope() {
        let endpoint = Endpoint::openrouter(Some("key"));
        // A named cap, so the routing this test is about has something to place; OpenRouter's
        // rule keeps every model on `max_tokens`, its reasoning-named ones included.
        let capped = LmConfig {
            max_tokens: Some(1024),
            ..LmConfig::default()
        };
        for model in ["openai/gpt-5", "openai/o3", "openai/gpt-oss-120b"] {
            let call = raise_request("be helpful", &[ChatTurn::user("hi")], OutputMode::Text, &capped);
            let body = request(
                model,
                &call,
                endpoint.json_format,
                endpoint.token_limit_rule,
            );
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
            Endpoint::openrouter(Some("key")).token_limit_rule,
            TokenLimitRule::AlwaysMaxTokens
        );
        let config = OpenAiConfig::default();
        assert_eq!(
            Endpoint::configured(&config)
                .token_limit_rule
                .field_for("o3"),
            TokenLimitField::MaxCompletionTokens
        );
    }

    #[test]
    fn a_reply_is_read_from_the_first_choice() {
        let body = json!({ "choices": [{ "message": { "content": "hello" } }] });
        assert_eq!(
            reply("openai", "gpt-4o-mini", reqwest::StatusCode::OK, &body)
                .expect("a reply")
                .first_text(),
            "hello"
        );
    }

    /// This format names its counts for the two halves of the call rather than for the tokens,
    /// and `prompt_tokens` is already the whole input — unlike Anthropic's split counters.
    #[test]
    fn the_prompt_and_completion_counts_become_the_shared_usage() {
        let body = json!({
            "choices": [{ "message": { "content": "hello" }, "finish_reason": "length" }],
            "usage": { "prompt_tokens": 26, "completion_tokens": 298, "total_tokens": 324 },
        });
        let answered =
            reply("openai", "gpt-4o-mini", reqwest::StatusCode::OK, &body).expect("a reply");
        let usage = answered.usage.expect("counts");
        assert_eq!(usage.input_tokens, Some(26));
        assert_eq!(usage.output_tokens, Some(298));
        assert_eq!(usage.total(), Some(324), "the service's own total agrees");
        assert_eq!(
            answered
                .provider_response
                .expect("a finish reason")["finish_reason"],
            "length"
        );
    }

    /// Asking for several is one round trip where re-asking would be several, so `n` has to
    /// reach the wire and every choice has to be read back rather than only the first.
    #[test]
    fn asking_for_several_completions_sends_n_and_reads_every_choice() {
        let asked = sampled_request(
            "gpt-4o-mini",
            TokenLimitRule::ByOpenAiModelFamily,
            LmConfig {
                completions: Some(3),
                ..LmConfig::default()
            },
        );
        assert_eq!(asked["n"], 3);

        let body = json!({ "choices": [
            { "message": { "content": "first" } },
            { "message": { "content": "second" } },
            { "message": { "content": "third" } },
        ]});
        let answered =
            reply("openai", "gpt-4o-mini", reqwest::StatusCode::OK, &body).expect("a reply");
        let texts: Vec<String> = answered.outputs.iter().map(api::LmOutput::as_text).collect();
        assert_eq!(texts, ["first", "second", "third"]);
        assert_eq!(
            answered.first_text(),
            "first",
            "an adapter parses the first and the rest stay available"
        );
    }

    /// Unset means the field is left off, so a service that rejects `n` is untouched by a
    /// caller who never asked for several.
    #[test]
    fn no_n_is_sent_when_none_was_asked_for() {
        let asked = text_request("gpt-4o-mini", TokenLimitRule::ByOpenAiModelFamily);
        assert_eq!(asked.get("n"), None);
    }

    #[test]
    fn a_reply_reporting_no_counts_reports_no_usage() {
        let body = json!({ "choices": [{ "message": { "content": "hello" } }] });
        assert_eq!(
            reply("openai", "gpt-4o-mini", reqwest::StatusCode::OK, &body)
                .expect("a reply")
                .usage,
            None
        );
    }

    #[test]
    fn a_refused_call_carries_the_status_and_the_services_own_message() {
        let body = json!({ "error": { "message": "Incorrect API key provided" } });
        let error = reply("openai", "gpt-4o-mini", reqwest::StatusCode::UNAUTHORIZED, &body)
            .expect_err("401 is a failure");
        assert!(error.to_string().contains("openai 401"), "got: {error}");
        assert!(
            error.to_string().contains("Incorrect API key provided"),
            "got: {error}"
        );
    }

    #[test]
    fn a_success_carrying_no_content_is_an_error_rather_than_an_empty_reply() {
        let error = reply(
            "openai",
            "gpt-4o-mini",
            reqwest::StatusCode::OK,
            &json!({ "choices": [] }),
        )
        .expect_err("nothing to read");
        assert!(
            error.to_string().contains("openai returned no content"),
            "got: {error}"
        );
    }
}
