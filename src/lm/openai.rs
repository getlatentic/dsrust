//! The OpenAI `/v1/chat/completions` wire format, which OpenAI, OpenRouter, Groq, Together,
//! vLLM and LM Studio all speak. They differ only in host, credential, and how much of
//! `response_format` they accept, so one request builder, one reply reader and one error
//! shape serve all of them rather than a copy per service.

use anyhow::{Context, Result, anyhow};
use serde_json::{Value, json};

use super::token_limit::TokenLimitRule;
use super::{ChatTurn, OutputMode, PROVIDER_TIMEOUT, env_nonempty, wire_messages};

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

/// The cap every OpenAI-shaped call is sent, large enough for a full structured reply
/// without inviting an unbounded one.
const MAX_OUTPUT_TOKENS: u32 = 1024;

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
        system: &str,
        turns: &[ChatTurn],
        mode: &OutputMode<'_>,
    ) -> Result<String> {
        let key = self
            .api_key
            .ok_or_else(|| anyhow!("{} is not set", self.key_var))?;
        let response = http
            .post(chat_completions_url(self.base_url))
            .bearer_auth(key)
            .timeout(PROVIDER_TIMEOUT)
            .json(&request(
                model,
                system,
                turns,
                mode,
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
        reply(self.label, status, &body)
    }
}

/// A base URL carrying a trailing slash names the same endpoint, and self-hosted setups are
/// routinely configured with one.
fn chat_completions_url(base_url: &str) -> String {
    format!("{}/chat/completions", base_url.trim_end_matches('/'))
}

fn request(
    model: &str,
    system: &str,
    turns: &[ChatTurn],
    mode: &OutputMode<'_>,
    json_format: JsonFormat,
    token_limit_rule: TokenLimitRule,
) -> Value {
    let mut request = json!({
        "model": model,
        "messages": wire_messages(system, turns),
    });
    request[token_limit_rule.field_for(model).wire_name()] = json!(MAX_OUTPUT_TOKENS);
    if let OutputMode::Json { schema } = mode {
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
fn reply(label: &str, status: reqwest::StatusCode, body: &Value) -> Result<String> {
    if !status.is_success() {
        let detail = body["error"]["message"].as_str().unwrap_or("unknown error");
        return Err(anyhow!("{label} {status}: {detail}"));
    }
    body["choices"][0]["message"]["content"]
        .as_str()
        .map(str::to_owned)
        .with_context(|| format!("{label} returned no content"))
}

#[cfg(test)]
mod tests {
    use super::super::token_limit::TokenLimitField;
    use super::*;

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
        request(
            "gpt-4o-mini",
            "be helpful",
            &[ChatTurn::user("hi")],
            &OutputMode::Json { schema: &schema },
            json_format,
            TokenLimitRule::ByOpenAiModelFamily,
        )
    }

    /// The body a text-mode call to `model` produces on OpenAI's own endpoint.
    fn text_request(model: &str, token_limit_rule: TokenLimitRule) -> Value {
        request(
            model,
            "be helpful",
            &[ChatTurn::user("hi")],
            &OutputMode::Text,
            JsonFormat::Object,
            token_limit_rule,
        )
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
        assert_eq!(request["max_tokens"], 1024);
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

    /// The bug this guards: OpenAI's reasoning models reject `max_tokens` outright, so the
    /// cap has to move to the other key rather than be sent alongside it.
    #[test]
    fn a_reasoning_model_caps_completion_tokens_and_sends_no_max_tokens() {
        let request = text_request("o3", TokenLimitRule::ByOpenAiModelFamily);
        assert_eq!(request["max_completion_tokens"], MAX_OUTPUT_TOKENS);
        assert_eq!(request.get("max_tokens"), None);
    }

    #[test]
    fn a_chat_model_caps_max_tokens_and_sends_no_completion_tokens() {
        let request = text_request("gpt-4o-mini", TokenLimitRule::ByOpenAiModelFamily);
        assert_eq!(request["max_tokens"], MAX_OUTPUT_TOKENS);
        assert_eq!(request.get("max_completion_tokens"), None);
    }

    /// OpenRouter's body is pinned whole. It reaches the wire through the shared builder,
    /// so a model name that OpenAI treats specially must still leave these exact bytes.
    #[test]
    fn openrouter_sends_every_model_on_the_max_tokens_envelope() {
        let endpoint = Endpoint::openrouter(Some("key"));
        for model in ["openai/gpt-5", "openai/o3", "openai/gpt-oss-120b"] {
            let body = request(
                model,
                "be helpful",
                &[ChatTurn::user("hi")],
                &OutputMode::Text,
                endpoint.json_format,
                endpoint.token_limit_rule,
            );
            assert_eq!(
                body.to_string(),
                format!(
                    r#"{{"max_tokens":1024,"messages":[{{"content":"be helpful","role":"system"}},{{"content":"hi","role":"user"}}],"model":"{model}"}}"#
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
            reply("openai", reqwest::StatusCode::OK, &body).expect("a reply"),
            "hello"
        );
    }

    #[test]
    fn a_refused_call_carries_the_status_and_the_services_own_message() {
        let body = json!({ "error": { "message": "Incorrect API key provided" } });
        let error = reply("openai", reqwest::StatusCode::UNAUTHORIZED, &body)
            .expect_err("401 is a failure");
        assert!(error.to_string().contains("openai 401"), "got: {error}");
        assert!(
            error.to_string().contains("Incorrect API key provided"),
            "got: {error}"
        );
    }

    #[test]
    fn a_success_carrying_no_content_is_an_error_rather_than_an_empty_reply() {
        let error = reply("openai", reqwest::StatusCode::OK, &json!({ "choices": [] }))
            .expect_err("nothing to read");
        assert!(
            error.to_string().contains("openai returned no content"),
            "got: {error}"
        );
    }
}
