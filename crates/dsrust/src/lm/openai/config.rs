//! How an OpenAI-compatible endpoint is reached: the stock base URL and key variable, what a
//! caller may override, and the environment the defaults are read from.

use crate::lm::env_nonempty;
use crate::lm::token_limit::TokenLimitRule;

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
pub(crate) const BASE_URL_VAR: &str = "OPENAI_BASE_URL";

/// litellm's own spelling of the same thing, and the one its documentation uses. `completion`
/// falls back to it — `get_secret("OPENAI_BASE_URL") or get_secret("OPENAI_API_BASE")` — so a
/// shell set up by following litellm rather than the OpenAI SDK names only this one.
pub(crate) const API_BASE_VAR: &str = "OPENAI_API_BASE";

pub(crate) const OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api/v1";

pub(crate) const OPENROUTER_KEY_VAR: &str = "OPENROUTER_API_KEY";

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
    /// dspy `model_type="text"`: the legacy completions endpoint, which takes one prompt string
    /// rather than a message list. See [`text`](super::text).
    Text,
}

/// Which OpenAI-shaped service [`Provider::OpenAiCompatible`](crate::lm::routing::Provider::OpenAiCompatible)
/// talks to.
///
/// ```
/// use dsrust::lm::{DEFAULT_OPENAI_BASE_URL, DEFAULT_OPENAI_KEY_VAR, OpenAiConfig};
///
/// // The default points at OpenAI and reads the variable the OpenAI SDKs themselves read, so a
/// // shell already set up for one of them needs nothing further.
/// let openai = OpenAiConfig::default();
/// assert_eq!(openai.base_url, DEFAULT_OPENAI_BASE_URL);
/// assert_eq!(openai.key_var, DEFAULT_OPENAI_KEY_VAR);
///
/// // A local vLLM or LM Studio is the same shape at another address.
/// let local = OpenAiConfig {
///     base_url: "http://localhost:8000/v1".to_owned(),
///     ..OpenAiConfig::default()
/// };
/// assert_ne!(local.base_url, DEFAULT_OPENAI_BASE_URL);
/// ```
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
    /// The stock endpoint with the environment's base URL and OPENAI_API_KEY laid over it.
    pub(crate) fn from_env() -> Self {
        let mut config = Self::default();
        if let Some(base_url) =
            base_url_from(env_nonempty(BASE_URL_VAR), env_nonempty(API_BASE_VAR))
        {
            config.base_url = base_url;
        }
        config.api_key = env_nonempty(&config.key_var);
        config
    }
}

/// litellm's order for the two variables that name an endpoint: `OPENAI_BASE_URL`, then
/// `OPENAI_API_BASE`, then neither and the stock host.
///
/// Taken as arguments rather than read here so the rule can be checked without a process-wide
/// environment, which no two tests can share.
pub(crate) fn base_url_from(base_url: Option<String>, api_base: Option<String>) -> Option<String> {
    base_url.or(api_base)
}
