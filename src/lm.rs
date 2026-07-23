pub mod api;
mod anthropic;
pub mod cache;
mod call;
pub mod dummy;
pub mod global;
mod ollama;
mod openai;
mod streaming;
mod token_limit;
pub mod usage;

use std::time::Duration;

use anyhow::{Result, anyhow};
use futures_util::Stream;
use serde_json::Value;

pub use cache::{Cached, ResponseCache};
pub use call::{LmConfig, LmUsage};
pub use global::{configure, configure_with_client};
pub use api::{Content, Detail, LmPart, LmSource};
pub use openai::{DEFAULT_OPENAI_BASE_URL, DEFAULT_OPENAI_KEY_VAR, JsonFormat, OpenAiConfig, OpenAiWire};
pub use token_limit::{TokenLimitField, TokenLimitRule};
pub use usage::{UsageTracker, track as track_usage};

/// Bound every provider call, so one slow upstream cannot hold a worker for the whole request
/// timeout while the agent's in-flight slots stay occupied.
const PROVIDER_TIMEOUT: Duration = Duration::from_secs(20);

/// The stock ollama port on the local machine, shared with the server's config so `LM::new`
/// and the server resolve the same host when OLLAMA_HOST is unset.
pub const DEFAULT_OLLAMA_HOST: &str = "http://localhost:11434";

/// Where a model runs. Each provider keeps its own wire format behind the one `LM` interface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    Anthropic,
    OpenRouter,
    /// Any service exposing OpenAI's `/v1/chat/completions`: OpenAI itself, Groq, Together,
    /// vLLM, LM Studio. Which one is a matter of [`OpenAiConfig`] rather than of the model
    /// prefix, since they are one wire format on different hosts.
    OpenAiCompatible,
    Ollama,
}

/// A LiteLLM-style model reference, `provider/model-id`: `anthropic/claude-opus-4-8`,
/// `openrouter/openai/gpt-oss-120b`, `openai/gpt-4o-mini`, `ollama/qwen2.5:7b-instruct`.
/// The id may itself contain slashes (OpenRouter namespaces models by vendor).
#[derive(Debug, Clone)]
pub struct ModelRef {
    pub provider: Provider,
    pub id: String,
}

impl ModelRef {
    pub fn parse(raw: &str) -> Result<Self> {
        let (prefix, id) = raw
            .split_once('/')
            .ok_or_else(|| anyhow!("model must be provider/model-id, got {raw:?}"))?;
        if id.is_empty() {
            return Err(anyhow!("model id is empty in {raw:?}"));
        }
        let provider = match prefix {
            "anthropic" => Provider::Anthropic,
            "openrouter" => Provider::OpenRouter,
            "openai" => Provider::OpenAiCompatible,
            "ollama" => Provider::Ollama,
            other => return Err(anyhow!("unknown provider {other:?} in {raw:?}")),
        };
        Ok(Self {
            provider,
            id: id.to_owned(),
        })
    }
}

/// One side of the conversation; every provider speaks the user/assistant pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
}

impl Role {
    /// The name this role travels under on the wire, and the one dspy's message dicts use.
    pub fn as_str(self) -> &'static str {
        match self {
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

/// One turn of the conversation. Retries append the model's own previous reply as an
/// assistant turn followed by a corrective user turn, so the model sees what it wrote.
#[derive(Debug, Clone)]
pub struct ChatTurn {
    pub role: Role,
    pub content: Content,
}

impl ChatTurn {
    pub fn user(content: impl Into<Content>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<Content>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
        }
    }
}

/// What the reply should look like on the wire. `Json` engages the provider's native
/// structured output (Anthropic json_schema, OpenRouter/ollama JSON mode); `Text` leaves
/// the reply free-form for marker-based adapters, which need no provider support at all.
#[derive(Debug, Clone, Copy)]
pub enum OutputMode<'a> {
    Text,
    Json { schema: &'a Value },
}

/// The object-safe form of [`ChatModel`], so a model can be stored behind a pointer.
///
/// `ChatModel` returns `impl Future`, which is ergonomic to implement and impossible to make
/// into a trait object. Every `ChatModel` gets this for free through the blanket impl below,
/// and the global configuration stores this form — which is what lets a test install a
/// scripted model the way dspy installs a `DummyLM`.
pub trait DynChatModel: Send + Sync {
    /// The object-safe form of [`ChatModel::forward`] — the typed 3.3 boundary behind a pointer,
    /// which is how a module reaching its model through `dyn DynChatModel` asks it.
    fn forward_dyn<'a>(
        &'a self,
        http: &'a reqwest::Client,
        request: &'a api::LmRequest,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<api::LmResponse>> + Send + 'a>>;
}

impl<T: ChatModel + Send + Sync> DynChatModel for T {
    fn forward_dyn<'a>(
        &'a self,
        http: &'a reqwest::Client,
        request: &'a api::LmRequest,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<api::LmResponse>> + Send + 'a>> {
        Box::pin(self.forward(http, request))
    }
}

/// The typed 3.3 model boundary: dspy's `forward(request: LMRequest) -> LMResponse`.
///
/// The one seam every model implements — a provider-backed [`LM`], a [`Cached`] wrapper, the
/// scripted doubles a test installs — and the one method a module calls to reach its model. Unit
/// tests script it with canned replies while production speaks to real providers through [`LM`].
pub trait ChatModel {
    fn forward<'a>(
        &'a self,
        http: &'a reqwest::Client,
        request: &'a api::LmRequest,
    ) -> impl Future<Output = Result<api::LmResponse>> + Send + 'a;
}

/// One configured language model: a model reference plus the credentials and hosts its
/// provider needs.
pub struct LM {
    pub model: ModelRef,
    pub anthropic_api_key: Option<String>,
    pub openrouter_api_key: Option<String>,
    pub ollama_host: String,
    pub openai: OpenAiConfig,
    /// Whether a repeated request is replayed from [`cache::shared`] rather than paid for again.
    ///
    /// dspy's `LM(cache=True)`, on for the same reason: a program asked the same thing twice
    /// almost never means to buy the answer twice, and every retry-shaped module depends on
    /// `rollout_id` having a cache to miss. [`Self::without_cache`] turns it off.
    pub cache: bool,
}

impl LM {
    /// A model with credentials resolved from the process environment: ANTHROPIC_API_KEY,
    /// OPENROUTER_API_KEY, OPENAI_API_KEY, OPENAI_BASE_URL and OLLAMA_HOST, with the same
    /// defaults the server config uses.
    pub fn new(model: &str) -> Result<Self> {
        Ok(Self {
            model: ModelRef::parse(model)?,
            anthropic_api_key: env_nonempty("ANTHROPIC_API_KEY"),
            openrouter_api_key: env_nonempty("OPENROUTER_API_KEY"),
            ollama_host: env_nonempty("OLLAMA_HOST").unwrap_or_else(|| DEFAULT_OLLAMA_HOST.into()),
            openai: OpenAiConfig::from_env(),
            cache: true,
        })
    }

    /// Reach the provider every time, replaying nothing. dspy's `LM(cache=False)`.
    pub fn without_cache(mut self) -> Self {
        self.cache = false;
        self
    }

    pub fn with_anthropic_key(mut self, key: impl Into<String>) -> Self {
        self.anthropic_api_key = Some(key.into());
        self
    }

    pub fn with_openrouter_key(mut self, key: impl Into<String>) -> Self {
        self.openrouter_api_key = Some(key.into());
        self
    }

    pub fn with_ollama_host(mut self, host: impl Into<String>) -> Self {
        self.ollama_host = host.into();
        self
    }

    pub fn with_openai_key(mut self, key: impl Into<String>) -> Self {
        self.openai.api_key = Some(key.into());
        self
    }

    /// Read the key from another variable — GROQ_API_KEY, TOGETHER_API_KEY — and remember the
    /// name, so a key that turns out to be missing names the variable the caller chose.
    pub fn with_openai_key_env(mut self, var: impl Into<String>) -> Self {
        let var = var.into();
        self.openai.api_key = env_nonempty(&var);
        self.openai.key_var = var;
        self
    }

    /// Point at another OpenAI-shaped service: `https://api.groq.com/openai/v1`,
    /// `http://localhost:8000/v1` for vLLM, `http://localhost:1234/v1` for LM Studio.
    pub fn with_openai_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.openai.base_url = base_url.into();
        self
    }

    /// Opt into schema-constrained decoding, which only some OpenAI-shaped services support.
    /// See [`JsonFormat`].
    pub fn with_openai_json_format(mut self, json_format: JsonFormat) -> Self {
        self.openai.json_format = json_format;
        self
    }

    /// Speak the Responses API rather than chat completions — OpenAI's wire for reasoning models,
    /// dspy's `model_type="responses"`. Non-streaming only for now. See [`OpenAiWire`].
    pub fn with_openai_responses_api(mut self) -> Self {
        self.openai.wire = OpenAiWire::Responses;
        self
    }

    /// Choose which generation-cap field this endpoint is sent. The default follows
    /// OpenAI's own rule; a service that predates `max_completion_tokens` wants
    /// [`TokenLimitRule::AlwaysMaxTokens`].
    pub fn with_openai_token_limit_rule(mut self, rule: TokenLimitRule) -> Self {
        self.openai.token_limit_rule = rule;
        self
    }
}

/// An unset variable and an empty one mean the same thing: not configured.
fn env_nonempty(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|value| !value.is_empty())
}

impl ChatModel for LM {
    async fn forward(
        &self,
        http: &reqwest::Client,
        request: &api::LmRequest,
    ) -> Result<api::LmResponse> {
        if !self.cache {
            let answered = self.ask_provider(http, request).await?;
            usage::record(&self.model.id, answered.spend());
            return Ok(answered);
        }
        let key = request.cache_key(&self.model.id);
        if let Some(replayed) = cache::shared().replay(&key) {
            return Ok(replayed);
        }
        let answered = self.ask_provider(http, request).await?;
        usage::record(&self.model.id, answered.spend());
        cache::shared().keep(key, answered.clone());
        Ok(answered)
    }
}

impl LM {
    /// The typed streaming boundary — dspy's stream of `LMStreamEvent`s.
    ///
    /// An OpenAI-shaped service streams real Server-Sent Events; a provider that does not stream
    /// answers once, and its reply is handed back as the events it would have arrived as, so a
    /// caller consuming a stream need not know which kind it asked. Streaming bypasses the
    /// response cache, as upstream's does — a stream is not a value to store and replay.
    ///
    /// The boxed stream is the same factory the non-streaming dispatch is: the arms return
    /// different stream types, and a `dyn Stream` is what makes them one return type.
    pub fn forward_stream<'a>(
        &'a self,
        http: &'a reqwest::Client,
        request: &'a api::LmRequest,
    ) -> std::pin::Pin<Box<dyn Stream<Item = Result<api::LmStreamEvent>> + Send + 'a>> {
        match self.model.provider {
            Provider::OpenAiCompatible => Box::pin(
                openai::Endpoint::configured(&self.model.id, &self.openai).stream(http, request),
            ),
            Provider::OpenRouter => Box::pin(
                openai::Endpoint::openrouter(&self.model.id, self.openrouter_api_key.as_deref())
                    .stream(http, request),
            ),
            Provider::Anthropic => Box::pin(anthropic::stream(
                http,
                &self.model.id,
                self.anthropic_api_key.as_deref(),
                request,
            )),
            Provider::Ollama => {
                Box::pin(ollama::stream(http, &self.model.id, &self.ollama_host, request))
            }
        }
    }

    /// The call itself, on whichever wire format this model's provider speaks.
    async fn ask_provider(
        &self,
        http: &reqwest::Client,
        request: &api::LmRequest,
    ) -> Result<api::LmResponse> {
        // Every arm resolves the model reference and this LM's credentials into a provider — each
        // its own [`ChatModel`] — then makes the one uniform call. The match is the factory that
        // maps a model string to its provider, which is inherent: dspy does the same in
        // `infer_provider`, and litellm does it inside its own dispatch. The trait is what makes
        // the four interchangeable, and a caller's own provider indistinguishable from these.
        match self.model.provider {
            Provider::Anthropic => {
                anthropic::Anthropic {
                    model: &self.model.id,
                    api_key: self.anthropic_api_key.as_deref(),
                }
                .forward(http, request)
                .await
            }
            Provider::OpenRouter => {
                openai::Endpoint::openrouter(&self.model.id, self.openrouter_api_key.as_deref())
                    .forward(http, request)
                    .await
            }
            Provider::OpenAiCompatible => {
                openai::Endpoint::configured(&self.model.id, &self.openai)
                    .forward(http, request)
                    .await
            }
            Provider::Ollama => {
                ollama::Ollama {
                    model: &self.model.id,
                    host: &self.ollama_host,
                }
                .forward(http, request)
                .await
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drive every `litellm_chat.json` case for one provider through its request builder and hold
    /// the body to litellm's own captured output. Shared by the Anthropic and ollama request
    /// modules so each provider proves parity the same way, against the same fixture.
    pub(crate) fn each_case(provider: &str, build: fn(&str, &api::LmRequest) -> serde_json::Value) {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/conformance/lm_api/litellm_chat.json");
        let fixture: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).expect("fixture is readable"))
                .expect("fixture is valid json");
        let mut checked = 0;
        for case in fixture["cases"].as_array().expect("cases array") {
            if case["provider"] != provider {
                continue;
            }
            let name = case["name"].as_str().expect("a case name");
            let call: api::LmRequest = serde_json::from_value(case["lm_request"].clone())
                .unwrap_or_else(|error| panic!("{name}: the typed request did not parse: {error}"));
            // The prefix before the first `/` is the wire format, which litellm strips off the
            // model before sending — `ModelRef` does the same, so the builder is given the bare id.
            let model = call.model.split_once('/').map_or(call.model.as_str(), |(_, id)| id);
            assert_eq!(
                build(model, &call),
                case["expected"],
                "{name}: our {provider} body diverges from litellm's"
            );
            checked += 1;
        }
        assert!(checked > 0, "no {provider} cases in the fixture");
    }

    #[test]
    fn parses_each_provider_prefix() {
        let anthropic = ModelRef::parse("anthropic/claude-opus-4-8").expect("anthropic");
        assert_eq!(anthropic.provider, Provider::Anthropic);
        assert_eq!(anthropic.id, "claude-opus-4-8");

        let ollama = ModelRef::parse("ollama/qwen2.5:7b-instruct").expect("ollama");
        assert_eq!(ollama.provider, Provider::Ollama);
        assert_eq!(ollama.id, "qwen2.5:7b-instruct");
    }

    #[test]
    fn keeps_slashes_inside_the_model_id() {
        let nested = ModelRef::parse("openrouter/openai/gpt-oss-120b").expect("openrouter");
        assert_eq!(nested.provider, Provider::OpenRouter);
        assert_eq!(nested.id, "openai/gpt-oss-120b");
    }

    #[test]
    fn parses_the_openai_compatible_prefix() {
        let openai = ModelRef::parse("openai/gpt-4o-mini").expect("openai");
        assert_eq!(openai.provider, Provider::OpenAiCompatible);
        assert_eq!(openai.id, "gpt-4o-mini");
    }

    #[test]
    fn rejects_unknown_providers_and_empty_ids() {
        assert!(ModelRef::parse("cohere/command-r").is_err());
        assert!(ModelRef::parse("anthropic/").is_err());
        assert!(ModelRef::parse("no-slash").is_err());
    }

    #[test]
    fn lm_new_parses_the_model_and_rejects_unknown_providers() {
        let lm = LM::new("ollama/qwen2.5:7b-instruct").expect("valid ref");
        assert_eq!(lm.model.provider, Provider::Ollama);
        assert_eq!(lm.model.id, "qwen2.5:7b-instruct");
        assert!(LM::new("cohere/command-r").is_err());
    }

    /// Only the overrides are asserted; the env-resolved values depend on the ambient process.
    #[test]
    fn builder_overrides_replace_the_env_resolved_values() {
        let lm = LM::new("anthropic/claude-opus-4-8")
            .expect("valid ref")
            .with_anthropic_key("ak")
            .with_openrouter_key("ok")
            .with_ollama_host("http://one:1");
        assert_eq!(lm.anthropic_api_key.as_deref(), Some("ak"));
        assert_eq!(lm.openrouter_api_key.as_deref(), Some("ok"));
        assert_eq!(lm.ollama_host, "http://one:1");
    }

    #[test]
    fn the_openai_builders_replace_the_env_resolved_endpoint() {
        let lm = LM::new("openai/llama-3.3-70b")
            .expect("valid ref")
            .with_openai_base_url("http://localhost:8000/v1")
            .with_openai_key("sk-local")
            .with_openai_json_format(JsonFormat::Schema)
            .with_openai_token_limit_rule(TokenLimitRule::AlwaysMaxTokens);
        assert_eq!(lm.openai.base_url, "http://localhost:8000/v1");
        assert_eq!(lm.openai.api_key.as_deref(), Some("sk-local"));
        assert_eq!(lm.openai.json_format, JsonFormat::Schema);
        assert_eq!(lm.openai.token_limit_rule, TokenLimitRule::AlwaysMaxTokens);
    }

    /// PATH stands in for a provider variable: the assertion needs a name that is certainly
    /// set, and setting one is unsafe in this edition and would be seen by every other test
    /// running in parallel.
    #[test]
    fn a_named_key_variable_is_read_and_kept_for_the_error_message() {
        let present = LM::new("openai/gpt-4o-mini")
            .expect("valid ref")
            .with_openai_key_env("PATH");
        assert_eq!(present.openai.key_var, "PATH");
        assert_eq!(present.openai.api_key, std::env::var("PATH").ok());

        let missing = LM::new("openai/gpt-4o-mini")
            .expect("valid ref")
            .with_openai_key_env("DSRS_KEY_VAR_THAT_IS_NOT_SET");
        assert_eq!(missing.openai.key_var, "DSRS_KEY_VAR_THAT_IS_NOT_SET");
        assert_eq!(missing.openai.api_key, None);
    }
}
