mod anthropic;
pub mod api;
pub mod builder;
pub mod cache;
mod call;
mod capabilities;
mod dispatch;
pub mod dummy;
mod error;
pub mod global;
mod model;
mod ollama;
pub mod openai;
pub mod retry;
mod routing;
pub mod saved;
mod streaming;
mod token_limit;
mod turn;
pub mod usage;

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;

use crate::Callback;

pub use api::{
    Assistant, Content, Detail, Developer, LmItem, LmMessage, LmPart, LmRequest, LmResponse,
    LmSource, System, User,
};
pub use builder::LmBuilder;
pub use cache::{Cached, ResponseCache};
pub use call::{LmUsage, Sampling};
pub use capabilities::Capabilities;
pub use error::{ContextWindowExceeded, LmErrorKind, LmFailure};
pub use global::{
    Scope, configure, configure_with_client, context, context_model, context_with_client,
};
pub use model::{ChatModel, DynChatModel};
pub use openai::{
    DEFAULT_OPENAI_BASE_URL, DEFAULT_OPENAI_KEY_VAR, JsonFormat, OpenAiConfig, OpenAiWire,
};
pub use retry::Retry;
pub use routing::{ModelRef, Provider};
pub use token_limit::{TokenLimitField, TokenLimitRule};
pub(crate) use turn::messages_of;
pub use turn::{ChatTurn, OutputMode, Role};
pub use usage::{Tracking, UsageTracker, track as track_usage};

/// What bounds a provider call unless the caller says otherwise: litellm's own default, which dspy
/// never overrides, so a program that answers upstream answers here.
///
/// The number is `litellm.request_timeout`, measured against the pinned install rather than read
/// off a docstring. It is long enough to look like no bound at all, and that is the point — a local
/// model serving a long prompt takes minutes, and a default tight enough to feel responsive is one
/// that fails a call dspy would have completed. A caller who wants a real bound sets it:
/// [`LM::timeout`].
pub const DEFAULT_PROVIDER_TIMEOUT: Duration = Duration::from_secs(6000);

/// The stock ollama port on the local machine, shared with the server's config so `LM::new`
/// and the server resolve the same host when OLLAMA_HOST is unset.
/// ```
/// // What `LmBuilder::ollama_host` starts from, so a caller can compare against it rather than
/// // repeating the literal.
/// assert_eq!(dsrust::lm::DEFAULT_OLLAMA_HOST, "http://localhost:11434");
/// ```
pub const DEFAULT_OLLAMA_HOST: &str = "http://localhost:11434";

/// One configured language model: a model reference plus the credentials and hosts its
/// provider needs.
pub struct LM {
    pub model: ModelRef,
    pub anthropic_api_key: Option<String>,
    pub openrouter_api_key: Option<String>,
    pub ollama_host: String,
    /// The credential a hosted ollama wants, from OLLAMA_API_KEY. A server on the local machine
    /// wants none, so this is normally unset.
    pub ollama_api_key: Option<String>,
    pub openai: OpenAiConfig,
    /// dspy's `lm.kwargs`: what `dspy.LM(model, temperature=…, max_tokens=…)` keeps on the
    /// instance and merges beneath every call. A module's own config overrides it field by field.
    pub config: api::LmConfig,
    /// Whether a repeated request is replayed from [`cache::shared`] rather than paid for again.
    ///
    /// dspy's `LM(cache=True)`, on for the same reason: a program asked the same thing twice
    /// almost never means to buy the answer twice, and every retry-shaped module depends on
    /// `rollout_id` having a cache to miss. [`Self::cache`] turns it off.
    pub cache: bool,
    /// How many times a transiently failing call is asked, dspy's `LM(num_retries=3)`. See
    /// [`retry`].
    pub retry: Retry,
    /// dspy's `LM(use_developer_role=False)`: send the system message under the o1-family's
    /// `developer` role instead. Applies on the Responses wire only, as upstream's does.
    pub use_developer_role: bool,
    /// How long any one call to this model may take. See [`Self::timeout`].
    pub timeout: Duration,
    /// What this model can be asked for natively, where the caller has stated it rather than
    /// leaving it to the registry. See [`Self::capabilities`].
    capabilities: Option<Capabilities>,
    /// dspy's `LM(model, callbacks=[…])`: watchers told about this model's calls and no other's.
    /// See [`Self::callbacks`].
    callbacks: Vec<Arc<dyn Callback>>,
}

impl LM {
    /// A model with credentials resolved from the process environment: ANTHROPIC_API_KEY,
    /// OPENROUTER_API_KEY, OPENAI_API_KEY, OPENAI_BASE_URL and OLLAMA_HOST, with the same
    /// defaults the server config uses.
    ///
    /// Takes anything that reads as a string, because the name is as often built at runtime as
    /// written in the source — an app with a model picker has a `String`, and asking it for
    /// `&format!(…)` is a borrow it should not have to think about.
    pub fn new(model: impl AsRef<str>) -> Result<Self> {
        Ok(Self {
            config: api::LmConfig::default(),
            model: ModelRef::parse(model.as_ref())?,
            anthropic_api_key: env_nonempty("ANTHROPIC_API_KEY"),
            openrouter_api_key: env_nonempty("OPENROUTER_API_KEY"),
            ollama_host: env_nonempty("OLLAMA_HOST").unwrap_or_else(|| DEFAULT_OLLAMA_HOST.into()),
            ollama_api_key: env_nonempty("OLLAMA_API_KEY"),
            openai: OpenAiConfig::from_env(),
            cache: true,
            retry: Retry::default(),
            use_developer_role: false,
            timeout: DEFAULT_PROVIDER_TIMEOUT,
            capabilities: None,
            callbacks: Vec::new(),
        })
    }

    /// Build a model, naming the required part first.
    ///
    /// dspy's own signature is `LM(model, temperature=None, max_tokens=None, …)`: one required
    /// argument, the rest optional. So the model is positional here too, and the compiler enforces
    /// it — there is no state in which `build` can be reached without one. A builder that made the
    /// model just another optional field would move that guarantee to runtime, or silently pick a
    /// model on the caller's behalf.
    ///
    /// ```no_run
    /// # use dsrust::lm::LM;
    /// let lm = LM::builder("openai/gpt-4o-mini")
    ///     .temperature(0.5)
    ///     .max_tokens(512)
    ///     .build()?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    ///
    /// The same knobs are on [`LM`] itself as `with_*`, for a caller who already has one.
    pub fn builder(model: impl AsRef<str>) -> LmBuilder {
        LmBuilder {
            model: model.as_ref().to_owned(),
            config: api::LmConfig::default(),
            settings: Vec::new(),
        }
    }

    /// How long any one call to this model may take, replacing [`DEFAULT_PROVIDER_TIMEOUT`]. dspy's
    /// `LM(..., timeout=…)`, which litellm applies the same way: to the whole request rather than to
    /// the idle gaps within it.
    ///
    /// Raise it for a local model reading a long prompt — the cost of a low bound is a call that
    /// would have answered being abandoned, not a slow one being made faster.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// How many times a transiently failing call is asked before the failure is handed back — dspy's
    /// `LM(num_retries=3)`, counting asks rather than retries. See [`retry`].
    pub fn retry(mut self, retry: Retry) -> Self {
        self.retry = retry;
        self
    }

    /// Send the system message as `developer`, which the o1 family takes instead — dspy's
    /// `LM(use_developer_role=True)`, and like upstream's it applies on the Responses wire only.
    pub fn use_developer_role(mut self, use_developer_role: bool) -> Self {
        self.use_developer_role = use_developer_role;
        self
    }

    /// State what this model can be asked for natively, rather than have it resolved.
    ///
    /// The one setter here that keeps a `with_` prefix, because there is no upstream name to take:
    /// dspy has no such constructor argument — it asks litellm's registry — and `capabilities` is
    /// already [`ChatModel::capabilities`], which *reads* them.
    ///
    /// Resolution is litellm's registry, and for an unlisted ollama model the server itself. This
    /// short-circuits both — for a provider-compatible endpoint serving a model under a name the
    /// registry does not know, or to keep a program off the network while it decides.
    pub fn with_capabilities(mut self, capabilities: Capabilities) -> Self {
        self.capabilities = Some(capabilities);
        self
    }

    /// Watch this model's calls, and no other model's — dspy's `LM(model, callbacks=[…])`.
    ///
    /// The second of upstream's two ways to register: this one is per instance, where
    /// [`configure_callbacks`](crate::configure_callbacks) is per process. Both are told, the
    /// process-wide ones first.
    pub fn callbacks(mut self, callbacks: impl IntoIterator<Item = Arc<dyn Callback>>) -> Self {
        self.callbacks = callbacks.into_iter().collect();
        self
    }

    /// Whether an identical earlier answer is replayed instead of asking again.
    ///
    /// dspy's `cache=` argument, and on by default as upstream has it. Turn it off to measure a
    /// model: with it on, a second run reads the first run's reply and reports it as fresh.
    pub fn cache(mut self, cache: bool) -> Self {
        self.cache = cache;
        self
    }

    pub fn anthropic_api_key(mut self, key: impl Into<String>) -> Self {
        self.anthropic_api_key = Some(key.into());
        self
    }

    pub fn openrouter_api_key(mut self, key: impl Into<String>) -> Self {
        self.openrouter_api_key = Some(key.into());
        self
    }

    /// Authenticate to a hosted ollama. litellm sends this as a bearer token, and so does every
    /// call this crate makes — the chat, the stream, and the capability probe alike.
    pub fn ollama_api_key(mut self, key: impl Into<String>) -> Self {
        self.ollama_api_key = Some(key.into());
        self
    }

    pub fn ollama_host(mut self, host: impl Into<String>) -> Self {
        self.ollama_host = host.into();
        self
    }

    pub fn openai_api_key(mut self, key: impl Into<String>) -> Self {
        self.openai.api_key = Some(key.into());
        self
    }

    /// Read the key from another variable — GROQ_API_KEY, TOGETHER_API_KEY — and remember the
    /// name, so a key that turns out to be missing names the variable the caller chose.
    pub fn openai_key_var(mut self, var: impl Into<String>) -> Self {
        let var = var.into();
        self.openai.api_key = env_nonempty(&var);
        self.openai.key_var = var;
        self
    }

    /// Point at another OpenAI-shaped service: `https://api.groq.com/openai/v1`,
    /// `http://localhost:8000/v1` for vLLM, `http://localhost:1234/v1` for LM Studio.
    pub fn openai_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.openai.base_url = base_url.into();
        self
    }

    /// Opt into schema-constrained decoding, which only some OpenAI-shaped services support.
    /// See [`JsonFormat`].
    pub fn openai_json_format(mut self, json_format: JsonFormat) -> Self {
        self.openai.json_format = json_format;
        self
    }

    /// Speak the Responses API rather than chat completions — OpenAI's wire for reasoning models,
    /// dspy's `model_type="responses"`. Non-streaming only for now. See [`OpenAiWire`].
    pub fn openai_responses_api(mut self) -> Self {
        self.openai.wire = OpenAiWire::Responses;
        self
    }

    /// Choose which generation-cap field this endpoint is sent. The default follows
    /// OpenAI's own rule; a service that predates `max_completion_tokens` wants
    /// [`TokenLimitRule::AlwaysMaxTokens`].
    pub fn openai_token_limit_rule(mut self, rule: TokenLimitRule) -> Self {
        self.openai.token_limit_rule = rule;
        self
    }
}

/// An unset variable and an empty one mean the same thing: not configured.
fn env_nonempty(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|value| !value.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// dspy's gpt-5-chat native-reasoning caveat, on the `LM` that knows the model and its route.
    #[test]
    fn native_reasoning_is_suppressed_only_for_gpt5_on_the_chat_route() {
        // gpt-5 on the chat-completions route: litellm 1.79.0 loses the reasoning content.
        assert!(
            !LM::new("openai/gpt-5-mini")
                .expect("an LM")
                .native_reasoning_usable()
        );
        // The Responses API is unaffected.
        assert!(
            LM::new("openai/gpt-5-mini")
                .expect("an LM")
                .openai_responses_api()
                .native_reasoning_usable()
        );
        // A non-gpt-5 model on the chat route is fine.
        assert!(
            LM::new("openai/gpt-4o")
                .expect("an LM")
                .native_reasoning_usable()
        );
        // Another provider is not the OpenAI chat API, so the caveat never applies.
        assert!(
            LM::new("anthropic/claude-opus-4-1")
                .expect("an LM")
                .native_reasoning_usable()
        );
    }

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
            let model = call
                .model
                .split_once('/')
                .map_or(call.model.as_str(), |(_, id)| id);
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
        // The two ollama routes are distinct providers, as they are in litellm.
        let chat = LM::new("ollama_chat/qwen2.5:7b-instruct").expect("valid ref");
        assert_eq!(chat.model.provider, Provider::OllamaChat);
        assert_eq!(chat.model.id, "qwen2.5:7b-instruct");
        assert!(LM::new("cohere/command-r").is_err());
    }

    /// Only the overrides are asserted; the env-resolved values depend on the ambient process.
    #[test]
    fn builder_overrides_replace_the_env_resolved_values() {
        let lm = LM::new("anthropic/claude-opus-4-8")
            .expect("valid ref")
            .anthropic_api_key("ak")
            .openrouter_api_key("ok")
            .ollama_host("http://one:1");
        assert_eq!(lm.anthropic_api_key.as_deref(), Some("ak"));
        assert_eq!(lm.openrouter_api_key.as_deref(), Some("ok"));
        assert_eq!(lm.ollama_host, "http://one:1");
    }

    #[test]
    fn the_openai_builders_replace_the_env_resolved_endpoint() {
        let lm = LM::new("openai/llama-3.3-70b")
            .expect("valid ref")
            .openai_base_url("http://localhost:8000/v1")
            .openai_api_key("sk-local")
            .openai_json_format(JsonFormat::Schema)
            .openai_token_limit_rule(TokenLimitRule::AlwaysMaxTokens);
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
            .openai_key_var("PATH");
        assert_eq!(present.openai.key_var, "PATH");
        assert_eq!(present.openai.api_key, std::env::var("PATH").ok());

        let missing = LM::new("openai/gpt-4o-mini")
            .expect("valid ref")
            .openai_key_var("DSRS_KEY_VAR_THAT_IS_NOT_SET");
        assert_eq!(missing.openai.key_var, "DSRS_KEY_VAR_THAT_IS_NOT_SET");
        assert_eq!(missing.openai.api_key, None);
    }
}
