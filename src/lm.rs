mod anthropic;
pub mod dummy;
pub mod global;
mod ollama;
mod openai;
mod token_limit;

use std::time::Duration;

use anyhow::{Result, anyhow};
use serde_json::{Value, json};

pub use global::{configure, configure_with_client};
pub use openai::{DEFAULT_OPENAI_BASE_URL, DEFAULT_OPENAI_KEY_VAR, JsonFormat, OpenAiConfig};
pub use token_limit::{TokenLimitField, TokenLimitRule};

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

/// What a turn says: prose, or the content blocks a multimodal field turns it into.
///
/// dspy types a message's content as `str | list[dict]` for the same reason. A field carrying an
/// image cannot reach the provider inside a string — the image travels as its own block, with
/// the prose around it split into blocks either side.
/// Serializes as what it is — a bare string or an array of blocks — which is the shape every
/// OpenAI-compatible provider expects in a message's `content`.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(untagged)]
pub enum Content {
    /// One string, which is every message a text-only signature produces.
    Text(String),
    /// Blocks in the order the provider reads them, each an OpenAI-shaped content part.
    Blocks(Vec<Value>),
}

impl Content {
    /// The prose of a text-only message, or `None` once it has been split into blocks.
    pub fn text(&self) -> Option<&str> {
        match self {
            Content::Text(text) => Some(text),
            Content::Blocks(_) => None,
        }
    }
}

impl<S: Into<String>> From<S> for Content {
    fn from(text: S) -> Self {
        Content::Text(text.into())
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

/// How a model should sample its reply.
///
/// dspy is normalising these onto a request rather than onto a model, because they belong to one
/// call: the same model answers twice and the two attempts differ only here. That is what lets
/// `BestOfN` mean anything and what lets a bootstrap round after the first not repeat itself.
///
/// Two of upstream's fields are deliberately absent, both because this seam cannot carry them.
/// `n`: [`ChatModel::chat`] answers with one completion, so asking for several could only ever
/// be billed and then discarded — asking for many is a change of return type, not a field.
/// `rollout_id`: upstream varies it to miss *its own* response cache and drops it before the
/// provider call, so it never reaches a wire. There is no cache here, which leaves nothing for
/// it to change; what makes a re-ask differ is `temperature`. It earns a field when a cache
/// lands and needs a key to vary.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Sampling {
    /// Unset leaves each provider on the default it is already sent.
    pub temperature: Option<f64>,
    pub max_tokens: Option<u32>,
}

/// One call: what to say, what shape to say it in, and how to sample the reply.
///
/// dspy's `LMRequest`, which its normalised API takes in place of loose keyword arguments. A
/// request travelling as a value is also why no ambient state is needed to override a model for
/// one call — `dspy.context` scopes a ContextVar to do the same thing.
pub struct LmRequest<'a> {
    pub system: &'a str,
    pub turns: &'a [ChatTurn],
    pub mode: OutputMode<'a>,
    pub sampling: Sampling,
}

impl<'a> LmRequest<'a> {
    /// One call at the provider's own defaults.
    pub fn new(system: &'a str, turns: &'a [ChatTurn], mode: OutputMode<'a>) -> Self {
        Self {
            system,
            turns,
            mode,
            sampling: Sampling::default(),
        }
    }

    pub fn sampled(mut self, sampling: Sampling) -> Self {
        self.sampling = sampling;
        self
    }
}

/// The raw model call behind every adapter — the one seam unit tests script with canned
/// replies while production speaks to real providers through [`LM`].
/// The object-safe form of [`ChatModel`], so a model can be stored behind a pointer.
///
/// `ChatModel` returns `impl Future`, which is ergonomic to implement and impossible to make
/// into a trait object. Every `ChatModel` gets this for free through the blanket impl below,
/// and the global configuration stores this form — which is what lets a test install a
/// scripted model the way dspy installs a `DummyLM`.
pub trait DynChatModel: Send + Sync {
    fn chat_dyn<'a>(
        &'a self,
        http: &'a reqwest::Client,
        request: &'a LmRequest<'a>,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>>;
}

impl<T: ChatModel + Send + Sync> DynChatModel for T {
    fn chat_dyn<'a>(
        &'a self,
        http: &'a reqwest::Client,
        request: &'a LmRequest<'a>,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>> {
        Box::pin(self.chat(http, request))
    }
}

pub trait ChatModel {
    fn chat<'a>(
        &'a self,
        http: &'a reqwest::Client,
        request: &'a LmRequest<'a>,
    ) -> impl Future<Output = Result<String>> + Send + 'a;
}

/// One configured language model: a model reference plus the credentials and hosts its
/// provider needs.
pub struct LM {
    pub model: ModelRef,
    pub anthropic_api_key: Option<String>,
    pub openrouter_api_key: Option<String>,
    pub ollama_host: String,
    pub openai: OpenAiConfig,
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
        })
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
    async fn chat(&self, http: &reqwest::Client, request: &LmRequest<'_>) -> Result<String> {
        match self.model.provider {
            Provider::Anthropic => {
                anthropic::chat(
                    http,
                    &self.model.id,
                    self.anthropic_api_key.as_deref(),
                    request,
                )
                .await
            }
            Provider::OpenRouter => {
                openai::Endpoint::openrouter(self.openrouter_api_key.as_deref())
                    .chat(http, &self.model.id, request)
                    .await
            }
            Provider::OpenAiCompatible => {
                openai::Endpoint::configured(&self.openai)
                    .chat(http, &self.model.id, request)
                    .await
            }
            Provider::Ollama => {
                ollama::chat(http, &self.model.id, &self.ollama_host, request).await
            }
        }
    }
}

fn turn_json(turn: &ChatTurn) -> Value {
    json!({ "role": turn.role.as_str(), "content": turn.content })
}

/// OpenAI-style message list: the system prompt leads, then the conversation turns.
fn wire_messages(system: &str, turns: &[ChatTurn]) -> Vec<Value> {
    std::iter::once(json!({ "role": "system", "content": system }))
        .chain(turns.iter().map(turn_json))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn wire_messages_lead_with_system_then_keep_turn_order() {
        let turns = [
            ChatTurn::user("draft it"),
            ChatTurn::assistant("first try"),
            ChatTurn::user("fix it"),
        ];
        let messages = wire_messages("be helpful", &turns);
        let roles: Vec<&str> = messages
            .iter()
            .map(|m| m["role"].as_str().unwrap_or_default())
            .collect();
        assert_eq!(roles, ["system", "user", "assistant", "user"]);
        assert_eq!(messages[2]["content"], "first try");
    }
}
