pub mod global;

use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use serde_json::{Value, json};

pub use global::{configure, configure_with_client};

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
    Ollama,
}

/// A LiteLLM-style model reference, `provider/model-id`: `anthropic/claude-opus-4-8`,
/// `openrouter/openai/gpt-oss-120b`, `ollama/qwen2.5:7b-instruct`. The id may itself
/// contain slashes (OpenRouter namespaces models by vendor).
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
    fn as_str(self) -> &'static str {
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
    pub content: String,
}

impl ChatTurn {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
        }
    }
}

/// What the reply should look like on the wire. `Json` engages the provider's native
/// structured output (Anthropic json_schema, OpenRouter/ollama JSON mode); `Text` leaves
/// the reply free-form for marker-based adapters, which need no provider support at all.
#[derive(Debug)]
pub enum OutputMode<'a> {
    Text,
    Json { schema: &'a Value },
}

/// The raw model call behind every adapter — the one seam unit tests script with canned
/// replies while production speaks to real providers through [`LM`].
pub trait ChatModel {
    fn chat(
        &self,
        http: &reqwest::Client,
        system: &str,
        turns: &[ChatTurn],
        mode: &OutputMode<'_>,
    ) -> impl Future<Output = Result<String>> + Send;
}

/// One configured language model: a model reference plus the credentials and hosts its
/// provider needs.
pub struct LM {
    pub model: ModelRef,
    pub anthropic_api_key: Option<String>,
    pub openrouter_api_key: Option<String>,
    pub ollama_host: String,
}

impl LM {
    /// A model with credentials resolved from the process environment: ANTHROPIC_API_KEY,
    /// OPENROUTER_API_KEY, and OLLAMA_HOST, with the same defaults the server config uses.
    pub fn new(model: &str) -> Result<Self> {
        Ok(Self {
            model: ModelRef::parse(model)?,
            anthropic_api_key: env_nonempty("ANTHROPIC_API_KEY"),
            openrouter_api_key: env_nonempty("OPENROUTER_API_KEY"),
            ollama_host: env_nonempty("OLLAMA_HOST").unwrap_or_else(|| DEFAULT_OLLAMA_HOST.into()),
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
}

/// An unset variable and an empty one mean the same thing: not configured.
fn env_nonempty(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|value| !value.is_empty())
}

impl ChatModel for LM {
    async fn chat(
        &self,
        http: &reqwest::Client,
        system: &str,
        turns: &[ChatTurn],
        mode: &OutputMode<'_>,
    ) -> Result<String> {
        match self.model.provider {
            Provider::Anthropic => self.anthropic(http, system, turns, mode).await,
            Provider::OpenRouter => self.openrouter(http, system, turns, mode).await,
            Provider::Ollama => self.ollama(http, system, turns, mode).await,
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

impl LM {
    async fn anthropic(
        &self,
        http: &reqwest::Client,
        system: &str,
        turns: &[ChatTurn],
        mode: &OutputMode<'_>,
    ) -> Result<String> {
        let key = self
            .anthropic_api_key
            .as_deref()
            .ok_or_else(|| anyhow!("ANTHROPIC_API_KEY is not set"))?;
        let mut request = json!({
            "model": self.model.id,
            "max_tokens": 1024,
            "system": system,
            "messages": turns.iter().map(turn_json).collect::<Vec<_>>(),
        });
        if let OutputMode::Json { schema } = mode {
            request["output_config"] =
                json!({ "format": { "type": "json_schema", "schema": schema } });
        }
        let response = http
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .timeout(PROVIDER_TIMEOUT)
            .json(&request)
            .send()
            .await
            .context("anthropic request failed")?;
        let status = response.status();
        let body: Value = response
            .json()
            .await
            .context("anthropic response was not JSON")?;
        if !status.is_success() {
            let detail = body["error"]["message"].as_str().unwrap_or("unknown error");
            return Err(anyhow!("anthropic {status}: {detail}"));
        }
        let text = body["content"]
            .as_array()
            .and_then(|blocks| {
                blocks
                    .iter()
                    .find(|block| block["type"] == "text")
                    .and_then(|block| block["text"].as_str())
            })
            .unwrap_or("{}");
        Ok(text.to_owned())
    }

    async fn openrouter(
        &self,
        http: &reqwest::Client,
        system: &str,
        turns: &[ChatTurn],
        mode: &OutputMode<'_>,
    ) -> Result<String> {
        let key = self
            .openrouter_api_key
            .as_deref()
            .ok_or_else(|| anyhow!("OPENROUTER_API_KEY is not set"))?;
        let mut request = json!({
            "model": self.model.id,
            "max_tokens": 1024,
            "messages": wire_messages(system, turns),
        });
        if matches!(mode, OutputMode::Json { .. }) {
            request["response_format"] = json!({ "type": "json_object" });
        }
        let response = http
            .post("https://openrouter.ai/api/v1/chat/completions")
            .bearer_auth(key)
            .timeout(PROVIDER_TIMEOUT)
            .json(&request)
            .send()
            .await
            .context("openrouter request failed")?;
        let status = response.status();
        let body: Value = response
            .json()
            .await
            .context("openrouter response was not JSON")?;
        if !status.is_success() {
            let detail = body["error"]["message"].as_str().unwrap_or("unknown error");
            return Err(anyhow!("openrouter {status}: {detail}"));
        }
        body["choices"][0]["message"]["content"]
            .as_str()
            .map(str::to_owned)
            .context("openrouter returned no content")
    }

    async fn ollama(
        &self,
        http: &reqwest::Client,
        system: &str,
        turns: &[ChatTurn],
        mode: &OutputMode<'_>,
    ) -> Result<String> {
        let mut request = json!({
            "model": self.model.id,
            "stream": false,
            "options": { "temperature": 0.7 },
            "messages": wire_messages(system, turns),
        });
        if matches!(mode, OutputMode::Json { .. }) {
            request["format"] = json!("json");
        }
        let response = http
            .post(format!("{}/api/chat", self.ollama_host))
            .timeout(PROVIDER_TIMEOUT)
            .json(&request)
            .send()
            .await
            .context("ollama request failed")?;
        if !response.status().is_success() {
            return Err(anyhow!("ollama {}", response.status()));
        }
        let body: Value = response
            .json()
            .await
            .context("ollama response was not JSON")?;
        body["message"]["content"]
            .as_str()
            .map(str::to_owned)
            .context("ollama returned no content")
    }
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
    fn rejects_unknown_providers_and_empty_ids() {
        assert!(ModelRef::parse("openai/gpt-4o").is_err());
        assert!(ModelRef::parse("anthropic/").is_err());
        assert!(ModelRef::parse("no-slash").is_err());
    }

    #[test]
    fn lm_new_parses_the_model_and_rejects_unknown_providers() {
        let lm = LM::new("ollama/qwen2.5:7b-instruct").expect("valid ref");
        assert_eq!(lm.model.provider, Provider::Ollama);
        assert_eq!(lm.model.id, "qwen2.5:7b-instruct");
        assert!(LM::new("openai/gpt-4o").is_err());
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
