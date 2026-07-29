//! Which provider answers, worked out from the model's name.
//!
//! dspy writes a model as `"openai/gpt-4o-mini"` and litellm decides from the prefix who to call.
//! The same string reaches this crate, so the same decision is made here — with the difference
//! that the set of providers is closed and named, since a Rust enum can say what litellm's
//! registry only discovers.

use anyhow::{Result, anyhow};

/// Where a model runs. Each provider keeps its own wire format behind the one `LM` interface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    Anthropic,
    OpenRouter,
    /// Any service exposing OpenAI's `/v1/chat/completions`: OpenAI itself, Groq, Together,
    /// vLLM, LM Studio. Which one is a matter of [`OpenAiConfig`] rather than of the model
    /// prefix, since they are one wire format on different hosts.
    OpenAiCompatible,
    /// ollama's `/api/generate` route, litellm's `ollama/` — one flattened prompt, no native
    /// tool calls. The legacy path; [`Provider::OllamaChat`] is the one to reach for tool use.
    Ollama,
    /// ollama's `/api/chat` route, litellm's `ollama_chat/` — a message list and native tool
    /// calls. What dspy's docs steer an ollama user to.
    OllamaChat,
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
            "ollama_chat" => Provider::OllamaChat,
            other => return Err(anyhow!("unknown provider {other:?} in {raw:?}")),
        };
        Ok(Self {
            provider,
            id: id.to_owned(),
        })
    }

    /// The reference as written, `provider/model-id`. What the capability registry is keyed by,
    /// and what dspy hands litellm.
    pub fn reference(&self) -> String {
        let prefix = match self.provider {
            Provider::Anthropic => "anthropic",
            Provider::OpenRouter => "openrouter",
            Provider::OpenAiCompatible => "openai",
            Provider::Ollama => "ollama",
            Provider::OllamaChat => "ollama_chat",
        };
        format!("{prefix}/{}", self.id)
    }
}
