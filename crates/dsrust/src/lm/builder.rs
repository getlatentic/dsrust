//! Building an [`LM`], with the one thing it cannot do without named first.
//!
//! dspy's signature is `LM(model, temperature=None, max_tokens=None, …)`: one required argument and
//! a run of optional ones. [`LM::builder`] takes the model positionally for the same reason the
//! compiler is the right place to enforce it — a builder that made the model optional would either
//! fail at `build` or pick one on the caller's behalf, and dspy-rs's does the latter.

use std::time::Duration;

use anyhow::Result;

use super::{LM, Retry, api};

/// The settings an LM carries for every call through it — dspy's `lm.kwargs`.
impl LM {
    /// This request with anything it left unset filled from the LM's own settings, and the system
    /// message renamed where [`use_developer_role`](LmBuilder::use_developer_role) asks for it.
    pub(super) fn with_defaults(&self, request: &api::LmRequest) -> api::LmRequest {
        let mut asked = request.clone();
        api::defaults::beneath(&mut asked.config, &self.config);
        if self.use_developer_role && matches!(self.openai.wire, super::OpenAiWire::Responses) {
            for message in &mut asked.messages {
                if message.role == "system" {
                    message.role = "developer".to_owned();
                }
            }
        }
        asked
    }

    /// dspy `lm.copy(**kwargs)`: this model again, with these settings replacing its own.
    ///
    /// The one post-construction setter, rather than one per field. Upstream's counterpart is a
    /// single `copy` taking keywords; `LM::builder` is where a field is named individually, and it
    /// uses dspy's own names for them.
    pub fn config(mut self, config: api::LmConfig) -> Self {
        self.config = config;
        self
    }
}

/// A model under construction. Reached through [`LM::builder`], which takes the model id, so every
/// state of this type already has the one thing an `LM` cannot do without.
///
/// `build` is fallible for one reason: the model id is parsed, and `openai/` with nothing after it
/// is not a model. That check happens once, here, rather than at the first call.
///
/// This is what `dspy.LM(model, temperature=…, max_tokens=…)` spells with keywords — Python takes
/// them in any order and Rust needs a name per field, which is the whole reason the type exists:
///
/// ```
/// use std::time::Duration;
///
/// let lm = dsrust::LM::builder("openai/gpt-4o-mini")
///     .temperature(0.0)
///     .max_tokens(256)
///     .timeout(Duration::from_secs(30))
///     .num_retries(3)
///     .build()
///     .expect("a model id with a provider and a name");
/// assert_eq!(lm.model.id, "gpt-4o-mini");
/// ```
///
/// Held as a value when the configuration is decided at run time, which is the case a chain cannot
/// express:
///
/// ```
/// use dsrust::lm::LmBuilder;
///
/// let mut building: LmBuilder = dsrust::LM::builder("openai/gpt-4o-mini");
/// if std::env::var("DSRUST_OFFLINE").is_ok() {
///     building = building.api_base("http://localhost:11434");
/// }
/// let lm = building.build().expect("a model id with a provider and a name");
/// assert_eq!(lm.model.id, "gpt-4o-mini");
/// ```
///
/// A model id the parse cannot read is refused here rather than at the first call:
///
/// ```
/// assert!(dsrust::LM::builder("openai/").build().is_err());
/// ```
pub struct LmBuilder {
    pub(super) model: String,
    pub(super) config: api::LmConfig,
    /// Deferred because each one needs the built `LM`, and the point of the builder is that a
    /// caller states what they want before anything is validated.
    pub(super) settings: Vec<Box<dyn FnOnce(LM) -> LM>>,
}

impl LmBuilder {
    /// The sampling temperature for every call through this model. dspy's `temperature=`.
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.config.temperature = Some(temperature);
        self
    }

    /// The token ceiling for every call. dspy's `max_tokens=`.
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.config.max_tokens = Some(max_tokens);
        self
    }

    /// Where an OpenAI-compatible host lives, for a service that is not OpenAI itself.
    pub fn api_base(mut self, base_url: impl Into<String>) -> Self {
        let base_url = base_url.into();
        self.settings
            .push(Box::new(move |lm| lm.openai_base_url(base_url)));
        self
    }

    /// The credential, where it is not coming from the environment.
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        let key = key.into();
        self.settings
            .push(Box::new(move |lm| match lm.model.provider {
                crate::lm::Provider::Anthropic => lm.anthropic_api_key(key),
                crate::lm::Provider::OpenRouter => lm.openrouter_api_key(key),
                crate::lm::Provider::Ollama | crate::lm::Provider::OllamaChat => {
                    lm.ollama_api_key(key)
                }
                crate::lm::Provider::OpenAiCompatible => lm.openai_api_key(key),
            }));
        self
    }

    /// Where ollama is listening, when it is not the default.
    pub fn ollama_host(mut self, host: impl Into<String>) -> Self {
        let host = host.into();
        self.settings.push(Box::new(move |lm| lm.ollama_host(host)));
        self
    }

    /// How long one call may take before it is abandoned.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.settings.push(Box::new(move |lm| lm.timeout(timeout)));
        self
    }

    /// Whether an identical earlier answer is replayed instead of asking again.
    ///
    /// dspy's `LM(model, cache=True)`, and a boolean for the same reason: a caller with a runtime
    /// switch writes `.cache(measuring_the_model)` rather than branching around a method call.
    pub fn cache(mut self, cache: bool) -> Self {
        self.settings.push(Box::new(move |lm| lm.cache(cache)));
        self
    }

    /// How many times a transiently failing call is asked before the failure is handed back.
    ///
    /// dspy's `LM(model, num_retries=3)`, and it counts asks the way upstream's does: three means
    /// two retries. `1` never asks twice, which is what a test measuring one call wants.
    pub fn num_retries(mut self, attempts: usize) -> Self {
        self.settings
            .push(Box::new(move |lm| lm.retry(Retry::attempts(attempts))));
        self
    }

    /// Watch this model's calls, and no other model's. dspy's `LM(model, callbacks=[…])`.
    pub fn callbacks(
        mut self,
        callbacks: impl IntoIterator<Item = std::sync::Arc<dyn crate::Callback>>,
    ) -> Self {
        let callbacks: Vec<_> = callbacks.into_iter().collect();
        self.settings
            .push(Box::new(move |lm| lm.callbacks(callbacks)));
        self
    }

    /// Send the system message as `developer` instead, which is what the o1 family takes. dspy's
    /// `LM(model, use_developer_role=True)`, and as upstream has it this applies on the Responses
    /// wire only.
    pub fn use_developer_role(mut self, use_developer_role: bool) -> Self {
        self.settings.push(Box::new(move |lm| {
            lm.use_developer_role(use_developer_role)
        }));
        self
    }

    /// Anything with no named setter — `top_p`, `stop`, `logprobs`, a provider knob.
    pub fn config(mut self, config: api::LmConfig) -> Self {
        let temperature = self.config.temperature;
        let max_tokens = self.config.max_tokens;
        self.config = config;
        // A named setter called before this one still stands: it was said more specifically.
        self.config.temperature = temperature.or(self.config.temperature);
        self.config.max_tokens = max_tokens.or(self.config.max_tokens);
        self
    }

    /// The model, or the reason its id is not one.
    pub fn build(self) -> Result<LM> {
        let built = LM::new(&self.model)?.config(self.config);
        Ok(self.settings.into_iter().fold(built, |lm, apply| apply(lm)))
    }
}
