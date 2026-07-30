//! Building an [`LM`], with the one thing it cannot do without named first.
//!
//! dspy's signature is `LM(model, temperature=None, max_tokens=None, …)`: one required argument and
//! a run of optional ones. [`LM::builder`] takes the model positionally for the same reason the
//! compiler is the right place to enforce it — a builder that made the model optional would either
//! fail at `build` or pick one on the caller's behalf, and dspy-rs's does the latter.

use std::time::Duration;

use anyhow::Result;

use super::{LM, api};

/// The settings an LM carries for every call through it — dspy's `lm.kwargs`.
impl LM {
    /// This request with anything it left unset filled from the LM's own settings.
    pub(super) fn with_defaults(&self, request: &api::LmRequest) -> api::LmRequest {
        let mut asked = request.clone();
        api::defaults::beneath(&mut asked.config, &self.config);
        asked
    }

    /// The sampling temperature every call through this model uses unless it states its own.
    /// dspy's `dspy.LM(model, temperature=…)`.
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.config.temperature = Some(temperature);
        self
    }

    /// The token ceiling every call through this model uses unless it states its own.
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.config.max_tokens = Some(max_tokens);
        self
    }

    /// Any other setting dspy would have taken as a keyword argument, as a whole config.
    ///
    /// The escape hatch for the fields with no named setter — `top_p`, `stop`, `logprobs`, a
    /// provider knob under `extensions`. Replaces whatever was set before it.
    pub fn with_config(mut self, config: api::LmConfig) -> Self {
        self.config = config;
        self
    }
}

/// A model under construction. Reached through [`LM::builder`], which takes the model id, so every
/// state of this type already has the one thing an `LM` cannot do without.
///
/// `build` is fallible for one reason: the model id is parsed, and `openai/` with nothing after it
/// is not a model. That check happens once, here, rather than at the first call.
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
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        let base_url = base_url.into();
        self.settings
            .push(Box::new(move |lm| lm.with_openai_base_url(base_url)));
        self
    }

    /// The credential, where it is not coming from the environment.
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        let key = key.into();
        self.settings
            .push(Box::new(move |lm| lm.with_openai_key(key)));
        self
    }

    /// How long one call may take before it is abandoned.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.settings
            .push(Box::new(move |lm| lm.with_timeout(timeout)));
        self
    }

    /// Ask the provider every time rather than replaying an identical earlier answer.
    pub fn no_cache(mut self) -> Self {
        self.settings.push(Box::new(LM::without_cache));
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
        let built = LM::new(&self.model)?.with_config(self.config);
        Ok(self.settings.into_iter().fold(built, |lm, apply| apply(lm)))
    }
}
