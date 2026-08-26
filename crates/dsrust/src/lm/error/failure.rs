//! dspy `DSPyError`'s structured metadata, for an LM call that failed.
//!
//! What upstream keeps beside the message — the model, the provider, the status, the request id,
//! the retry delay — is what a caller needs to decide what to do next without parsing prose. It
//! travels as an `anyhow::Error` like [`ContextWindowExceeded`](super::ContextWindowExceeded) and
//! is found the same way, by downcast.

use std::fmt;

use super::LmErrorKind;

/// An LM call that failed, with what the provider said about it.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct LmFailure {
    pub kind: LmErrorKind,
    pub message: String,
    /// The model asked, which upstream puts in the rendered message as `[model] …`.
    pub model: Option<String>,
    /// Which wire answered — `openai`, `anthropic`, `ollama`.
    pub provider: Option<String>,
    /// The provider's own error code, when it sent one, distinct from [`LmErrorKind::code`].
    pub provider_code: Option<String>,
    pub status: Option<u16>,
    pub request_id: Option<String>,
    /// Seconds the provider asked the caller to wait, from `Retry-After`.
    pub retry_after: Option<f64>,
    /// dspy `LMUnsupportedFeatureError.features`: which features were asked for and refused —
    /// `"temperature"`, `"reasoning"`. Empty for every other kind, as upstream's is.
    ///
    /// The kind alone says a feature was refused; this says which, which is what a caller needs to
    /// drop one and retry rather than give up on the call.
    pub features: Vec<String>,
    /// dspy `LMUnsupportedFeatureError.issues`: why each refusal happened, in the caller's terms.
    pub issues: Vec<String>,
}

impl Default for LmErrorKind {
    fn default() -> Self {
        Self::Unexpected
    }
}

impl LmFailure {
    /// A failure of this kind, with nothing known about it yet.
    pub fn new(kind: LmErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            ..Self::default()
        }
    }

    /// What the provider's HTTP status says this is — dspy's `_lm_error_class_from_status`.
    pub fn from_status(status: u16, message: impl Into<String>) -> Self {
        Self {
            kind: LmErrorKind::from_status(Some(status)),
            message: message.into(),
            status: Some(status),
            ..Self::default()
        }
    }

    /// What an OpenAI-shaped or Anthropic-shaped error body says, as a failure.
    ///
    /// Both put the human message at `error.message` and their own code at `error.code`, and
    /// upstream keeps the two apart: the code is the provider's word for this, and the kind is
    /// dspy's. An absent message reads `unknown error`, as it did before this was typed.
    pub fn from_body(status: u16, model: &str, provider: &str, body: &serde_json::Value) -> Self {
        let detail = body["error"]["message"].as_str().unwrap_or("unknown error");
        let mut failure = Self::from_status(status, detail)
            .on_model(model)
            .from_provider(provider);
        if let Some(code) = body["error"]["code"].as_str() {
            failure = failure.provider_code(code);
        }
        failure
    }

    /// A request that never got an answer, as the kind dspy gives it.
    ///
    /// Upstream reads litellm's exception class and message — `"timeout" in class_name`,
    /// `"connection" in message` — because that is all litellm exposes. reqwest says so directly,
    /// which is the same decision made from a better source. Both are retryable, and that is the
    /// point: a refused connection is the most ordinary transient failure there is, and a caller
    /// who cannot see it as one has to match on prose to retry.
    pub fn from_transport(error: &reqwest::Error, model: &str, provider: &str) -> Self {
        let kind = match error.is_timeout() {
            true => LmErrorKind::Timeout,
            false => LmErrorKind::Transport,
        };
        // reqwest's Display names the url and stops, so the reason — timed out, refused, DNS —
        // lives in the source chain. A caller reading only the message would see neither.
        let mut message = error.to_string();
        let mut cause = std::error::Error::source(error);
        while let Some(reason) = cause {
            message.push_str(": ");
            message.push_str(&reason.to_string());
            cause = reason.source();
        }
        Self::new(kind, message)
            .on_model(model)
            .from_provider(provider)
    }

    pub fn on_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    pub fn from_provider(mut self, provider: impl Into<String>) -> Self {
        self.provider = Some(provider.into());
        self
    }

    pub fn request_id(mut self, request_id: impl Into<String>) -> Self {
        self.request_id = Some(request_id.into());
        self
    }

    pub fn retry_after(mut self, seconds: f64) -> Self {
        self.retry_after = Some(seconds);
        self
    }

    /// What the failed response's headers said about it — dspy's `_exception_retry_after` and
    /// `_exception_request_id`, applied together because every provider reads the same two.
    ///
    /// Not folded into [`from_body`](Self::from_body): the body is parsed after the response has
    /// been consumed, so the headers have to be taken before that and handed in.
    pub fn headers(mut self, headers: &reqwest::header::HeaderMap) -> Self {
        self.retry_after = super::headers::retry_after(headers);
        self.request_id = super::headers::request_id(headers);
        self
    }

    pub fn provider_code(mut self, code: impl Into<String>) -> Self {
        self.provider_code = Some(code.into());
        self
    }

    /// Whether asking again is generally safe. See [`LmErrorKind::is_retryable`].
    pub fn is_retryable(&self) -> bool {
        self.kind.is_retryable()
    }
}

impl fmt::Display for LmFailure {
    /// dspy's `DSPyError.__str__`: `[model] message`, and just the model where there is no message.
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (&self.model, self.message.is_empty()) {
            (Some(model), false) => write!(out, "[{model}] {}", self.message),
            (Some(model), true) => write!(out, "[{model}]"),
            (None, _) => out.write_str(&self.message),
        }
    }
}

impl std::error::Error for LmFailure {}

#[cfg(test)]
mod tests {
    use super::*;

    /// Upstream renders `[model] message`, which is the line a caller sees first.
    #[test]
    fn the_model_prefixes_the_message_as_dspy_renders_it() {
        let failed = LmFailure::from_status(429, "slow down").on_model("openai/gpt-4o-mini");
        assert_eq!(failed.to_string(), "[openai/gpt-4o-mini] slow down");
        assert_eq!(
            LmFailure::new(LmErrorKind::Auth, "bad key").to_string(),
            "bad key"
        );
    }

    /// A status decides the kind, so a caller branches on the kind rather than on the number.
    #[test]
    fn a_status_carries_its_kind_and_its_retryability() {
        let limited = LmFailure::from_status(429, "slow down");
        assert_eq!(limited.kind, LmErrorKind::RateLimit);
        assert!(limited.is_retryable());

        let rejected = LmFailure::from_status(401, "bad key");
        assert_eq!(rejected.kind, LmErrorKind::Auth);
        assert!(
            !rejected.is_retryable(),
            "a rejected key fails the same way twice"
        );
    }
}
