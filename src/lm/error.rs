//! The one error whose *identity* a module branches on.
//!
//! dspy has a taxonomy of about eighteen error types; the crate answers with `anyhow::Error`
//! throughout, and for almost all of them that is the same thing said differently — a caller reads
//! the message either way. One is not: `ContextWindowExceededError` is caught by name, and what
//! happens next is different from what happens on any other failure. `ReAct` trims the oldest tool
//! call out of its trajectory and asks again rather than giving up, which is the difference between
//! a long agent run finishing and failing.
//!
//! So this is that one type, recognised off what a provider actually says. It travels as an
//! `anyhow::Error` like everything else and is found by downcast, the way
//! [`FieldMismatch`](crate::adapter::parse::FieldMismatch) already is. The rest of the taxonomy is
//! issue #10.

use serde_json::Value;

/// A request the model refused because the prompt was longer than it can read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextWindowExceeded {
    pub model: String,
    pub message: String,
}

impl std::fmt::Display for ContextWindowExceeded {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(out, "context window exceeded for {}: {}", self.model, self.message)
    }
}

impl std::error::Error for ContextWindowExceeded {}

impl ContextWindowExceeded {
    /// This refusal where the provider's is one, read off what it said.
    ///
    /// No provider agrees on how to say it. OpenAI has a machine-readable `code`; Anthropic and
    /// ollama say it in prose, and prose is what there is to read. litellm — which is what dspy
    /// sees — matches on the same phrases, so this is the same evidence upstream acts on rather
    /// than a looser guess.
    pub fn detected(model: &str, body: &Value) -> Option<Self> {
        let error = body.get("error").unwrap_or(body);
        let code = error.get("code").and_then(Value::as_str).unwrap_or_default();
        let message = match error {
            Value::String(text) => text.clone(),
            _ => error.get("message").and_then(Value::as_str).unwrap_or_default().to_owned(),
        };
        let says_so = code == "context_length_exceeded"
            || code == "context_window_exceeded"
            || names_the_limit(&message.to_ascii_lowercase());
        says_so.then(|| Self { model: model.to_owned(), message })
    }
}

/// The phrases a provider uses for it. Each is a whole clause rather than a word, because a word
/// like "token" appears in refusals that are not this one — an invalid token, a token budget for
/// something else — and treating those as this would have `ReAct` trim its trajectory over a
/// failure trimming cannot fix.
fn names_the_limit(message: &str) -> bool {
    const PHRASES: [&str; 6] = [
        "maximum context length",
        "context length exceeded",
        "context window",
        "prompt is too long",
        "too many tokens",
        "reduce the length of the messages",
    ];
    PHRASES.iter().any(|phrase| message.contains(phrase))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn openais_machine_readable_code_is_read() {
        let refusal = json!({
            "error": {
                "code": "context_length_exceeded",
                "message": "This model's maximum context length is 8192 tokens.",
            }
        });
        let found = ContextWindowExceeded::detected("openai/gpt-4", &refusal).expect("detected");
        assert_eq!(found.model, "openai/gpt-4");
        assert!(found.message.starts_with("This model's maximum context length"));
    }

    /// Anthropic and ollama say it in prose, which is all there is to read.
    #[test]
    fn a_prose_refusal_is_read_too() {
        for message in [
            "prompt is too long: 210000 tokens > 200000 maximum",
            "Please reduce the length of the messages.",
            "input exceeds the context window for this model",
        ] {
            let refusal = json!({ "error": { "message": message } });
            assert!(
                ContextWindowExceeded::detected("anthropic/claude", &refusal).is_some(),
                "should be detected: {message}"
            );
        }
        // ollama answers with a bare string under `error`.
        assert!(
            ContextWindowExceeded::detected(
                "ollama_chat/qwen",
                &json!({ "error": "too many tokens for this model" })
            )
            .is_some()
        );
    }

    /// Every other refusal is left alone. Trimming a trajectory does not fix an expired key, and
    /// reading one as the other would spend three more calls before failing anyway.
    #[test]
    fn another_refusal_is_not_mistaken_for_this_one() {
        for refusal in [
            json!({ "error": { "code": "invalid_api_key", "message": "Incorrect API key provided" } }),
            json!({ "error": { "code": "rate_limit_exceeded", "message": "Rate limit reached for tokens" } }),
            json!({ "error": { "message": "The model produced invalid tokens" } }),
            json!({ "error": { "message": "insufficient_quota" } }),
            json!({}),
        ] {
            assert_eq!(
                ContextWindowExceeded::detected("openai/gpt-4", &refusal),
                None,
                "should not be detected: {refusal}"
            );
        }
    }
}
