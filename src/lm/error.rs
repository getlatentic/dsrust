//! The one error whose *identity* a module branches on.
//!
//! dspy has a taxonomy of about eighteen error types; the crate answers with `anyhow::Error`
//! throughout, and for almost all of them that is the same thing said differently — a caller reads
//! the message either way. One is not: `ContextWindowExceededError` is caught by name, and what
//! happens next is different from what happens on any other failure. `ReAct` trims the oldest tool
//! call out of its trajectory and asks again rather than giving up, which is the difference between
//! a long agent run finishing and failing.
//!
//! It travels as an `anyhow::Error` like everything else and is found by downcast, the way
//! [`FieldMismatch`](crate::adapter::parse::FieldMismatch) already is. The rest of the taxonomy is
//! issue #10.
//!
//! **What decides it.** dspy does not decide this itself: `_wrap_litellm_exception` asks
//! `isinstance(error, litellm.ContextWindowExceededError)`, and litellm decides by matching the
//! provider's error *string*. So litellm's list is the rule dspy actually acts on, and it is
//! reproduced here rather than approximated — including the two exclusions, which exist because a
//! parameter-validation error says "string too long" and is not this.

use serde_json::Value;

/// A request the model refused because the prompt was longer than it can read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextWindowExceeded {
    pub model: String,
    pub message: String,
}

impl std::fmt::Display for ContextWindowExceeded {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            out,
            "context window exceeded for {}: {}",
            self.model, self.message
        )
    }
}

impl std::error::Error for ContextWindowExceeded {}

impl ContextWindowExceeded {
    /// This refusal where the provider's is one, read off what it said.
    pub fn detected(model: &str, body: &Value) -> Option<Self> {
        let error = body.get("error").unwrap_or(body);
        let message = match error {
            Value::String(text) => text.clone(),
            _ => error
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
        };
        // OpenAI's machine-readable code. litellm has no equivalent branch — its message for this
        // code contains "this model's maximum context length is", so the substring catches it
        // there. Reading the code as well costs nothing and does not depend on prose.
        let code = error
            .get("code")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let says_so = code == "context_length_exceeded" || names_the_limit(&message);
        says_so.then(|| Self {
            model: model.to_owned(),
            message,
        })
    }
}

/// litellm's `ExceptionCheckers.is_error_str_context_window_exceeded`, plus the two provider
/// branches that add to it, since that is the whole of what dspy ends up acting on.
fn names_the_limit(message: &str) -> bool {
    let message = message.to_ascii_lowercase();

    // The exclusions come first, and they are the point of having read upstream rather than
    // guessed: OpenAI's `user` parameter has a 64-character cap, and refusing it says "string too
    // long" — which is a substring below. Trimming a trajectory cannot fix a bad parameter.
    if message.contains("string_above_max_length")
        || (message.contains("invalid 'user'") && message.contains("string too long"))
    {
        return false;
    }

    const PHRASES: [&str; 12] = [
        "exceed context limit",
        "this model's maximum context length is",
        "string too long. expected a string with maximum length",
        "model's maximum context limit",
        "is longer than the model's context length",
        "input tokens exceed the configured limit",
        "`inputs` tokens + `max_new_tokens` must be",
        // llama.cpp and Lemonade.
        "exceeds the available context size",
        // Gemini.
        "exceeds the maximum number of tokens allowed",
        // litellm's Anthropic branch adds these two beside the shared list.
        "prompt is too long",
        "prompt: length",
        // Its Replicate branch; kept because a compatible endpoint may relay the same words.
        "input is too long",
    ];
    if PHRASES.iter().any(|phrase| message.contains(phrase)) {
        return true;
    }

    // Two providers say it as a pair of clauses rather than one phrase.
    (message.contains("current length is") && message.contains("while limit is"))
        || (message.contains("maximum input length is") && message.contains("tokens"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn detected(message: &str) -> bool {
        ContextWindowExceeded::detected("m", &json!({ "error": { "message": message } })).is_some()
    }

    /// Every phrase litellm matches, which is what dspy ends up acting on. Taken from
    /// `ExceptionCheckers.is_error_str_context_window_exceeded` and the two provider branches
    /// beside it, rather than from a guess at how a provider might word it.
    #[test]
    fn every_phrase_litellm_matches_is_matched() {
        for message in [
            "Requested tokens exceed context limit of 8192",
            "This model's maximum context length is 8192 tokens, however you requested 9000",
            "string too long. Expected a string with maximum length 4096",
            "Input exceeds model's maximum context limit",
            "Your input is longer than the model's context length",
            "Input tokens exceed the configured limit for this deployment",
            "`inputs` tokens + `max_new_tokens` must be <= 2048",
            "the request exceeds the available context size",
            "The input token count exceeds the maximum number of tokens allowed",
            "prompt is too long: 210000 tokens > 200000 maximum",
            "prompt: length exceeds the limit",
            "input is too long for this model",
            "Current length is 9000 while limit is 8192",
            "The maximum input length is 4096 tokens",
        ] {
            assert!(
                detected(message),
                "litellm matches this and so must we: {message}"
            );
        }
    }

    /// litellm's own exclusions. A parameter that is too long is not a prompt that is too long, and
    /// trimming a trajectory cannot fix one — it would spend three more calls first.
    #[test]
    fn a_parameter_that_is_too_long_is_not_this() {
        assert!(!detected(
            "Invalid 'user': string too long. Expected a string with maximum length 64"
        ));
        assert!(!detected(
            "string_above_max_length: expected a string with maximum length 256"
        ));
    }

    /// OpenAI's machine-readable code, which litellm has no branch for — its message carries a
    /// matched phrase, so reading the code as well is ours and costs nothing.
    #[test]
    fn openais_code_is_read_as_well_as_its_message() {
        let refusal = json!({
            "error": { "code": "context_length_exceeded", "message": "no phrase here" }
        });
        assert!(ContextWindowExceeded::detected("openai/gpt-4", &refusal).is_some());
    }

    /// ollama answers with a bare string under `error` rather than an object.
    #[test]
    fn a_bare_string_body_is_read() {
        assert!(
            ContextWindowExceeded::detected(
                "ollama_chat/qwen",
                &json!({ "error": "this model's maximum context length is 4096" })
            )
            .is_some()
        );
    }

    /// Every other refusal is left alone. Trimming does not fix an expired key or a rate limit.
    #[test]
    fn another_refusal_is_not_mistaken_for_this_one() {
        for refusal in [
            json!({ "error": { "code": "invalid_api_key", "message": "Incorrect API key provided" } }),
            json!({ "error": { "code": "rate_limit_exceeded", "message": "Rate limit reached for tokens" } }),
            json!({ "error": { "message": "The model produced invalid tokens" } }),
            json!({ "error": { "message": "insufficient_quota" } }),
            json!({ "error": { "message": "context window" } }),
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
