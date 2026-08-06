//! An ollama server, over either of the two routes litellm exposes.
//!
//! litellm splits ollama across two providers on two endpoints: `ollama_chat/` speaks `/api/chat`
//! — a message list, native tool calls — and `ollama/` speaks `/api/generate`, an older path that
//! takes one flattened prompt and cannot carry tools natively. dsrust mirrors the split: the prefix
//! a caller writes picks the route, exactly as it does in dspy through litellm.
//!
//! The two [`ChatModel`]s live in [`chat`] and [`generate`]; this module holds what they share —
//! the credential a hosted server wants, and the counts and stop reason both endpoints report the
//! same way.

use serde_json::{Value, json};

use super::LmUsage;

mod capabilities;
mod chat;
mod generate;
mod request;
mod stream;

pub(crate) use capabilities::capabilities;
pub(crate) use chat::Chat;
pub(crate) use generate::Generate;
pub(crate) use generate::stream as generate_stream;
pub(crate) use stream::stream as chat_stream;

/// Carry the credential a hosted ollama needs, on whichever call is being made.
///
/// litellm sends `OLLAMA_API_KEY` as a bearer token, and it has to reach every endpoint this
/// crate touches: a server that authenticates `/api/chat` authenticates `/api/show` too, so a
/// probe that skipped it would report that a perfectly capable model can do nothing.
pub(super) fn authorized(
    request: reqwest::RequestBuilder,
    api_key: Option<&str>,
) -> reqwest::RequestBuilder {
    match api_key {
        Some(key) => request.bearer_auth(key),
        None => request,
    }
}

/// ollama counts at the top level rather than under a usage object, and names the two counts
/// after the passes that produce them. Both endpoints report them the same way.
pub(super) fn usage(body: &Value) -> Option<LmUsage> {
    let input = body["prompt_eval_count"].as_u64();
    let output = body["eval_count"].as_u64();
    // A count the provider omitted stays unknown rather than becoming zero, which is what
    // optional counters buy: reporting one of the two is now sayable.
    (input.is_some() || output.is_some()).then(|| {
        LmUsage {
            input_tokens: input.map(|count| count as u32),
            output_tokens: output.map(|count| count as u32),
            ..LmUsage::default()
        }
        .fill_aliases()
    })
}

/// ollama's own name for why generation stopped, which is `length` when the reply hit
/// `num_predict`.
pub(super) fn provider_data(body: &Value) -> Option<Value> {
    let done_reason = body["done_reason"].as_str()?;
    Some(json!({ "done_reason": done_reason }))
}

/// What ollama said when it refused the call.
///
/// Its errors carry a bare `{"error": "…"}` rather than OpenAI's nested `error.message`, and
/// without it a caller gets only a status — a 500 on an over-long prompt reads exactly like a 500
/// on a broken request. The other providers already surface their own message; this is ollama's.
pub(crate) fn refusal(body: &serde_json::Value) -> String {
    body["error"].as_str().unwrap_or("unknown error").to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The counters are optional on each side independently: ollama omits `prompt_eval_count`
    /// entirely on a fully cached prompt, and the one it did report must survive alone.
    #[test]
    fn one_sided_counts_still_report_and_absent_ones_stay_unknown() {
        let cached = usage(&json!({ "eval_count": 9 })).expect("output was counted");
        assert_eq!(cached.input_tokens, None, "omitted is unknown, not zero");
        assert_eq!(cached.output_tokens, Some(9));
        let input_only = usage(&json!({ "prompt_eval_count": 4 })).expect("input was counted");
        assert_eq!(input_only.input_tokens, Some(4));
        assert_eq!(input_only.output_tokens, None);
        assert_eq!(usage(&json!({ "done": true })), None, "neither counted");
    }

    /// ollama's error body is a bare `{"error": …}`, and the message is the difference between
    /// "prompt too long" and a blank 500.
    #[test]
    fn the_refusal_is_ollamas_own_message() {
        assert_eq!(
            refusal(&json!({ "error": "model 'x' not found" })),
            "model 'x' not found"
        );
        assert_eq!(refusal(&json!({})), "unknown error");
    }
}
