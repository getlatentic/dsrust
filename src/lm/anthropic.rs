//! Anthropic's messages API.
//!
//! It differs from the OpenAI-shaped services in three ways that matter here: the system
//! prompt is its own top-level field rather than the first message, a generation cap is
//! mandatory rather than optional, and structured output travels under `output_config`.

use anyhow::{Context, Result, anyhow};
use serde_json::{Value, json};

use super::{LmRequest, OutputMode, PROVIDER_TIMEOUT, turn_json};

const MESSAGES_URL: &str = "https://api.anthropic.com/v1/messages";

/// Anthropic rejects a request that names no cap, so one is always sent.
const MAX_OUTPUT_TOKENS: u32 = 1024;

pub(super) async fn chat(
    http: &reqwest::Client,
    model: &str,
    api_key: Option<&str>,
    call: &LmRequest<'_>,
) -> Result<String> {
    let key = api_key.ok_or_else(|| anyhow!("ANTHROPIC_API_KEY is not set"))?;
    let response = http
        .post(MESSAGES_URL)
        .header("x-api-key", key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .timeout(PROVIDER_TIMEOUT)
        .json(&request(model, call))
        .send()
        .await
        .context("anthropic request failed")?;
    let status = response.status();
    let body: Value = response
        .json()
        .await
        .context("anthropic response was not JSON")?;
    reply(status, &body)
}

fn request(model: &str, call: &LmRequest<'_>) -> Value {
    let mut request = json!({
        "model": model,
        "max_tokens": call.sampling.max_tokens.unwrap_or(MAX_OUTPUT_TOKENS),
        "system": call.system,
        "messages": call.turns.iter().map(turn_json).collect::<Vec<_>>(),
    });
    if let Some(temperature) = call.sampling.temperature {
        request["temperature"] = json!(temperature);
    }
    if let OutputMode::Json { schema } = call.mode {
        request["output_config"] = json!({ "format": { "type": "json_schema", "schema": schema } });
    }
    request
}

/// The first text block, or the message Anthropic itself gave for refusing the call. A reply
/// carrying only non-text blocks stands in as an empty object, which the JSON adapters parse
/// into a missing-field error rather than a panic.
fn reply(status: reqwest::StatusCode, body: &Value) -> Result<String> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lm::Sampling;

    fn sampled(sampling: Sampling) -> LmRequest<'static> {
        LmRequest::new("be helpful", &[], OutputMode::Text).sampled(sampling)
    }

    /// Anthropic rejects a request with no cap, so the default has to survive an otherwise
    /// empty `Sampling` rather than be left off along with the rest.
    #[test]
    fn every_call_is_capped_and_a_temperature_is_named_only_when_asked() {
        let default = request("claude-opus-4-8", &sampled(Sampling::default()));
        assert_eq!(default["max_tokens"], MAX_OUTPUT_TOKENS);
        assert_eq!(default.get("temperature"), None);

        let named = request(
            "claude-opus-4-8",
            &sampled(Sampling {
                temperature: Some(1.0),
                max_tokens: Some(64),
            }),
        );
        assert_eq!(named["max_tokens"], 64);
        assert_eq!(named["temperature"], 1.0);
    }

    /// The system prompt is a field of its own here, not the leading message.
    #[test]
    fn the_system_prompt_stays_out_of_the_message_list() {
        let body = request("claude-opus-4-8", &sampled(Sampling::default()));
        assert_eq!(body["system"], "be helpful");
        assert_eq!(body["messages"].as_array().expect("a list").len(), 0);
    }

    #[test]
    fn a_refusal_carries_the_reason_anthropic_gave() {
        let error = reply(
            reqwest::StatusCode::BAD_REQUEST,
            &json!({ "error": { "message": "credit balance is too low" } }),
        )
        .expect_err("a 400 is an error");
        assert!(error.to_string().contains("credit balance is too low"));
    }

    #[test]
    fn the_first_text_block_is_the_reply() {
        let body = json!({ "content": [
            { "type": "thinking", "thinking": "hmm" },
            { "type": "text", "text": "the reply" },
        ]});
        assert_eq!(
            reply(reqwest::StatusCode::OK, &body).expect("a text block"),
            "the reply"
        );
    }
}
