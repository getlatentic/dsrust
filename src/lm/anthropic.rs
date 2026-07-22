//! Anthropic's messages API.
//!
//! It differs from the OpenAI-shaped services in three ways that matter here: the system
//! prompt is its own top-level field rather than the first message, a generation cap is
//! mandatory rather than optional, and structured output travels under `output_config`.

use std::future::Future;

use anyhow::{Context, Result, anyhow};
use serde_json::{Value, json};

use super::{ChatModel, LmUsage, PROVIDER_TIMEOUT, api};

const MESSAGES_URL: &str = "https://api.anthropic.com/v1/messages";

/// Anthropic rejects a request that names no cap, so one is always sent.
const MAX_OUTPUT_TOKENS: u32 = 1024;

/// Anthropic's messages API as a [`ChatModel`], the model and credential it needs held beside it.
pub(crate) struct Anthropic<'a> {
    pub model: &'a str,
    pub api_key: Option<&'a str>,
}

impl ChatModel for Anthropic<'_> {
    fn forward<'a>(
        &'a self,
        http: &'a reqwest::Client,
        call: &'a api::LmRequest,
    ) -> impl Future<Output = Result<api::LmResponse>> + Send + 'a {
        async move {
            let key = self
                .api_key
                .ok_or_else(|| anyhow!("ANTHROPIC_API_KEY is not set"))?;
            let response = http
                .post(MESSAGES_URL)
                .header("x-api-key", key)
                .header("anthropic-version", "2023-06-01")
                .header("content-type", "application/json")
                .timeout(PROVIDER_TIMEOUT)
                .json(&request(self.model, call))
                .send()
                .await
                .context("anthropic request failed")?;
            let status = response.status();
            let body: Value = response
                .json()
                .await
                .context("anthropic response was not JSON")?;
            reply(self.model, status, &body)
        }
    }
}

fn request(model: &str, call: &api::LmRequest) -> Value {
    let mut request = json!({
        "model": model,
        "max_tokens": call.config.max_tokens.unwrap_or(MAX_OUTPUT_TOKENS),
        "system": call.system(),
        "messages": call.user_messages(),
    });
    if let Some(temperature) = call.config.temperature {
        request["temperature"] = json!(temperature);
    }
    if let Some(schema) = call.output_schema() {
        request["output_config"] = json!({ "format": { "type": "json_schema", "schema": schema } });
    }
    request
}

/// The first text block, or the message Anthropic itself gave for refusing the call. A reply
/// carrying only non-text blocks stands in as an empty object, which the JSON adapters parse
/// into a missing-field error rather than a panic.
fn reply(model: &str, status: reqwest::StatusCode, body: &Value) -> Result<api::LmResponse> {
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
    Ok(api::LmResponse::completions([text.to_owned()])
        .with_usage(usage(&body["usage"]))
        .with_provider_response(provider_data(body))
        .with_model(model))
}

/// Anthropic's cache counters are charged separately from `input_tokens`, so a cached call whose
/// prompt tokens were all read from cache reports zero input without being free.
fn usage(usage: &Value) -> Option<LmUsage> {
    let count = |key: &str| usage[key].as_u64().unwrap_or(0) as u32;
    let input = count("input_tokens") + count("cache_creation_input_tokens");
    let output = count("output_tokens");
    (usage.is_object() && (input > 0 || output > 0)).then(|| {
        LmUsage {
            input_tokens: Some(input + count("cache_read_input_tokens")),
            output_tokens: Some(output),
            // Anthropic states its cache work separately, and upstream has counters for it.
            // They are additional detail rather than a different total: a cache read is already
            // inside `input_tokens` above, which is what this crate has always reported.
            cache_read_tokens: reported(count("cache_read_input_tokens")),
            cache_write_tokens: reported(count("cache_creation_input_tokens")),
            ..LmUsage::default()
        }
        .fill_aliases()
    })
}

/// A counter a provider did not mention reads as unknown rather than as zero.
fn reported(count: u32) -> Option<u32> {
    (count > 0).then_some(count)
}

/// Why generation stopped, which is the one thing here a caller acts on: `max_tokens` means the
/// reply was cut off, so a parse failure is a budget problem rather than a prompt problem.
fn provider_data(body: &Value) -> Option<Value> {
    let stop_reason = body["stop_reason"].as_str()?;
    Some(json!({ "stop_reason": stop_reason }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lm::api::interop::raise_request;
    use crate::lm::{LmConfig, OutputMode};

    fn sampled(config: LmConfig) -> api::LmRequest {
        raise_request("be helpful", &[], OutputMode::Text, &config)
    }

    /// Anthropic rejects a request with no cap, so the default has to survive an otherwise
    /// empty `LmConfig` rather than be left off along with the rest.
    #[test]
    fn every_call_is_capped_and_a_temperature_is_named_only_when_asked() {
        let default = request("claude-opus-4-8", &sampled(LmConfig::default()));
        assert_eq!(default["max_tokens"], MAX_OUTPUT_TOKENS);
        assert_eq!(default.get("temperature"), None);

        let named = request(
            "claude-opus-4-8",
            &sampled(LmConfig {
                temperature: Some(1.0),
                max_tokens: Some(64),
                ..LmConfig::default()
            }),
        );
        assert_eq!(named["max_tokens"], 64);
        assert_eq!(named["temperature"], 1.0);
    }

    /// The system prompt is a field of its own here, not the leading message.
    #[test]
    fn the_system_prompt_stays_out_of_the_message_list() {
        let body = request("claude-opus-4-8", &sampled(LmConfig::default()));
        assert_eq!(body["system"], "be helpful");
        assert_eq!(body["messages"].as_array().expect("a list").len(), 0);
    }

    #[test]
    fn a_refusal_carries_the_reason_anthropic_gave() {
        let error = reply(
            "claude-opus-4-8",
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
            reply("claude-opus-4-8", reqwest::StatusCode::OK, &body)
                .expect("a text block")
                .first_text(),
            "the reply"
        );
    }

    /// Anthropic charges cache reads and cache writes on their own counters, so a caller adding
    /// up `input_tokens` alone would under-report a prompt that was mostly cached.
    #[test]
    fn every_input_counter_is_charged_to_the_input_total() {
        let body = json!({
            "content": [{ "type": "text", "text": "hi" }],
            "usage": {
                "input_tokens": 10,
                "cache_creation_input_tokens": 4,
                "cache_read_input_tokens": 100,
                "output_tokens": 7,
            },
        });
        let usage = reply("claude-opus-4-8", reqwest::StatusCode::OK, &body)
            .expect("a reply")
            .usage
            .expect("a usage block");
        assert_eq!(usage.input_tokens, Some(114));
        assert_eq!(usage.output_tokens, Some(7));
    }

    /// A provider that reported nothing must not read as a call that cost nothing.
    #[test]
    fn a_reply_with_no_usage_block_reports_none() {
        let body = json!({ "content": [{ "type": "text", "text": "hi" }] });
        assert_eq!(
            reply("claude-opus-4-8", reqwest::StatusCode::OK, &body)
                .expect("a reply")
                .usage,
            None
        );
    }

    /// The one field a caller acts on: a reply cut off at the cap failed for a reason the prompt
    /// cannot fix.
    #[test]
    fn the_stop_reason_survives_as_provider_data() {
        let body = json!({
            "content": [{ "type": "text", "text": "hi" }],
            "stop_reason": "max_tokens",
        });
        let data = reply("claude-opus-4-8", reqwest::StatusCode::OK, &body)
            .expect("a reply")
            .provider_response
            .expect("a stop reason");
        assert_eq!(data["stop_reason"], "max_tokens");
    }
}
