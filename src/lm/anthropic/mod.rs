//! Anthropic's messages API.
//!
//! dspy 3.3 reaches Anthropic through litellm, so the [request body](request) is built to match
//! litellm's byte for byte. The reply is read straight from Anthropic's own response: text blocks
//! become the answer, `tool_use` blocks become [`ToolCall`](api::LmPart::ToolCall) parts, and the
//! `json_tool_call` litellm forces for a schema is read back as the structured reply's text.

use std::future::Future;

use anyhow::{Context, Result, anyhow};
use serde_json::{Value, json};

use super::{ChatModel, LmUsage, PROVIDER_TIMEOUT, api};

mod request;
mod stream;

pub(crate) use stream::stream;

pub(super) const MESSAGES_URL: &str = "https://api.anthropic.com/v1/messages";

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
                .json(&request::request(self.model, call))
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

/// The reply as a typed response, or the message Anthropic itself gave for refusing the call. A
/// reply with no readable block stands in as a missing-content error rather than a panic, which the
/// JSON adapters surface as the empty answer it is.
fn reply(model: &str, status: reqwest::StatusCode, body: &Value) -> Result<api::LmResponse> {
    if !status.is_success() {
        let detail = body["error"]["message"].as_str().unwrap_or("unknown error");
        return Err(anyhow!("anthropic {status}: {detail}"));
    }
    let output = output_of(body);
    if output.parts.is_empty() {
        return Err(anyhow!("anthropic returned no content"));
    }
    Ok(api::LmResponse {
        outputs: vec![output],
        ..api::LmResponse::default()
    }
    .with_usage(usage(&body["usage"]))
    .with_provider_response(provider_data(body))
    .with_model(model))
}

/// The message's content blocks as a typed output: text as text, each `tool_use` as a tool call,
/// and why generation stopped.
fn output_of(body: &Value) -> api::LmOutput {
    let mut parts = Vec::new();
    for block in body["content"].as_array().into_iter().flatten() {
        match block["type"].as_str() {
            Some("text") => {
                if let Some(text) = block["text"].as_str().filter(|text| !text.is_empty()) {
                    parts.push(api::LmPart::text(text));
                }
            }
            Some("tool_use") => parts.push(tool_use(block)),
            _ => {}
        }
    }
    let reason = body["stop_reason"].as_str();
    api::LmOutput {
        parts,
        truncated: reason == Some("max_tokens"),
        finish_reason: reason.map(str::to_owned),
        ..api::LmOutput::default()
    }
}

/// One `tool_use` block. The `json_tool_call` litellm forces for a schema is not a tool the caller
/// asked for — its arguments are the structured reply, so they are surfaced as the answer's text,
/// which is what makes json mode on Anthropic parse the way every other provider's does.
fn tool_use(block: &Value) -> api::LmPart {
    if block["name"] == "json_tool_call" {
        return api::LmPart::text(block["input"].to_string());
    }
    api::LmPart::ToolCall {
        id: block["id"].as_str().map(str::to_owned),
        name: block["name"].as_str().unwrap_or_default().to_owned(),
        args: block["input"].as_object().cloned().unwrap_or_default(),
        provider_data: api::Metadata::new(),
        metadata: api::Metadata::new(),
    }
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

    /// A function-calling reply carries its calls as `ToolCall` parts, the arguments read straight
    /// from Anthropic's `input` object, and the stop reason kept.
    #[test]
    fn a_tool_use_reply_parses_into_a_tool_call_part() {
        let body = json!({
            "content": [{
                "type": "tool_use",
                "id": "toolu_1",
                "name": "get_weather",
                "input": { "city": "Paris" },
            }],
            "stop_reason": "tool_use",
        });
        let response = reply("claude-3-5-sonnet", reqwest::StatusCode::OK, &body).expect("parses");
        let output = &response.outputs[0];
        assert_eq!(output.finish_reason.as_deref(), Some("tool_use"));
        let api::LmPart::ToolCall { id, name, args, .. } = &output.parts[0] else {
            panic!("expected a tool call, got {:?}", output.parts)
        };
        assert_eq!(id.as_deref(), Some("toolu_1"));
        assert_eq!(name, "get_weather");
        assert_eq!(args["city"], json!("Paris"));
    }

    /// The other half of the forced `json_tool_call`: Anthropic answers a schema by calling that
    /// tool, and its arguments — not a text block — are the reply an adapter then parses.
    #[test]
    fn the_forced_json_tool_call_is_read_back_as_the_reply_text() {
        let body = json!({
            "content": [{
                "type": "tool_use",
                "id": "toolu_2",
                "name": "json_tool_call",
                "input": { "answer": "Paris" },
            }],
            "stop_reason": "tool_use",
        });
        let response = reply("claude-3-5-sonnet", reqwest::StatusCode::OK, &body).expect("parses");
        assert_eq!(
            serde_json::from_str::<Value>(&response.first_text()).expect("the reply is json"),
            json!({ "answer": "Paris" }),
            "the tool's arguments are surfaced as the answer, not lost"
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

    /// A reply cut off at the cap is marked truncated, `max_tokens` being Anthropic's name for it.
    #[test]
    fn a_stop_reason_of_max_tokens_marks_the_output_truncated() {
        let body = json!({
            "content": [{ "type": "text", "text": "hi" }],
            "stop_reason": "max_tokens",
        });
        assert!(
            reply("claude-opus-4-8", reqwest::StatusCode::OK, &body)
                .expect("a reply")
                .outputs[0]
                .truncated
        );
    }
}
