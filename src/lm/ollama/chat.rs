//! The ollama `/api/chat` route — litellm's `ollama_chat/` provider.
//!
//! A message list in, a `message` object out, and native tool calls on both sides. This is the
//! route to reach for tool use; `generate` is the older one-prompt path. The [request body](request)
//! is built the way litellm builds it, and the reply is read straight from ollama's response: the
//! message content is the answer and any `tool_calls` become [`ToolCall`](api::LmPart::ToolCall)
//! parts.

use std::future::Future;
use std::time::Duration;

use anyhow::{Context, Result, anyhow};

use super::refusal;
use serde_json::Value;

use super::request::request;
use super::{authorized, provider_data, usage};
use crate::lm::api;
use crate::lm::ChatModel;

/// An ollama server reached over `/api/chat`, the model and host beside it.
pub(crate) struct Chat<'a> {
    pub model: &'a str,
    pub host: &'a str,
    pub api_key: Option<&'a str>,
    pub timeout: Duration,
}

impl ChatModel for Chat<'_> {
    fn forward<'a>(
        &'a self,
        http: &'a reqwest::Client,
        call: &'a api::LmRequest,
    ) -> impl Future<Output = Result<api::LmResponse>> + Send + 'a {
        async move {
            let request = http
                .post(format!("{}/api/chat", self.host))
                .timeout(self.timeout)
                .json(&request(self.model, call));
            let response = authorized(request, self.api_key)
                .send()
                .await
                .context("ollama request failed")?;
            let status = response.status();
            let body: Value = response.json().await.context("ollama response was not JSON")?;
            if !status.is_success() {
                return Err(anyhow!("ollama {status}: {}", refusal(&body)));
            }
            reply(self.model, &body)
        }
    }
}

/// The reply as a typed response. A reply with neither content nor a tool call is nothing to parse,
/// so it is surfaced as the error it is rather than an empty answer.
fn reply(model: &str, body: &Value) -> Result<api::LmResponse> {
    let output = output_of(body);
    if output.parts.is_empty() {
        return Err(anyhow!("ollama returned no content"));
    }
    Ok(api::LmResponse { outputs: vec![output], ..api::LmResponse::default() }
        .with_usage(usage(body))
        .with_provider_response(provider_data(body))
        .with_model(model))
}

/// The message as a typed output: a reasoning model's `thinking` as a thinking part first, its
/// content as text, each `tool_calls` entry as a tool call, and why generation stopped — `length`
/// being ollama's name for a reply cut off at the cap.
fn output_of(body: &Value) -> api::LmOutput {
    let message = &body["message"];
    let mut parts = Vec::new();
    if let Some(thinking) = message["thinking"].as_str().filter(|text| !text.is_empty()) {
        parts.push(api::LmPart::thinking(thinking, false));
    }
    if let Some(content) = message["content"].as_str().filter(|text| !text.is_empty()) {
        parts.push(api::LmPart::text(content));
    }
    for call in message["tool_calls"].as_array().into_iter().flatten() {
        parts.push(tool_call(call));
    }
    let reason = body["done_reason"].as_str();
    api::LmOutput {
        parts,
        truncated: reason == Some("length"),
        finish_reason: reason.map(str::to_owned),
        ..api::LmOutput::default()
    }
}

/// One ollama tool call. Its arguments arrive already parsed as an object, not the JSON string
/// OpenAI hands them in, and ollama assigns no call id.
fn tool_call(call: &Value) -> api::LmPart {
    api::LmPart::ToolCall {
        id: None,
        name: call["function"]["name"].as_str().unwrap_or_default().to_owned(),
        args: call["function"]["arguments"].as_object().cloned().unwrap_or_default(),
        provider_data: api::Metadata::new(),
        metadata: api::Metadata::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// ollama names its counts after the passes that produce them and puts them at the top
    /// level, so nothing about reading them looks like the other two providers.
    #[test]
    fn the_eval_counts_become_the_shared_usage() {
        let body = json!({
            "message": { "content": "the reply" },
            "prompt_eval_count": 26,
            "eval_count": 298,
            "done_reason": "length",
        });
        let answered = reply("qwen2.5:7b-instruct", &body).expect("a reply");
        assert_eq!(answered.first_text(), "the reply");
        let usage = answered.usage.expect("counts");
        assert_eq!(usage.input_tokens, Some(26));
        assert_eq!(usage.output_tokens, Some(298));
        assert_eq!(
            answered.provider_response.expect("a done reason")["done_reason"],
            "length"
        );
    }

    /// A reasoning model on ollama returns its reasoning in a `thinking` field beside the content;
    /// it becomes a thinking part before the text, as it does on the other providers.
    #[test]
    fn a_thinking_field_becomes_a_thinking_part_before_the_text() {
        let body = json!({
            "message": { "role": "assistant", "thinking": "2 + 2 = 4", "content": "The answer is 4." },
            "done_reason": "stop",
        });
        let output = &reply("qwen3:4b", &body).expect("a reply").outputs[0];
        assert!(
            matches!(&output.parts[0], api::LmPart::Thinking { text, .. } if text == "2 + 2 = 4"),
            "thinking is first, got {:?}", output.parts,
        );
        assert!(matches!(&output.parts[1], api::LmPart::Text { text, .. } if text == "The answer is 4."));
    }

    /// A tool-calling reply carries its calls as `ToolCall` parts — the arguments read straight from
    /// ollama's already-parsed `arguments` object, with no call id since ollama assigns none.
    #[test]
    fn a_tool_call_reply_parses_into_a_tool_call_part() {
        let body = json!({
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{ "function": { "name": "get_weather", "arguments": { "city": "Paris" } } }],
            },
            "done_reason": "stop",
        });
        let output = &reply("llama3.2", &body).expect("parses").outputs[0];
        let api::LmPart::ToolCall { id, name, args, .. } = &output.parts[0] else {
            panic!("expected a tool call, got {:?}", output.parts)
        };
        assert_eq!(*id, None);
        assert_eq!(name, "get_weather");
        assert_eq!(args["city"], json!("Paris"));
    }

    #[test]
    fn a_reply_reporting_no_counts_reports_no_usage() {
        let body = json!({ "message": { "content": "the reply" } });
        assert_eq!(reply("qwen2.5:7b-instruct", &body).expect("a reply").usage, None);
    }

    #[test]
    fn a_reply_carrying_no_content_is_an_error() {
        let error = reply("qwen2.5:7b-instruct", &json!({ "message": {} })).expect_err("no content");
        assert!(error.to_string().contains("no content"));
    }
}
