//! A local ollama server.
//!
//! It takes the OpenAI-shaped message list but keeps config under an `options` object, and
//! names the generation cap `num_predict` rather than `max_tokens`.

use std::future::Future;

use anyhow::{Context, Result, anyhow};
use serde_json::{Value, json};

use super::{ChatModel, LmUsage, PROVIDER_TIMEOUT, api};

/// ollama samples at 0.8 when told nothing, which is loose for a program parsing the reply
/// back into fields.
const TEMPERATURE: f64 = 0.7;

/// A local ollama server as a [`ChatModel`], the model and host it needs held beside it.
pub(crate) struct Ollama<'a> {
    pub model: &'a str,
    pub host: &'a str,
}

impl ChatModel for Ollama<'_> {
    fn forward<'a>(
        &'a self,
        http: &'a reqwest::Client,
        call: &'a api::LmRequest,
    ) -> impl Future<Output = Result<api::LmResponse>> + Send + 'a {
        async move {
            let response = http
                .post(format!("{}/api/chat", self.host))
                .timeout(PROVIDER_TIMEOUT)
                .json(&request(self.model, call))
                .send()
                .await
                .context("ollama request failed")?;
            if !response.status().is_success() {
                return Err(anyhow!("ollama {}", response.status()));
            }
            let body: Value = response
                .json()
                .await
                .context("ollama response was not JSON")?;
            reply(self.model, &body)
        }
    }
}

fn reply(model: &str, body: &Value) -> Result<api::LmResponse> {
    let text = body["message"]["content"]
        .as_str()
        .context("ollama returned no content")?;
    Ok(api::LmResponse::completions([text.to_owned()])
        .with_usage(usage(body))
        .with_provider_response(provider_data(body))
        .with_model(model))
}

/// ollama counts at the top level rather than under a usage object, and names the two counts
/// after the passes that produce them.
fn usage(body: &Value) -> Option<LmUsage> {
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
fn provider_data(body: &Value) -> Option<Value> {
    let done_reason = body["done_reason"].as_str()?;
    Some(json!({ "done_reason": done_reason }))
}

fn request(model: &str, call: &api::LmRequest) -> Value {
    let temperature = call.config.temperature.unwrap_or(TEMPERATURE);
    let mut request = json!({
        "model": model,
        "stream": false,
        "options": { "temperature": temperature },
        "messages": call.wire_messages(),
    });
    if let Some(max_tokens) = call.config.max_tokens {
        request["options"]["num_predict"] = json!(max_tokens);
    }
    if call.output_schema().is_some() {
        request["format"] = json!("json");
    }
    request
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lm::api::interop::raise_request;
    use crate::lm::{LmConfig, OutputMode};

    fn sampled(config: LmConfig) -> api::LmRequest {
        raise_request("be helpful", &[], OutputMode::Text, &config)
    }

    /// ollama's own default samples loosely, so an unnamed temperature keeps the tighter one
    /// this crate has always sent rather than falling through to the server's.
    #[test]
    fn sampling_travels_under_options_with_the_cap_renamed() {
        let default = request("qwen2.5:7b-instruct", &sampled(LmConfig::default()));
        assert_eq!(default["options"]["temperature"], TEMPERATURE);
        assert_eq!(default["options"].get("num_predict"), None);

        let named = request(
            "qwen2.5:7b-instruct",
            &sampled(LmConfig {
                temperature: Some(0.1),
                max_tokens: Some(64),
                ..LmConfig::default()
            }),
        );
        assert_eq!(named["options"]["temperature"], 0.1);
        assert_eq!(named["options"]["num_predict"], 64);
    }

    /// A cap belongs inside `options` alongside the temperature; sent at the top level ollama
    /// ignores it silently.
    #[test]
    fn the_cap_never_lands_at_the_top_level() {
        let body = request(
            "qwen2.5:7b-instruct",
            &sampled(LmConfig {
                max_tokens: Some(64),
                ..LmConfig::default()
            }),
        );
        assert_eq!(body.get("num_predict"), None);
        assert_eq!(body.get("max_tokens"), None);
    }

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
            answered
                .provider_response
                .expect("a done reason")["done_reason"],
            "length"
        );
    }

    #[test]
    fn a_reply_reporting_no_counts_reports_no_usage() {
        let body = json!({ "message": { "content": "the reply" } });
        assert_eq!(
            reply("qwen2.5:7b-instruct", &body).expect("a reply").usage,
            None
        );
    }

    #[test]
    fn a_reply_carrying_no_content_is_an_error() {
        let error =
            reply("qwen2.5:7b-instruct", &json!({ "message": {} })).expect_err("no content");
        assert!(error.to_string().contains("no content"));
    }
}
