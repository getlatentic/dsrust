//! A local ollama server.
//!
//! It takes the OpenAI-shaped message list but keeps sampling under an `options` object, and
//! names the generation cap `num_predict` rather than `max_tokens`.

use anyhow::{Context, Result, anyhow};
use serde_json::{Value, json};

use super::{LmRequest, OutputMode, PROVIDER_TIMEOUT, wire_messages};

/// ollama samples at 0.8 when told nothing, which is loose for a program parsing the reply
/// back into fields.
const TEMPERATURE: f64 = 0.7;

pub(super) async fn chat(
    http: &reqwest::Client,
    model: &str,
    host: &str,
    call: &LmRequest<'_>,
) -> Result<String> {
    let response = http
        .post(format!("{host}/api/chat"))
        .timeout(PROVIDER_TIMEOUT)
        .json(&request(model, call))
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
    body["message"]["content"]
        .as_str()
        .map(str::to_owned)
        .context("ollama returned no content")
}

fn request(model: &str, call: &LmRequest<'_>) -> Value {
    let temperature = call.sampling.temperature.unwrap_or(TEMPERATURE);
    let mut request = json!({
        "model": model,
        "stream": false,
        "options": { "temperature": temperature },
        "messages": wire_messages(call.system, call.turns),
    });
    if let Some(max_tokens) = call.sampling.max_tokens {
        request["options"]["num_predict"] = json!(max_tokens);
    }
    if matches!(call.mode, OutputMode::Json { .. }) {
        request["format"] = json!("json");
    }
    request
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lm::Sampling;

    fn sampled(sampling: Sampling) -> LmRequest<'static> {
        LmRequest::new("be helpful", &[], OutputMode::Text).sampled(sampling)
    }

    /// ollama's own default samples loosely, so an unnamed temperature keeps the tighter one
    /// this crate has always sent rather than falling through to the server's.
    #[test]
    fn sampling_travels_under_options_with_the_cap_renamed() {
        let default = request("qwen2.5:7b-instruct", &sampled(Sampling::default()));
        assert_eq!(default["options"]["temperature"], TEMPERATURE);
        assert_eq!(default["options"].get("num_predict"), None);

        let named = request(
            "qwen2.5:7b-instruct",
            &sampled(Sampling {
                temperature: Some(0.1),
                max_tokens: Some(64),
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
            &sampled(Sampling {
                max_tokens: Some(64),
                ..Sampling::default()
            }),
        );
        assert_eq!(body.get("num_predict"), None);
        assert_eq!(body.get("max_tokens"), None);
    }
}
