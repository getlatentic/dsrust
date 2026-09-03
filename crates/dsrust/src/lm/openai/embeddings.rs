//! The OpenAI-compatible `/embeddings` call litellm makes for `dspy.Embedder("openai/...")`.
//!
//! The body is what litellm sends — `model`, `input`, and whatever keyword arguments the caller
//! added — held to `tests/conformance/lm_api/embedding_wire.json`, recorded from litellm at the
//! HTTP layer. The reply's `data` carries one `embedding` per input, in input order.

use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use serde_json::{Map, Value};

use super::OpenAiConfig;

/// The request body litellm posts: `input`, then `model`, then the caller's keyword arguments,
/// then the `encoding_format` of `float` the OpenAI SDK adds unless the caller chose one.
pub fn request_body(model_id: &str, inputs: &[String], kwargs: &Map<String, Value>) -> Value {
    let mut body = Map::new();
    body.insert(
        "input".to_owned(),
        Value::Array(
            inputs
                .iter()
                .map(|text| Value::String(text.clone()))
                .collect(),
        ),
    );
    body.insert("model".to_owned(), Value::String(model_id.to_owned()));
    for (key, value) in kwargs {
        body.insert(key.clone(), value.clone());
    }
    body.entry("encoding_format".to_owned())
        .or_insert_with(|| Value::String("float".to_owned()));
    Value::Object(body)
}

/// One embedding per input, read off the reply's `data`.
pub fn embeddings_of(reply: &Value) -> Result<Vec<Vec<f32>>> {
    let data = reply
        .get("data")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("the embeddings reply carries no `data` list: {reply}"))?;
    data.iter()
        .map(|entry| {
            let embedding = entry
                .get("embedding")
                .and_then(Value::as_array)
                .ok_or_else(|| anyhow!("an embeddings entry carries no `embedding`: {entry}"))?;
            embedding
                .iter()
                .map(|value| {
                    value
                        .as_f64()
                        .map(|float| float as f32)
                        .ok_or_else(|| anyhow!("an embedding value is not a number: {value}"))
                })
                .collect()
        })
        .collect()
}

/// `POST {base_url}/embeddings`, bearer-authenticated as the chat call is.
pub(crate) async fn embed(
    http: &reqwest::Client,
    config: &OpenAiConfig,
    model_id: &str,
    inputs: &[String],
    kwargs: &Map<String, Value>,
    timeout: Duration,
) -> Result<Vec<Vec<f32>>> {
    let url = format!("{}/embeddings", config.base_url.trim_end_matches('/'));
    let mut request = http
        .post(&url)
        .timeout(timeout)
        .json(&request_body(model_id, inputs, kwargs));
    if let Some(key) = &config.api_key {
        request = request.bearer_auth(key);
    }
    let response = request
        .send()
        .await
        .with_context(|| format!("embeddings request to {url} failed"))?;
    let status = response.status();
    let text = response
        .text()
        .await
        .context("reading the embeddings reply")?;
    if !status.is_success() {
        return Err(anyhow!(
            "embeddings request to {url} answered {status}: {text}"
        ));
    }
    let reply: Value = serde_json::from_str(&text).context("the embeddings reply is not JSON")?;
    embeddings_of(&reply)
}
