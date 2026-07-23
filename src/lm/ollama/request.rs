//! The ollama `/api/chat` body, built the way litellm builds it.
//!
//! dspy 3.3 reaches a local ollama through litellm, so litellm's transform is the reference here.
//! Each message keeps a bare-string `content` beside an `images` list rather than the block list the
//! other providers read; sampling travels under `options` with nothing defaulted — litellm sends no
//! temperature the caller did not name — and the cap is renamed `num_predict`; a requested schema
//! rides as `format`; tools keep the OpenAI shape. `tests/lm_api_conformance.rs` holds the body to
//! litellm's own, case for case.

use serde_json::{Value, json};

use crate::lm::api::{self, Content, LmConfig, LmToolSpec, content_of};

/// The ollama chat body for one call.
pub(super) fn request(model: &str, call: &api::LmRequest) -> Value {
    let mut body = json!({
        "model": model,
        "messages": messages(call),
        "options": options(&call.config),
        "stream": false,
    });
    if let Some(schema) = call.output_schema() {
        body["format"] = schema.clone();
    }
    if !call.tools.is_empty() {
        body["tools"] = Value::Array(call.tools.iter().map(tool).collect());
    }
    body
}

/// Every message as `{role, content, images}`, the system message kept in place rather than lifted
/// out the way Anthropic lifts it.
fn messages(call: &api::LmRequest) -> Vec<Value> {
    call.messages
        .iter()
        .map(|message| {
            let (content, images) = split(&message.parts);
            json!({ "role": message.role, "content": content, "images": images })
        })
        .collect()
}

/// A message's parts split into ollama's two fields: text concatenated into `content`, images pulled
/// out as base64 into `images`. The parts render first to the OpenAI-shaped blocks [`content_of`]
/// builds — litellm's own input — so a data URI becomes the bare base64 ollama wants.
fn split(parts: &[api::LmPart]) -> (String, Vec<String>) {
    match content_of(parts) {
        Ok(Content::Text(text)) => (text, Vec::new()),
        Ok(Content::Blocks(blocks)) => {
            let mut text = String::new();
            let mut images = Vec::new();
            for block in &blocks {
                match block["type"].as_str() {
                    Some("text") => text.push_str(block["text"].as_str().unwrap_or_default()),
                    Some("image_url") => images.push(image_data(&block["image_url"])),
                    _ => {}
                }
            }
            (text, images)
        }
        Err(_) => (String::new(), Vec::new()),
    }
}

/// The base64 ollama takes: the payload of a `data:` URI, or the reference itself when it is not one.
fn image_data(image_url: &Value) -> String {
    let url = image_url["url"].as_str().unwrap_or_default();
    url.split_once(";base64,")
        .map_or(url, |(_, data)| data)
        .to_owned()
}

/// Sampling under `options`, only the keys the caller set. litellm defaults none, so an unnamed
/// temperature is left to ollama's own rather than one this crate invents; the cap is `num_predict`.
fn options(config: &LmConfig) -> Value {
    let mut options = serde_json::Map::new();
    if let Some(temperature) = config.temperature {
        options.insert("temperature".to_owned(), json!(temperature));
    }
    if let Some(stop) = &config.stop
        && !stop.is_empty()
    {
        options.insert("stop".to_owned(), json!(stop));
    }
    if let Some(max_tokens) = config.max_tokens {
        options.insert("num_predict".to_owned(), json!(max_tokens));
    }
    Value::Object(options)
}

/// litellm keeps the OpenAI tool shape for ollama: a function wrapper with the schema under
/// `parameters`, the description kept when the caller gave one.
fn tool(spec: &LmToolSpec) -> Value {
    let mut function = json!({ "name": spec.name, "parameters": spec.parameters });
    if let Some(description) = &spec.description {
        function["description"] = json!(description);
    }
    json!({ "type": "function", "function": function })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Faithfulness to dspy 3.3's actual ollama path: our body equals the one litellm puts on the
    /// wire for the same typed request — bare content beside `images`, `options` with nothing
    /// defaulted, `num_predict` for the cap, `format` for a schema, OpenAI-shaped tools. Every
    /// expectation is litellm's captured output, so a divergence is litellm's word against ours.
    #[test]
    fn our_body_matches_litellm_for_ollama() {
        crate::lm::tests::each_case("ollama", request);
    }
}
