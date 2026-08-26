//! dspy `model_type="text"`: the legacy completions wire, which takes one prompt string.
//!
//! Upstream reaches it through `litellm_text_completion`, whose own contribution is two rules and
//! nothing else — the prompt is every message's content joined by a blank line with
//! `BEGIN RESPONSE:` appended, and the model is re-prefixed `text-completion-openai/`. That prefix
//! is litellm's *routing* token, not part of the wire, so it does not travel here: this crate posts
//! the bare model id to `/completions` the way the provider documents.
//!
//! The prompt is the part that reaches a model, and it is held byte for byte against dspy by
//! `tests/text_completion_conformance.rs` over a fixture recorded from the pinned dspy.

use anyhow::Result;
use serde_json::{Value, json};

use crate::lm::api;

/// What upstream appends to every prompt, so a completion model knows where its turn starts.
const BEGIN: &str = "BEGIN RESPONSE:";

/// dspy's prompt: each message's content, then `BEGIN RESPONSE:`, joined by blank lines.
///
/// Every message contributes, whatever its role — upstream reads `x["content"]` and nothing else,
/// so a system prompt and an assistant demo turn arrive as bare paragraphs with no marker saying
/// which was which. An empty message list still yields `BEGIN RESPONSE:`, since the marker is
/// appended to the list rather than joined onto a non-empty one.
pub fn prompt(messages: &[api::LmMessage]) -> String {
    messages
        .iter()
        .map(|message| message.text().unwrap_or_default())
        .chain(std::iter::once(BEGIN.to_owned()))
        .collect::<Vec<String>>()
        .join("\n\n")
}

/// The completions body: the prompt, and the sampling fields this wire takes.
///
/// A narrower set than chat's, because the endpoint is narrower — no tools, no response format, no
/// reasoning. What a caller set that this wire cannot carry is left out rather than sent and
/// rejected, which is what dspy's `**request` pass-through would do with it.
pub(super) fn request(model: &str, call: &api::LmRequest) -> Result<Value> {
    let mut request = json!({ "model": model, "prompt": prompt(&call.messages) });
    let config = &call.config;
    for (key, value) in &config.extensions {
        request[key] = value.clone();
    }
    if let Some(temperature) = config.temperature {
        request["temperature"] = json!(temperature);
    }
    if let Some(top_p) = config.top_p {
        request["top_p"] = json!(top_p);
    }
    if let Some(max_tokens) = config.max_tokens {
        // Always `max_tokens` here: the completions endpoint never took the chat wire's
        // `max_completion_tokens`, so the reasoning-model rule that picks between them does not
        // apply to it.
        request["max_tokens"] = json!(max_tokens);
    }
    if let Some(stop) = &config.stop
        && !stop.is_empty()
    {
        request["stop"] = json!(stop);
    }
    if let Some(n) = config.n {
        request["n"] = json!(n);
    }
    Ok(request)
}

/// One completion choice as a reply part: `choices[].text`, where chat has `message.content`.
pub(super) fn reply(
    label: &str,
    model: &str,
    status: reqwest::StatusCode,
    headers: &reqwest::header::HeaderMap,
    body: &Value,
) -> Result<api::LmResponse> {
    // The failure path is the chat wire's: the endpoint differs, the error envelope does not.
    if !status.is_success() {
        if let Some(too_long) = crate::lm::ContextWindowExceeded::detected(model, body) {
            return Err(too_long.into());
        }
        return Err(
            crate::lm::LmFailure::from_body(status.as_u16(), model, label, body)
                .headers(headers)
                .into(),
        );
    }
    let mut answer = api::LmResponse::default();
    for choice in body["choices"].as_array().into_iter().flatten() {
        let text = choice["text"].as_str().unwrap_or_default();
        let mut output = api::LmOutput::default();
        output.parts.push(api::LmPart::text(text));
        output.finish_reason = choice["finish_reason"].as_str().map(str::to_owned);
        answer.outputs.push(output);
    }
    if answer.outputs.is_empty() {
        return Err(anyhow::anyhow!("{label} returned no content"));
    }
    answer.usage = super::response::usage(&body["usage"]);
    answer.response_id = body["id"].as_str().map(str::to_owned);
    answer.model = Some(body["model"].as_str().unwrap_or(model).to_owned());
    Ok(answer)
}

/// The completions stream's body: the request with `stream` set.
pub(super) fn streaming_body(model: &str, call: &api::LmRequest) -> Result<Value> {
    let mut body = request(model, call)?;
    body["stream"] = json!(true);
    Ok(body)
}

/// One SSE frame of a completions stream.
///
/// Simpler than the chat wire's by the whole of what this endpoint cannot do: a chunk carries
/// `choices[].text` and nothing else — no reasoning, no tool calls — so every delta is text at
/// part 0 of its choice.
fn frame(
    frame: &str,
    _state: &mut crate::lm::streaming::StreamState,
) -> crate::lm::streaming::Framed {
    use crate::lm::streaming::Framed;
    let Some(data) = frame.trim().strip_prefix("data:") else {
        return Framed::of(Vec::new());
    };
    let data = data.trim();
    if data == "[DONE]" {
        return Framed::closing(Vec::new());
    }
    let Ok(chunk) = serde_json::from_str::<Value>(data) else {
        return Framed::of(Vec::new());
    };
    let mut events = Vec::new();
    for choice in chunk["choices"].as_array().into_iter().flatten() {
        let index = choice["index"].as_u64().unwrap_or(0) as usize;
        if let Some(text) = choice["text"].as_str()
            && !text.is_empty()
        {
            events.push(api::LmStreamEvent::Delta {
                output_index: index,
                part_index: 0,
                delta: api::LmDelta::text(text),
            });
        }
    }
    Framed::of(events)
}

/// The typed events of one streaming completions call.
pub(super) fn stream(
    http: &reqwest::Client,
    url: String,
    key: Option<String>,
    label: String,
    model: String,
    body: Value,
    timeout: std::time::Duration,
) -> impl futures_util::Stream<Item = Result<api::LmStreamEvent>> + Send + 'static {
    let mut request = http.post(url).timeout(timeout).json(&body);
    if let Some(key) = key {
        request = request.bearer_auth(key);
    }
    crate::lm::streaming::events(
        request.send(),
        label,
        model,
        crate::lm::streaming::Framing {
            separator: b"\n\n",
            frame,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user(text: &str) -> api::LmMessage {
        api::LmMessage::user([api::LmPart::text(text)])
    }

    /// The marker is appended to the list, not joined onto it, so an empty conversation still
    /// carries it — which is what upstream's `[...] + ["BEGIN RESPONSE:"]` does.
    #[test]
    fn an_empty_conversation_is_the_marker_alone() {
        assert_eq!(prompt(&[]), "BEGIN RESPONSE:");
    }

    /// Content already holding a blank line is joined as it stands: upstream neither escapes nor
    /// re-wraps, so the paragraphs a model sees can run together and that is dspy's behaviour.
    #[test]
    fn a_blank_line_inside_content_is_left_alone() {
        assert_eq!(
            prompt(&[user("first\n\nsecond")]),
            "first\n\nsecond\n\nBEGIN RESPONSE:"
        );
    }
}
