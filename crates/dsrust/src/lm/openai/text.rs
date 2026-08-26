//! dspy `model_type="text"`: the legacy completions wire, which takes one prompt string.
//!
//! dspy has two paths to it. `litellm_text_completion` hands the work to litellm and re-prefixes
//! the model `text-completion-openai/` for that router; `to_openai_text_request` builds the body
//! itself from a typed request, which is the path this crate follows and the one that sends the
//! bare model id. They agree on the prompt.
//!
//! Both halves are dspy's own and derivable from nothing: the prompt is each message's text parts
//! concatenated with no separator, the messages joined by blank lines with `BEGIN RESPONSE:`
//! appended to the *list*, and the config is `text_config_kwargs` — extensions first, then
//! temperature, max_tokens, top_p, stop, logprobs, n, in that order. Held byte for byte by
//! `tests/openai_text_conformance.rs`.

use anyhow::Result;
use serde_json::{Value, json};

use crate::lm::api;

/// What upstream appends to every prompt, so a completion model knows where its turn starts.
const BEGIN: &str = "BEGIN RESPONSE:";

/// dspy `messages_to_text_prompt`: each message flattened, then `BEGIN RESPONSE:`, joined by blank
/// lines.
///
/// Every message contributes, whatever its role — nothing marks which paragraph was the system
/// prompt, which is why this wire is worth having only for a model that has no chat endpoint. A
/// message's own parts concatenate with *no* separator, so two text parts run together where two
/// messages do not. An empty conversation still yields `BEGIN RESPONSE:`, since the marker joins
/// the list rather than a non-empty one.
///
/// Refuses a part that is not text, as upstream does: this endpoint carries no images, and a
/// silently dropped one is a prompt the caller did not write.
pub fn prompt(messages: &[api::LmMessage]) -> Result<String> {
    let mut chunks = Vec::with_capacity(messages.len() + 1);
    for message in messages {
        let mut chunk = String::new();
        for part in &message.parts {
            let Some(text) = part.as_text() else {
                return Err(anyhow::anyhow!(
                    "OpenAI text completions only support text parts, but received {}.",
                    part.kind()
                ));
            };
            chunk.push_str(text);
        }
        chunks.push(chunk);
    }
    chunks.push(BEGIN.to_owned());
    Ok(chunks.join("\n\n"))
}

/// The completions body: the prompt, and the sampling fields this wire takes.
///
/// A narrower set than chat's, because the endpoint is narrower — no tools, no response format, no
/// reasoning. What a caller set that this wire cannot carry is left out rather than sent and
/// rejected, which is what dspy's `**request` pass-through would do with it.
pub(super) fn request(model: &str, call: &api::LmRequest) -> Result<Value> {
    let mut request = json!({ "model": model, "prompt": prompt(&call.messages)? });
    let config = &call.config;
    // `dict(config.extensions)` opens `text_config_kwargs`, so an unknown keyword is written before
    // every known one and a known one written twice keeps the later value.
    for (key, value) in &config.extensions {
        request[key] = value.clone();
    }
    // Upstream's own order — temperature, max_tokens, top_p — which is not the chat wire's.
    if let Some(temperature) = config.temperature {
        request["temperature"] = json!(temperature);
    }
    if let Some(max_tokens) = config.max_tokens {
        // Always `max_tokens`: the completions endpoint never took `max_completion_tokens`, so the
        // reasoning-model rule that picks between them does not apply here.
        request["max_tokens"] = json!(max_tokens);
    }
    if let Some(top_p) = config.top_p {
        request["top_p"] = json!(top_p);
    }
    if let Some(stop) = &config.stop
        && !stop.is_empty()
    {
        request["stop"] = json!(stop);
    }
    if let Some(logprobs) = &config.logprobs {
        request["logprobs"] = serde_json::to_value(logprobs).unwrap_or(Value::Null);
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

    /// Our completions body equals the one dspy 3.3's own `to_openai_text_request` builds from the
    /// same typed request — the prompt rule and the config mapping both, key order included.
    ///
    /// The fixture is generated by `scripts/generate_openai_text_fixture.py` running dspy, so a
    /// divergence is dspy's word against ours. Built first against `litellm_text_completion`,
    /// which agrees on the prompt and disagrees on the model, and got the config order wrong
    /// (`temperature, top_p, max_tokens`) and dropped `logprobs` — neither visible from that path,
    /// which forwards the config untouched.
    #[test]
    fn our_body_matches_dspy_33_to_openai_text_request() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/conformance/lm_api/openai_text.json");
        let fixture: Value =
            serde_json::from_str(&std::fs::read_to_string(path).expect("fixture is readable"))
                .expect("fixture is valid json");

        for case in fixture["cases"].as_array().expect("cases array") {
            let name = case["name"].as_str().expect("a case name");
            let call: api::LmRequest = serde_json::from_value(case["lm_request"].clone())
                .unwrap_or_else(|error| panic!("{name}: the typed request did not parse: {error}"));
            let body = request(&call.model, &call).expect("the body builds");
            assert_eq!(
                body, case["expected"],
                "{name}: our completions body diverges from dspy 3.3's to_openai_text_request"
            );
        }
    }

    /// This endpoint carries text and nothing else, and upstream refuses a part that is not — in
    /// those words, naming the Python class, since a caller matching the sentence is matching
    /// dspy's.
    #[test]
    fn a_part_that_is_not_text_is_refused_in_dspys_words() {
        let call = api::LmRequest::new(
            "gpt-3.5-turbo-instruct",
            vec![api::LmMessage::user([api::LmPart::image_url(
                "https://example.com/cat.png",
            )])],
        );
        let refused = request("gpt-3.5-turbo-instruct", &call).expect_err("an image is refused");
        assert_eq!(
            refused.to_string(),
            "OpenAI text completions only support text parts, but received LMImagePart."
        );
    }

    fn user(text: &str) -> api::LmMessage {
        api::LmMessage::user([api::LmPart::text(text)])
    }

    /// The marker is appended to the list, not joined onto it, so an empty conversation still
    /// carries it — which is what upstream's `[...] + ["BEGIN RESPONSE:"]` does.
    #[test]
    fn an_empty_conversation_is_the_marker_alone() {
        assert_eq!(prompt(&[]).expect("no parts to refuse"), "BEGIN RESPONSE:");
    }

    /// Content already holding a blank line is joined as it stands: upstream neither escapes nor
    /// re-wraps, so the paragraphs a model sees can run together and that is dspy's behaviour.
    #[test]
    fn a_blank_line_inside_content_is_left_alone() {
        assert_eq!(
            prompt(&[user("first\n\nsecond")]).expect("text parts"),
            "first\n\nsecond\n\nBEGIN RESPONSE:"
        );
    }
}
