//! OpenAI streaming: Server-Sent Events read back as the typed [`LmStreamEvent`]s.
//!
//! There is no dspy reference for this mapping — dspy 3.3 streams through litellm's
//! `ModelResponseStream` — so it is faithful to OpenAI's SSE wire (one `data:` frame per chunk,
//! `[DONE]` to close) and the typed streaming vocabulary rather than to a shipped implementation.
//!
//! Production is buffered for now: the whole body is read before its events are handed out. The
//! incremental read is a later refinement, and no stub test can tell the two apart, since a stub
//! sends the body whole.

use anyhow::{Context, Result, anyhow};
use futures_util::{Stream, StreamExt, stream};
use serde_json::Value;

use crate::lm::PROVIDER_TIMEOUT;
use crate::lm::api::{LmDelta, LmStreamEvent};

/// One streaming chunk as the events it implies: a content delta per choice, and an
/// [`LmStreamEvent::OutputEnd`] where a choice named why it stopped.
fn events_from_chunk(chunk: &Value) -> Vec<LmStreamEvent> {
    let mut events = Vec::new();
    for choice in chunk["choices"].as_array().into_iter().flatten() {
        let index = choice["index"].as_u64().unwrap_or(0) as usize;
        if let Some(content) = choice["delta"]["content"].as_str()
            && !content.is_empty()
        {
            events.push(LmStreamEvent::Delta {
                output_index: index,
                part_index: 0,
                delta: LmDelta::text(content),
            });
        }
        if let Some(reason) = choice["finish_reason"].as_str() {
            events.push(LmStreamEvent::OutputEnd {
                output_index: index,
                finish_reason: Some(reason.to_owned()),
                truncated: reason == "length",
            });
        }
    }
    events
}

/// A whole SSE body as the events it carries: [`Start`](LmStreamEvent::Start) first, each chunk's
/// deltas and output-ends in order, then [`End`](LmStreamEvent::End) with whatever usage the final
/// chunk reported — OpenAI sends it when the request asked for `stream_options.include_usage`.
pub(super) fn parse_sse(body: &str, model: &str) -> Vec<Result<LmStreamEvent>> {
    let mut events = vec![Ok(LmStreamEvent::Start {
        model: Some(model.to_owned()),
    })];
    let mut usage = None;
    for frame in body.split("\n\n") {
        let Some(data) = frame.trim().strip_prefix("data:") else {
            continue; // comments, keep-alives, the trailing blank
        };
        let data = data.trim();
        if data == "[DONE]" {
            break;
        }
        match serde_json::from_str::<Value>(data) {
            Ok(chunk) => {
                if let Some(reported) = super::usage(&chunk["usage"]) {
                    usage = Some(reported);
                }
                events.extend(events_from_chunk(&chunk).into_iter().map(Ok));
            }
            Err(error) => {
                events.push(Err(anyhow!("openai stream chunk was not JSON: {error}")));
            }
        }
    }
    events.push(Ok(LmStreamEvent::End {
        usage,
        cost: None,
        response: None,
    }));
    events
}

/// The events of one streaming call. Borrows only the client — the owned url, key, model and body
/// are lifted out of the endpoint before the stream is built, so it outlives the temporary that
/// built it.
pub(super) fn events<'h>(
    http: &'h reqwest::Client,
    url: String,
    key: Option<String>,
    label: String,
    model: String,
    body: Value,
) -> impl Stream<Item = Result<LmStreamEvent>> + Send + 'h {
    stream::once(fetch(http, url, key, label, model, body)).flat_map(stream::iter)
}

async fn fetch(
    http: &reqwest::Client,
    url: String,
    key: Option<String>,
    label: String,
    model: String,
    body: Value,
) -> Vec<Result<LmStreamEvent>> {
    match send(http, url, key, label, &body).await {
        Ok(text) => parse_sse(&text, &model),
        Err(error) => vec![Err(error)],
    }
}

async fn send(
    http: &reqwest::Client,
    url: String,
    key: Option<String>,
    label: String,
    body: &Value,
) -> Result<String> {
    let key = key.ok_or_else(|| anyhow!("{label} streaming needs an API key"))?;
    let response = http
        .post(url)
        .bearer_auth(key)
        .timeout(PROVIDER_TIMEOUT)
        .json(body)
        .send()
        .await
        .with_context(|| format!("{label} streaming request failed"))?;
    let status = response.status();
    let text = response
        .text()
        .await
        .with_context(|| format!("{label} streaming response was unreadable"))?;
    if !status.is_success() {
        return Err(anyhow!("{label} {status}: {text}"));
    }
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A representative OpenAI stream: a role-only opener, two content deltas, a stop, then the
    /// usage chunk and `[DONE]`.
    const SSE: &str = "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\n\
        data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Par\"}}]}\n\n\
        data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"is\"}}]}\n\n\
        data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
        data: {\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":4,\"total_tokens\":14}}\n\n\
        data: [DONE]\n\n";

    #[test]
    fn a_body_of_frames_parses_into_start_deltas_end() {
        let events: Vec<LmStreamEvent> = parse_sse(SSE, "openai/gpt-4o")
            .into_iter()
            .map(|event| event.expect("every frame parsed"))
            .collect();

        assert!(matches!(&events[0], LmStreamEvent::Start { model } if model.as_deref() == Some("openai/gpt-4o")));
        // The role-only opener carries no content, so it contributes no delta.
        assert_eq!(
            events[1],
            LmStreamEvent::delta(0, LmDelta::text("Par")),
            "first content delta"
        );
        assert_eq!(events[2], LmStreamEvent::delta(0, LmDelta::text("is")));
        assert!(matches!(
            &events[3],
            LmStreamEvent::OutputEnd { finish_reason, truncated, .. }
                if finish_reason.as_deref() == Some("stop") && !truncated
        ));
        let LmStreamEvent::End { usage, .. } = events.last().expect("an end") else {
            panic!("the last event is the end, got {:?}", events.last());
        };
        assert_eq!(usage.as_ref().and_then(|u| u.total()), Some(14));
    }

    /// The whole point of the typed vocabulary: the events reassemble into the reply, via the same
    /// [`LmOutputBuilder`](crate::lm::api::LmOutputBuilder) a live stream would drive.
    #[test]
    fn the_events_reassemble_into_the_reply() {
        use crate::lm::api::LmOutputBuilder;
        let mut builder = LmOutputBuilder::new();
        let mut response = None;
        for event in parse_sse(SSE, "openai/gpt-4o") {
            if let Some(assembled) = builder.apply(event.expect("valid event")).expect("applies") {
                response = Some(assembled);
            }
        }
        let response = response.expect("the end event produced a reply");
        assert_eq!(response.first_text(), "Paris");
        assert_eq!(response.usage.and_then(|u| u.total()), Some(14));
    }

    #[test]
    fn a_finish_reason_of_length_marks_the_output_truncated() {
        let body = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"x\"},\"finish_reason\":\"length\"}]}\n\ndata: [DONE]\n\n";
        let events = parse_sse(body, "m");
        assert!(events.iter().any(|event| matches!(
            event.as_ref().ok(),
            Some(LmStreamEvent::OutputEnd { truncated: true, .. })
        )));
    }
}
