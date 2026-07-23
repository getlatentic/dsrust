//! OpenAI streaming: Server-Sent Events read back as the typed [`LmStreamEvent`]s.
//!
//! There is no dspy reference for this mapping — dspy 3.3 streams through litellm's
//! `ModelResponseStream` — so it is faithful to OpenAI's SSE wire (one `data:` frame per chunk,
//! `[DONE]` to close) and the typed streaming vocabulary rather than to a shipped implementation.
//! The reading, framing and assembly are shared; this is only the OpenAI frame's meaning.

use anyhow::Result;
use futures_util::Stream;
use serde_json::Value;

use crate::lm::PROVIDER_TIMEOUT;
use crate::lm::api::{LmDelta, LmStreamEvent};
use crate::lm::streaming::{Framed, Framing, StreamState};

/// One streaming chunk as the events it implies: a reasoning or content delta per choice, tool-call
/// fragments, and an [`LmStreamEvent::OutputEnd`] where a choice named why it stopped.
fn events_from_chunk(chunk: &Value, state: &mut StreamState) -> Vec<LmStreamEvent> {
    let mut events = Vec::new();
    for choice in chunk["choices"].as_array().into_iter().flatten() {
        let index = choice["index"].as_u64().unwrap_or(0) as usize;
        // A reasoning model streams its reasoning ahead of the answer; it takes part 0 and pushes the
        // content that follows to the next slot, the thinking-then-text order the reply is read in.
        if let Some(reasoning) = choice["delta"]["reasoning_content"].as_str()
            && !reasoning.is_empty()
        {
            state.content_offset = 1;
            events.push(LmStreamEvent::Delta {
                output_index: index,
                part_index: 0,
                delta: LmDelta::thinking(reasoning),
            });
        }
        if let Some(content) = choice["delta"]["content"].as_str()
            && !content.is_empty()
        {
            events.push(LmStreamEvent::Delta {
                output_index: index,
                part_index: state.content_offset,
                delta: LmDelta::text(content),
            });
        }
        // A tool call arrives in fragments across chunks — its id and name once, its arguments a
        // slice at a time — under an index of its own. `LmOutputBuilder` reassembles them.
        for call in choice["delta"]["tool_calls"]
            .as_array()
            .into_iter()
            .flatten()
        {
            let slot = call["index"].as_u64().unwrap_or(0) as usize;
            events.push(LmStreamEvent::Delta {
                output_index: index,
                part_index: state.content_offset + slot,
                delta: LmDelta::ToolCallDelta {
                    id: call["id"].as_str().map(str::to_owned),
                    name: call["function"]["name"].as_str().map(str::to_owned),
                    args_delta: call["function"]["arguments"].as_str().map(str::to_owned),
                },
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

/// One SSE frame — a `data:` line, `data: [DONE]`, or a keep-alive — as its events. The usage the
/// final chunk carries is accumulated for the stream's [`End`](LmStreamEvent::End); OpenAI sends it
/// when the request asked for `stream_options.include_usage`.
fn frame(frame: &str, state: &mut StreamState) -> Framed {
    let Some(data) = frame.trim().strip_prefix("data:") else {
        return Framed::of(Vec::new()); // a comment or keep-alive carries nothing
    };
    let data = data.trim();
    if data == "[DONE]" {
        return Framed::closing(Vec::new());
    }
    let Ok(chunk) = serde_json::from_str::<Value>(data) else {
        // A chunk that will not parse is skipped rather than failing the stream mid-flight, which
        // is how a keep-alive or a partial line is tolerated.
        return Framed::of(Vec::new());
    };
    if let Some(reported) = super::response::usage(&chunk["usage"]) {
        state.usage = Some(reported);
    }
    Framed::of(events_from_chunk(&chunk, state))
}

/// The typed events of one streaming call. The owned url, key, model and body are lifted out of
/// the endpoint before the stream is built, so it borrows only the client.
pub(super) fn events<'h>(
    http: &'h reqwest::Client,
    url: String,
    key: Option<String>,
    label: String,
    model: String,
    body: Value,
) -> impl Stream<Item = Result<LmStreamEvent>> + Send + 'h {
    let mut request = http.post(url).timeout(PROVIDER_TIMEOUT).json(&body);
    if let Some(key) = key {
        request = request.bearer_auth(key);
    }
    crate::lm::streaming::events(
        request.send(),
        label,
        model,
        Framing {
            separator: b"\n\n",
            frame,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn events_of(body: &str) -> Vec<LmStreamEvent> {
        let mut state = StreamState::default();
        let mut events = Vec::new();
        for line in body.split("\n\n") {
            let framed = frame(line, &mut state);
            events.extend(framed.events);
            if framed.done {
                events.push(LmStreamEvent::End {
                    usage: state.usage.take(),
                    cost: None,
                    response: None,
                });
                break;
            }
        }
        events
    }

    const SSE: &str = "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\n\
        data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Par\"}}]}\n\n\
        data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"is\"}}]}\n\n\
        data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
        data: {\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":4,\"total_tokens\":14}}\n\n\
        data: [DONE]\n\n";

    #[test]
    fn a_role_only_opener_carries_no_delta_and_content_deltas_do() {
        let events = events_of(SSE);
        assert_eq!(
            events[0],
            LmStreamEvent::delta(0, LmDelta::text("Par")),
            "the empty opener contributed nothing; the first delta is content"
        );
        assert_eq!(events[1], LmStreamEvent::delta(0, LmDelta::text("is")));
        assert!(matches!(
            &events[2],
            LmStreamEvent::OutputEnd { finish_reason, truncated, .. }
                if finish_reason.as_deref() == Some("stop") && !truncated
        ));
    }

    /// The whole point of the typed vocabulary: the frames' events reassemble into the reply, via
    /// the same [`LmOutputBuilder`](crate::lm::api::LmOutputBuilder) the live stream drives.
    #[test]
    fn the_events_reassemble_into_the_reply() {
        use crate::lm::api::LmOutputBuilder;
        let mut builder = LmOutputBuilder::new();
        builder
            .apply(LmStreamEvent::Start {
                model: Some("openai/gpt-4o".to_owned()),
            })
            .expect("start applies");
        let mut response = None;
        for event in events_of(SSE) {
            if let Some(assembled) = builder.apply(event).expect("applies") {
                response = Some(assembled);
            }
        }
        let response = response.expect("the end event produced a reply");
        assert_eq!(response.first_text(), "Paris");
        assert_eq!(response.usage.and_then(|u| u.total()), Some(14));
    }

    /// A reasoning model streams its reasoning ahead of the answer: the `reasoning_content` deltas
    /// build a thinking part at index 0, and the content that follows takes index 1 — the
    /// thinking-then-text order the non-streamed reply has.
    #[test]
    fn reasoning_content_streams_as_a_thinking_part_before_the_text() {
        use crate::lm::api::{LmOutputBuilder, LmPart};
        let sse = "data: {\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"2+2\"}}]}\n\n\
            data: {\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\" = 4\"}}]}\n\n\
            data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"The answer is 4.\"}}]}\n\n\
            data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
            data: [DONE]\n\n";

        let mut builder = LmOutputBuilder::new();
        builder.apply(LmStreamEvent::Start { model: None }).expect("start");
        let mut response = None;
        for event in events_of(sse) {
            if let Some(assembled) = builder.apply(event).expect("applies") {
                response = Some(assembled);
            }
        }
        let output = &response.expect("assembled").outputs[0];
        assert!(
            matches!(&output.parts[0], LmPart::Thinking { text, .. } if text == "2+2 = 4"),
            "thinking is first, got {:?}", output.parts,
        );
        assert!(matches!(&output.parts[1], LmPart::Text { text, .. } if text == "The answer is 4."));
    }

    /// A streamed tool call arrives in fragments — id and name first, arguments a slice at a time
    /// — and reassembles through the builder into one `ToolCall` part with its arguments whole.
    #[test]
    fn a_streamed_tool_call_reassembles_through_the_builder() {
        use crate::lm::api::{LmOutputBuilder, LmPart};
        let sse = "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"\"}}]}}]}\n\n\
            data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"city\\\":\"}}]}}]}\n\n\
            data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"Paris\\\"}\"}}]}}]}\n\n\
            data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n\
            data: [DONE]\n\n";

        let mut builder = LmOutputBuilder::new();
        builder
            .apply(LmStreamEvent::Start { model: None })
            .expect("start");
        let mut response = None;
        for event in events_of(sse) {
            if let Some(assembled) = builder.apply(event).expect("applies") {
                response = Some(assembled);
            }
        }
        let output = &response.expect("assembled").outputs[0];
        assert_eq!(output.finish_reason.as_deref(), Some("tool_calls"));
        let LmPart::ToolCall { name, args, .. } = &output.parts[0] else {
            panic!("expected a tool call, got {:?}", output.parts)
        };
        assert_eq!(name, "get_weather");
        assert_eq!(args["city"], serde_json::json!("Paris"), "arguments reassembled whole");
    }

    #[test]
    fn a_finish_reason_of_length_marks_the_output_truncated() {
        let events =
            events_of("data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"x\"},\"finish_reason\":\"length\"}]}\n\ndata: [DONE]\n\n");
        assert!(events.iter().any(|event| matches!(
            event,
            LmStreamEvent::OutputEnd { truncated: true, .. }
        )));
    }
}
