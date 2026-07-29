//! Anthropic streaming: its named SSE events read back as the typed [`LmStreamEvent`]s.
//!
//! There is no dspy reference for this mapping — dspy 3.3 streams Anthropic through litellm — so it
//! is faithful to Anthropic's own event wire: usage split across two events (the input with
//! `message_start`, the output with `message_delta`), text arriving as `content_block_delta`, and
//! `message_stop` closing. The reading, framing and assembly are shared; this is the frame's meaning.

use std::time::Duration;

use anyhow::Result;
use futures_util::Stream;
use serde_json::{Value, json};

use super::request::request;
use crate::lm::api::{self, LmDelta, LmStreamEvent};
use crate::lm::streaming::{Framed, Framing, StreamState};

/// The streaming form of the call: the same body with `stream` set, its event stream read back as
/// the typed vocabulary. Its owned inputs are lifted out before the stream is built, so it borrows
/// only the client.
pub(crate) fn stream<'h>(
    http: &'h reqwest::Client,
    model: &str,
    api_key: Option<&str>,
    timeout: Duration,
    call: &api::LmRequest,
) -> impl Stream<Item = Result<api::LmStreamEvent>> + Send + use<'h> {
    let mut body = request(model, call);
    body["stream"] = json!(true);
    let mut request = http
        .post(super::MESSAGES_URL)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .timeout(timeout)
        .json(&body);
    if let Some(key) = api_key {
        request = request.header("x-api-key", key);
    }
    crate::lm::streaming::events(
        request.send(),
        "anthropic".to_owned(),
        model.to_owned(),
        Framing {
            separator: b"\n\n",
            frame,
        },
    )
}

/// One Anthropic SSE frame — an `event:`/`data:` pair — as the events it carries.
fn frame(frame: &str, state: &mut StreamState) -> Framed {
    let Some(data) = frame
        .lines()
        .find_map(|line| line.trim().strip_prefix("data:"))
    else {
        return Framed::of(Vec::new());
    };
    let Ok(event) = serde_json::from_str::<Value>(data.trim()) else {
        return Framed::of(Vec::new());
    };
    match event["type"].as_str() {
        // The input side, cache included; Anthropic reports the output only at message_delta.
        Some("message_start") => {
            state.usage = super::usage(&event["message"]["usage"]);
            Framed::of(Vec::new())
        }
        // A tool-use block opens with its id and name, its arguments arriving as later deltas.
        Some("content_block_start") if event["content_block"]["type"] == "tool_use" => {
            let block = &event["content_block"];
            Framed::of(vec![block_delta(
                &event,
                LmDelta::ToolCallDelta {
                    id: block["id"].as_str().map(str::to_owned),
                    name: block["name"].as_str().map(str::to_owned),
                    args_delta: None,
                },
            )])
        }
        // A block's increment, under its own index: text, extended-thinking text, or a slice of a
        // tool call's json arguments. A signature delta carries nothing to reassemble.
        Some("content_block_delta") => {
            let delta = &event["delta"];
            let mapped = match delta["type"].as_str() {
                Some("text_delta") => {
                    Some(LmDelta::text(delta["text"].as_str().unwrap_or_default()))
                }
                Some("thinking_delta") => Some(LmDelta::thinking(
                    delta["thinking"].as_str().unwrap_or_default(),
                )),
                Some("input_json_delta") => Some(LmDelta::ToolCallDelta {
                    id: None,
                    name: None,
                    args_delta: delta["partial_json"].as_str().map(str::to_owned),
                }),
                _ => None,
            };
            Framed::of(
                mapped
                    .into_iter()
                    .map(|delta| block_delta(&event, delta))
                    .collect(),
            )
        }
        // The final output count, and why generation stopped.
        Some("message_delta") => {
            let mut merged = state.usage.take().unwrap_or_default();
            merged.output_tokens = event["usage"]["output_tokens"].as_u64().map(|n| n as u32);
            merged.completion_tokens = merged.output_tokens;
            merged.total_tokens = None;
            state.usage = Some(merged.fill_aliases());
            let ended = event["delta"]["stop_reason"]
                .as_str()
                .map(|reason| {
                    vec![LmStreamEvent::OutputEnd {
                        output_index: 0,
                        finish_reason: Some(reason.to_owned()),
                        truncated: reason == "max_tokens",
                    }]
                })
                .unwrap_or_default();
            Framed::of(ended)
        }
        Some("message_stop") => Framed::closing(Vec::new()),
        _ => Framed::of(Vec::new()),
    }
}

/// A block's delta under its own index, which is the part it accumulates into — Anthropic numbers
/// its content blocks, so thinking, text and a tool call each land in their own part in order.
fn block_delta(event: &Value, delta: LmDelta) -> LmStreamEvent {
    LmStreamEvent::Delta {
        output_index: 0,
        part_index: event["index"].as_u64().unwrap_or(0) as usize,
        delta,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Extended thinking and a tool call stream under their own block indices: a thinking_delta
    /// builds a thinking part at 0, content_block_start opens the tool call at 1, and input_json_delta
    /// fills its arguments — reassembling into a thinking part then a tool call.
    #[test]
    fn thinking_and_tool_use_stream_into_their_own_parts() {
        use crate::lm::api::{LmOutputBuilder, LmPart};
        let sse = "event: message_start\n\
            data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":10,\"output_tokens\":1}}}\n\n\
            event: content_block_delta\n\
            data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"Paris is in France.\"}}\n\n\
            event: content_block_start\n\
            data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"get_weather\",\"input\":{}}}\n\n\
            event: content_block_delta\n\
            data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"city\\\":\"}}\n\n\
            event: content_block_delta\n\
            data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"\\\"Paris\\\"}\"}}\n\n\
            event: message_delta\n\
            data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":8}}\n\n\
            event: message_stop\n\
            data: {\"type\":\"message_stop\"}\n\n";

        let mut builder = LmOutputBuilder::new();
        builder
            .apply(LmStreamEvent::Start { model: None })
            .expect("start");
        let mut state = StreamState::default();
        let mut response = None;
        for block in sse.split("\n\n") {
            let framed = frame(block, &mut state);
            for event in framed.events {
                if let Some(assembled) = builder.apply(event).expect("applies") {
                    response = Some(assembled);
                }
            }
            if framed.done {
                let end = LmStreamEvent::End {
                    usage: state.usage.take(),
                    cost: None,
                    response: None,
                };
                if let Some(assembled) = builder.apply(end).expect("end") {
                    response = Some(assembled);
                }
            }
        }
        let output = &response.expect("assembled").outputs[0];
        assert!(
            matches!(&output.parts[0], LmPart::Thinking { text, .. } if text == "Paris is in France."),
            "thinking at index 0, got {:?}",
            output.parts,
        );
        let LmPart::ToolCall { id, name, args, .. } = &output.parts[1] else {
            panic!("expected a tool call at index 1, got {:?}", output.parts)
        };
        assert_eq!(id.as_deref(), Some("toolu_1"));
        assert_eq!(name, "get_weather");
        assert_eq!(args["city"], serde_json::json!("Paris"));
    }

    /// Anthropic's SSE: usage split across message_start (input) and message_delta (output), the
    /// text arriving as content_block_delta, message_stop closing.
    #[test]
    fn the_sse_events_reassemble_into_the_reply() {
        let sse = "event: message_start\n\
            data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":10,\"output_tokens\":1}}}\n\n\
            event: content_block_delta\n\
            data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Par\"}}\n\n\
            event: content_block_delta\n\
            data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"is\"}}\n\n\
            event: message_delta\n\
            data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":4}}\n\n\
            event: message_stop\n\
            data: {\"type\":\"message_stop\"}\n\n";

        let mut state = StreamState::default();
        let mut text = String::new();
        let mut closed = false;
        let mut stop = None;
        for block in sse.split("\n\n") {
            let framed = frame(block, &mut state);
            for event in framed.events {
                match event {
                    LmStreamEvent::Delta {
                        delta: LmDelta::TextDelta { text: piece },
                        ..
                    } => text.push_str(&piece),
                    LmStreamEvent::OutputEnd { finish_reason, .. } => stop = finish_reason,
                    _ => {}
                }
            }
            closed |= framed.done;
        }
        assert_eq!(text, "Paris");
        assert_eq!(stop.as_deref(), Some("end_turn"));
        assert!(closed, "message_stop closed the stream");
        let usage = state.usage.expect("usage arrived across two events");
        assert_eq!(usage.input_tokens, Some(10), "from message_start");
        assert_eq!(
            usage.output_tokens,
            Some(4),
            "from message_delta, not the opener's 1"
        );
        assert_eq!(usage.total(), Some(14));
    }
}
