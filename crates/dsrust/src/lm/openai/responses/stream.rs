//! One Responses SSE frame as the events it carries.
//!
//! The Responses wire streams each output item's delta on its own event name, so where the chat
//! wire has one `choices[].delta` to read, this has a small table: which event carries text, which
//! carries reasoning, which closes the reply. `response.completed` closes with the whole reply,
//! `response.failed`/`incomplete` with an error.

use serde_json::Value;

use super::responses_to_lm_response;
use crate::lm::api::{LmDelta, LmStreamEvent};
use crate::lm::streaming::{Framed, StreamState};

/// One Responses SSE frame as the events it carries. The reply's items each stream their own delta;
/// `response.completed` closes with the whole reply, `response.failed`/`incomplete` with an error.
pub(super) fn frame(frame: &str, _state: &mut StreamState) -> Framed {
    let Some(data) = frame
        .lines()
        .find_map(|line| line.trim().strip_prefix("data:"))
    else {
        return Framed::of(Vec::new());
    };
    let Ok(event) = serde_json::from_str::<Value>(data.trim()) else {
        return Framed::of(Vec::new());
    };
    let delta = |delta| Framed::of(vec![at(&event, delta)]);
    match event["type"].as_str() {
        Some("response.output_text.delta") => delta(LmDelta::text(event_delta(&event))),
        Some("response.reasoning_summary_text.delta" | "response.reasoning_text.delta") => {
            delta(LmDelta::thinking(event_delta(&event)))
        }
        Some("response.function_call_arguments.delta") => delta(LmDelta::ToolCallDelta {
            id: None,
            name: None,
            args_delta: event["delta"].as_str().map(str::to_owned),
        }),
        Some("response.output_item.added") if event["item"]["type"] == "function_call" => {
            let item = &event["item"];
            delta(LmDelta::ToolCallDelta {
                id: item["call_id"]
                    .as_str()
                    .or_else(|| item["id"].as_str())
                    .map(str::to_owned),
                name: item["name"].as_str().map(str::to_owned),
                args_delta: None,
            })
        }
        Some("response.completed") => {
            Framed::complete(Vec::new(), responses_to_lm_response(&event["response"], ""))
        }
        Some("response.failed" | "response.incomplete") => {
            let detail = event["response"]["error"]["message"]
                .as_str()
                .unwrap_or("the responses stream did not complete");
            Framed::closing(vec![LmStreamEvent::Error {
                error: detail.to_owned(),
            }])
        }
        _ => Framed::of(Vec::new()),
    }
}

/// One output item's delta, under its own part index so reasoning, text and a tool call accumulate
/// separately. There is a single candidate, so the output index is always zero.
fn at(event: &Value, delta: LmDelta) -> LmStreamEvent {
    LmStreamEvent::Delta {
        output_index: 0,
        part_index: event["output_index"].as_u64().unwrap_or(0) as usize,
        delta,
    }
}

fn event_delta(event: &Value) -> &str {
    event["delta"].as_str().unwrap_or_default()
}

// -------- reply: Responses body -> LmResponse --------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lm::api::{LmDelta, LmStreamEvent};
    use crate::lm::streaming::StreamState;

    /// Only a `function_call` item announces a tool call on add; a message item added must emit
    /// nothing. The guard's mutants announced a nameless call for every item, or none ever.
    #[test]
    fn only_a_function_call_item_announces_itself_on_add() {
        let mut state = StreamState::default();
        let call = frame(
            "data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"f\"}}",
            &mut state,
        );
        let [
            LmStreamEvent::Delta {
                delta: LmDelta::ToolCallDelta { id, name, .. },
                ..
            },
        ] = &call.events[..]
        else {
            panic!("one tool-call delta: {:?}", call.events)
        };
        assert_eq!(id.as_deref(), Some("call_1"));
        assert_eq!(name.as_deref(), Some("f"));

        let mut state = StreamState::default();
        let message = frame(
            "data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"message\"}}",
            &mut state,
        );
        assert!(message.events.is_empty(), "{:?}", message.events);
    }

    /// A failed or incomplete response closes the stream with the provider's own message — the
    /// arm deleted, a dead stream just went silent and the caller waited on nothing.
    #[test]
    fn a_failed_response_closes_with_the_providers_message() {
        let mut state = StreamState::default();
        let failed = frame(
            "data: {\"type\":\"response.failed\",\"response\":{\"error\":{\"message\":\"overloaded\"}}}",
            &mut state,
        );
        assert!(failed.done, "the stream is over");
        let [LmStreamEvent::Error { error }] = &failed.events[..] else {
            panic!("one error event: {:?}", failed.events)
        };
        assert_eq!(error, "overloaded");

        let mut state = StreamState::default();
        let incomplete = frame("data: {\"type\":\"response.incomplete\"}", &mut state);
        assert!(incomplete.done);
        let [LmStreamEvent::Error { error }] = &incomplete.events[..] else {
            panic!("one error event: {:?}", incomplete.events)
        };
        assert_eq!(error, "the responses stream did not complete");
    }
}
