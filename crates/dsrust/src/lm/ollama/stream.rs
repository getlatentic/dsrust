//! ollama `/api/chat` streaming: its line-delimited JSON read back as the typed [`LmStreamEvent`]s.
//!
//! There is no dspy reference for this mapping — dspy 3.3 streams ollama through litellm — so it is
//! faithful to ollama's own wire: one reply chunk per line, the last carrying `done`, the counts and
//! the reason generation stopped. The reading, framing and assembly are shared; this is the line's
//! meaning. The `/api/generate` route streams the same way over its own `response` field, in
//! [`generate`](super::generate).

use std::time::Duration;

use anyhow::Result;
use futures_util::Stream;
use serde_json::{Value, json};

use super::request::request;
use crate::lm::api::{self, LmDelta, LmStreamEvent};
use crate::lm::streaming::{Framed, Framing, StreamState};

/// The streaming form: the request with `stream` set true, ollama's line-delimited JSON read back as
/// the typed vocabulary. Each line is a reply chunk; the last carries `done` and the counts.
pub(crate) fn stream(
    http: &reqwest::Client,
    model: &str,
    host: &str,
    api_key: Option<&str>,
    timeout: Duration,
    call: &api::LmRequest,
) -> impl Stream<Item = Result<api::LmStreamEvent>> + Send + use<> {
    let mut body = request(model, call);
    body["stream"] = json!(true);
    let connect = super::authorized(
        http.post(format!("{host}/api/chat"))
            .timeout(timeout)
            .json(&body),
        api_key,
    )
    .send();
    crate::lm::streaming::events(
        connect,
        "ollama".to_owned(),
        model.to_owned(),
        Framing {
            separator: b"\n",
            frame,
        },
    )
}

/// One ollama line as its events. A `done` line closes the stream with the counts and the reason
/// generation stopped.
fn frame(line: &str, state: &mut StreamState) -> Framed {
    let Ok(chunk) = serde_json::from_str::<Value>(line.trim()) else {
        return Framed::of(Vec::new()); // a blank line between objects
    };
    let mut events = Vec::new();
    if let Some(thinking) = chunk["message"]["thinking"].as_str()
        && !thinking.is_empty()
    {
        state.content_offset = 1;
        events.push(LmStreamEvent::Delta {
            output_index: 0,
            part_index: 0,
            delta: LmDelta::thinking(thinking),
        });
    }
    if let Some(content) = chunk["message"]["content"].as_str()
        && !content.is_empty()
    {
        events.push(LmStreamEvent::Delta {
            output_index: 0,
            part_index: state.content_offset,
            delta: LmDelta::text(content),
        });
    }
    if chunk["done"].as_bool() != Some(true) {
        return Framed::of(events);
    }
    if let Some(reported) = super::usage(&chunk) {
        state.usage = Some(reported);
    }
    if let Some(reason) = chunk["done_reason"].as_str() {
        events.push(LmStreamEvent::OutputEnd {
            output_index: 0,
            finish_reason: Some(reason.to_owned()),
            truncated: reason == "length",
        });
    }
    Framed::closing(events)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An empty `thinking` string is silence, not a thinking part: without the filter every plain
    /// chunk would open a thinking delta and shift the text to part 1.
    #[test]
    fn empty_thinking_emits_nothing_and_shifts_nothing() {
        let mut state = StreamState::default();
        let framed = frame(
            r#"{"message": {"thinking": "", "content": "hi"}, "done": false}"#,
            &mut state,
        );
        let [LmStreamEvent::Delta { part_index, .. }] = &framed.events[..] else {
            panic!("one text delta: {:?}", framed.events)
        };
        assert_eq!(*part_index, 0, "text stays at part 0 when nothing thought");
    }

    /// The close carries the reason, and only `length` reads as truncated.
    #[test]
    fn the_close_reason_reads_truncated_only_for_length() {
        let mut state = StreamState::default();
        let cut = frame(
            r#"{"message": {"content": ""}, "done": true, "done_reason": "length"}"#,
            &mut state,
        );
        let [LmStreamEvent::OutputEnd { truncated, .. }] = &cut.events[..] else {
            panic!("one closing event: {:?}", cut.events)
        };
        assert!(truncated);

        let mut state = StreamState::default();
        let done = frame(
            r#"{"message": {"content": ""}, "done": true, "done_reason": "stop"}"#,
            &mut state,
        );
        let [LmStreamEvent::OutputEnd { truncated, .. }] = &done.events[..] else {
            panic!("one closing event: {:?}", done.events)
        };
        assert!(!truncated);
    }

    /// ollama's line-delimited JSON: a reply chunk per line, the last carrying `done` and the counts
    /// named after the passes that produced them.
    #[test]
    fn the_ndjson_lines_reassemble_into_the_reply() {
        let ndjson = "{\"message\":{\"content\":\"Par\"},\"done\":false}\n\
            {\"message\":{\"content\":\"is\"},\"done\":false}\n\
            {\"message\":{\"content\":\"\"},\"done\":true,\"done_reason\":\"stop\",\"prompt_eval_count\":10,\"eval_count\":4}\n";

        let mut state = StreamState::default();
        let mut text = String::new();
        let mut closed = false;
        for line in ndjson.split('\n') {
            let framed = frame(line, &mut state);
            for event in framed.events {
                if let LmStreamEvent::Delta {
                    delta: LmDelta::TextDelta { text: piece },
                    ..
                } = event
                {
                    text.push_str(&piece);
                }
            }
            closed |= framed.done;
        }
        assert_eq!(text, "Paris");
        assert!(closed, "the done line closed the stream");
        assert_eq!(state.usage.and_then(|u| u.total()), Some(14));
    }
}
