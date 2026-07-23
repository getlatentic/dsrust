//! ollama streaming: its line-delimited JSON read back as the typed [`LmStreamEvent`]s.
//!
//! There is no dspy reference for this mapping — dspy 3.3 streams ollama through litellm — so it is
//! faithful to ollama's own wire: one reply chunk per line, the last carrying `done`, the counts and
//! the reason generation stopped. The reading, framing and assembly are shared; this is the line's
//! meaning.

use anyhow::Result;
use futures_util::Stream;
use serde_json::{Value, json};

use super::request::request;
use crate::lm::api::{self, LmDelta, LmStreamEvent};
use crate::lm::streaming::{Framed, Framing, StreamState};
use crate::lm::PROVIDER_TIMEOUT;

/// The streaming form: the request with `stream` set true, ollama's line-delimited JSON read back as
/// the typed vocabulary. Each line is a reply chunk; the last carries `done` and the counts.
pub(crate) fn stream<'h>(
    http: &'h reqwest::Client,
    model: &str,
    host: &str,
    call: &api::LmRequest,
) -> impl Stream<Item = Result<api::LmStreamEvent>> + Send + use<'h> {
    let mut body = request(model, call);
    body["stream"] = json!(true);
    let connect = http
        .post(format!("{host}/api/chat"))
        .timeout(PROVIDER_TIMEOUT)
        .json(&body)
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
