//! Streaming one field of a reply as its text arrives — dspy's `StreamListener` and `streamify`.
//!
//! A ChatAdapter reply names each field with `[[ ## name ## ]]`, and a streamed reply's deltas
//! cross those boundaries mid-token. This watches the delta stream for one field's marker and
//! hands back that field's text, and nothing else, until the next marker closes it — the essence
//! of dspy's `streamify(program, StreamListener(signature_field_name=...))` for a single
//! predictor's output over the ChatAdapter.

use anyhow::Result;
use futures_util::{Stream, StreamExt};

use crate::lm::api::{LmDelta, LmStreamEvent};

/// The sentinel that opens any field's section — the start of the *next* one is what closes the
/// field being watched.
const MARKER: &str = "[[ ##";

/// Watches a ChatAdapter reply for one field, handing back its text as the deltas complete it.
pub struct FieldListener {
    start: String,
    buffer: String,
    inside: bool,
    done: bool,
}

impl FieldListener {
    pub fn new(field: &str) -> Self {
        Self {
            start: format!("[[ ## {field} ## ]]"),
            buffer: String::new(),
            inside: false,
            done: false,
        }
    }

    /// Feed one delta; get whatever of the field it now completes, if any.
    pub fn push(&mut self, delta: &str) -> Option<String> {
        if self.done {
            return None;
        }
        self.buffer.push_str(delta);
        if !self.inside && !self.enter() {
            return None;
        }
        self.emit()
    }

    /// Enter the field once its marker has fully arrived, dropping everything up to it. While it
    /// has not, only a tail that could be the marker forming across the next delta is kept.
    fn enter(&mut self) -> bool {
        match self.buffer.find(&self.start) {
            Some(at) => {
                self.buffer.drain(..at + self.start.len());
                // The value begins on the line after its marker.
                self.buffer = self.buffer.trim_start_matches('\n').to_owned();
                self.inside = true;
                true
            }
            None => {
                let keep = self.start.len().saturating_sub(1);
                if self.buffer.len() > keep {
                    self.buffer.drain(..self.buffer.len() - keep);
                }
                false
            }
        }
    }

    /// Yield the field's text up to the next marker, holding back a suffix that could be a marker
    /// forming so a partial one never leaks out.
    fn emit(&mut self) -> Option<String> {
        if let Some(at) = self.buffer.find(MARKER) {
            let value = self.buffer[..at].trim_end().to_owned();
            self.done = true;
            self.buffer.clear();
            return (!value.is_empty()).then_some(value);
        }
        let keep = MARKER.len().saturating_sub(1);
        if self.buffer.len() <= keep {
            return None;
        }
        let value: String = self.buffer.drain(..self.buffer.len() - keep).collect();
        (!value.is_empty()).then_some(value)
    }
}

/// One field's text streamed out of an LM event stream — dspy's `streamify` over one predictor's
/// output. Non-text events pass silently; the field's tokens come out in order.
pub fn stream_field<'a>(
    events: impl Stream<Item = Result<LmStreamEvent>> + Send + 'a,
    field: &str,
) -> impl Stream<Item = Result<String>> + Send + 'a {
    let mut listener = FieldListener::new(field);
    events.filter_map(move |event| {
        let out = match event {
            Ok(LmStreamEvent::Delta {
                delta: LmDelta::TextDelta { text },
                ..
            }) => listener.push(&text).map(Ok),
            Ok(_) => None,
            Err(error) => Some(Err(error)),
        };
        std::future::ready(out)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::stream;

    /// The field's text comes out between its marker and the next, whatever else surrounds it —
    /// the reasoning section before it is discarded, the completed marker after it closes it.
    #[test]
    fn a_field_streams_out_between_its_marker_and_the_next() {
        let mut listener = FieldListener::new("answer");
        let deltas = [
            "[[ ## reasoning ## ]]\nbecause the sky",
            " scatters blue",
            "\n\n[[ ## answer ## ]]\n",
            "Par",
            "is",
            "\n\n[[ ## completed ## ]]",
        ];
        let streamed: String = deltas.iter().filter_map(|delta| listener.push(delta)).collect();
        assert_eq!(streamed, "Paris", "only the answer field, no markers, no trailing newline");
    }

    /// A marker split across two deltas is still recognised — the reason the watcher holds a tail
    /// rather than deciding on each delta alone.
    #[test]
    fn a_marker_split_across_deltas_is_still_found() {
        let mut listener = FieldListener::new("answer");
        let streamed: String = ["[[ ## ans", "wer ## ]]\nBer", "lin", "\n\n[[ ## completed ## ]]"]
            .iter()
            .filter_map(|delta| listener.push(delta))
            .collect();
        assert_eq!(streamed, "Berlin", "the marker split across two deltas still opened the field");
    }

    #[tokio::test]
    async fn stream_field_pulls_the_field_from_an_event_stream() {
        let events = vec![
            Ok(LmStreamEvent::Start { model: None }),
            Ok(LmStreamEvent::delta(0, LmDelta::text("[[ ## answer ## ]]\n"))),
            Ok(LmStreamEvent::delta(0, LmDelta::text("Par"))),
            Ok(LmStreamEvent::delta(0, LmDelta::text("is"))),
            Ok(LmStreamEvent::delta(0, LmDelta::text("\n\n[[ ## completed ## ]]"))),
            Ok(LmStreamEvent::end()),
        ];
        let streamed: String = stream_field(stream::iter(events), "answer")
            .map(|piece| piece.expect("a piece"))
            .collect()
            .await;
        assert_eq!(streamed, "Paris");
    }
}
