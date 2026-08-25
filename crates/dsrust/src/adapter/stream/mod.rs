//! Streaming one field of a reply as its text arrives — dspy's `StreamListener`.
//!
//! A ChatAdapter reply names each field with `[[ ## name ## ]]`, and a streamed reply's deltas
//! cross those boundaries mid-token. This watches the delta stream for one field's marker and
//! hands back that field's text, and nothing else, until the next marker closes it.
//!
//! **The chunk boundaries are the contract, not just the concatenation.** A caller streaming into a
//! UI renders what it is handed, so a listener that regrouped `"To"`, `" get"`, `" to"` into
//! `"To"`, `" ge"`, `"t to"` would split words mid-render — which is what this did before, by
//! buffering a fixed number of *characters*. Upstream keeps the model's own token boundaries by
//! holding whole deltas and deciding per delta whether the buffer could still be growing into the
//! closing marker. `the_chunks_are_dspys_own_token_boundaries` holds it to a stream recorded from
//! gpt-4o-mini in upstream's `test_stream_listener_returns_correct_chunk_chat_adapter`.

mod json;
mod partial;
mod program;
mod status;

pub use json::JsonFieldListener;
pub use partial::{is_complete, keys_with_values};
pub use program::{Streamed, StreamedField, Watching, streamify};
pub use status::{DefaultStatus, StatusMessages};

use std::collections::VecDeque;

use anyhow::Result;
use futures_util::{Stream, StreamExt};

use crate::lm::api::{LmDelta, LmStreamEvent};

/// How many deltas may sit in the buffer while they could still be forming the closing marker.
///
/// dspy's own number and its own reasoning: "10 is a heuristic number that is sufficient to capture
/// the end_identifier for all LMs". It counts *deltas*, not characters — which is what keeps a
/// flush aligned to the model's token boundaries.
const BUFFERED_DELTAS: usize = 10;

/// What one wire's field boundaries look like — dspy's `adapter_identifiers`, whose entries differ
/// only in these strings. Keeping them a value rather than a second listener is upstream's own
/// arrangement: one `receive`, a table of identifiers.
struct Wire {
    /// What opens the field being watched.
    start: String,
    /// The character that could be `start` beginning — upstream's `start_indicator`.
    indicator: char,
    /// What closes it. For markers this is the *next* field's marker, since a section ends where
    /// the following one begins; for XML it is this field's own closing tag.
    end: String,
    /// The prefixes a buffer may end with while `end` could still be forming.
    end_prefixes: &'static [&'static str],
    /// What a buffer containing it could still be growing into — upstream's
    /// `end_pattern_contains`, which is *not* the end identifier. They coincide on the marker wire
    /// and do not on the tag wire: `</` could be any closing tag, while `</answer>` is only ours,
    /// and holding on the former is what keeps the closing tag from leaking out a piece at a time.
    end_contains: &'static str,
}

impl Wire {
    /// dspy `ChatAdapter`: `[[ ## name ## ]]`, closed by whichever marker comes next.
    fn markers(field: &str) -> Self {
        Self {
            start: format!("[[ ## {field} ## ]]"),
            indicator: '[',
            end: "[[ ##".to_owned(),
            end_prefixes: &["[", "[[", "[[ ", "[[ #", "[[ ##"],
            end_contains: "[[ ##",
        }
    }

    /// dspy `XMLAdapter`: `<name>`, closed by `</name>` — its *own* tag, so an unclosed field is
    /// never ended by the next one starting.
    fn tags(field: &str) -> Self {
        Self {
            start: format!("<{field}>"),
            indicator: '<',
            end: format!("</{field}>"),
            end_prefixes: &["<", "</"],
            end_contains: "</",
        }
    }

    /// Whether the buffer could still be growing into the closing identifier — upstream's
    /// `_could_form_end_identifier`, which is what lets an ordinary token be released at once
    /// instead of waiting out the ten-delta window.
    fn could_close(&self, buffered: &str) -> bool {
        self.end_prefixes
            .iter()
            .any(|prefix| buffered.ends_with(prefix))
            || buffered.contains(self.end_contains)
    }
}

/// One piece of a field's text, as the model produced it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldChunk {
    pub text: String,
    /// dspy's `StreamResponse.is_last_chunk`: the field's closing marker has been seen, so nothing
    /// further will arrive for it.
    pub is_last: bool,
}

/// Watches a ChatAdapter reply for one field, handing back its text as the deltas complete it.
pub struct FieldListener {
    wire: Wire,
    /// Deltas seen since the first that could be opening the marker, held until they either
    /// complete it or prove they cannot.
    opening: Vec<String>,
    /// Deltas of the field's own text, held only while they could be forming the closing marker.
    pending: VecDeque<String>,
    started: bool,
    ended: bool,
}

impl FieldListener {
    /// A listener over the marker wire — dspy's `ChatAdapter`.
    pub fn new(field: &str) -> Self {
        Self::over(Wire::markers(field))
    }

    /// A listener over the tag wire — dspy's `XMLAdapter`. The same rule with different
    /// identifiers, which is how upstream arranges it too.
    pub fn xml(field: &str) -> Self {
        Self::over(Wire::tags(field))
    }

    fn over(wire: Wire) -> Self {
        Self {
            wire,
            opening: Vec::new(),
            pending: VecDeque::new(),
            started: false,
            ended: false,
        }
    }

    /// Feed one delta; get whatever of the field it now completes, if any.
    ///
    /// dspy's `StreamListener.receive`, which answers with at most one `StreamResponse` per chunk.
    pub fn push(&mut self, delta: &str) -> Option<FieldChunk> {
        if self.ended || delta.is_empty() {
            return None;
        }
        // A cache hit — or a model whose chunk is the whole reply, as gemini's can be — carries the
        // field's opening *and* closing marker at once, so it never streams.
        if !self.started
            && let Some(at) = delta.find(&self.wire.start)
            && delta[at + self.wire.start.len()..].contains(&self.wire.end)
        {
            self.started = true;
            self.ended = true;
            return None;
        }
        let text = match self.started {
            true => delta.to_owned(),
            false => self.opening(delta)?,
        };
        self.emit(text)
    }

    /// The stream ended without the field's closing marker arriving. dspy's `finalize`: whatever is
    /// still buffered is the field's, and it is the last of it.
    pub fn finish(&mut self) -> Option<FieldChunk> {
        if self.ended || !self.started {
            return None;
        }
        self.ended = true;
        let text = self.flush();
        (!text.is_empty()).then_some(FieldChunk {
            text,
            is_last: true,
        })
    }

    /// Look for the field's opening marker across deltas, answering with whatever text follows it
    /// once it has fully arrived.
    fn opening(&mut self, delta: &str) -> Option<String> {
        if self.opening.is_empty() {
            // Upstream buffers the first delta that could be opening the field and answers nothing
            // for it, testing only once a second arrives. That matters even when the whole
            // identifier is inside this one delta: the trim below is applied to the *joined* text,
            // so acting a delta early drops a different amount of leading space.
            if delta.contains(self.wire.indicator) {
                self.opening.push(delta.to_owned());
            }
            return None;
        }
        self.opening.push(delta.to_owned());
        let concat: String = self.opening.concat();
        if let Some(at) = concat.find(&self.wire.start) {
            self.started = true;
            self.opening.clear();
            return Some(concat[at + self.wire.start.len()..].trim_start().to_owned());
        }
        // Keep looking only while some suffix of what we have could be the marker's own start;
        // anything else was a `[` that meant nothing.
        if !ends_with_prefix_of(concat.trim(), &self.wire.start) {
            self.opening.clear();
        }
        None
    }

    /// Hold the delta, then hand back whatever can no longer be part of the closing marker.
    fn emit(&mut self, delta: String) -> Option<FieldChunk> {
        if delta.is_empty() {
            return None;
        }
        self.pending.push_back(delta);
        let mut token = match self.wire.could_close(held(&self.pending).trim()) {
            // Nothing here could be the marker forming, so all of it is the field's.
            false => self.flush(),
            // Otherwise release the oldest, once the buffer is longer than the marker could need.
            true if self.pending.len() > BUFFERED_DELTAS => {
                self.pending.pop_front().unwrap_or_default()
            }
            true => String::new(),
        };
        if held(&self.pending).trim().contains(&self.wire.end) {
            self.ended = true;
            token.push_str(&self.flush());
            token = token.trim_end().to_owned();
        }
        (!token.is_empty() || self.ended).then_some(FieldChunk {
            text: token,
            is_last: self.ended,
        })
    }

    /// Everything still held, cut at the closing marker if it is in there.
    fn flush(&mut self) -> String {
        let held: String = self.pending.drain(..).collect();
        // Cut at the closing identifier's own opening, which for a marker is `[[` and for a tag is
        // `</` — upstream cuts at exactly these two.
        let cut = self
            .wire
            .end_contains
            .get(..2)
            .unwrap_or(self.wire.end_contains);
        match held.find(cut) {
            Some(at) => held[..at].to_owned(),
            None => held,
        }
    }
}

/// The buffered deltas as one string. `VecDeque` has no `concat`, and joining is what every
/// decision here is made against.
pub(super) fn held(pending: &VecDeque<String>) -> String {
    pending.iter().map(String::as_str).collect()
}

/// Whether any suffix of `text` is a prefix of `identifier` — upstream's
/// `_buffered_message_end_with_start_identifier`, the test that keeps a marker split across deltas
/// alive without holding every `[` forever.
pub(super) fn ends_with_prefix_of(text: &str, identifier: &str) -> bool {
    (1..=text.len()).any(|len| {
        text.is_char_boundary(text.len() - len) && identifier.starts_with(&text[text.len() - len..])
    })
}

/// One field's text streamed out of an LM event stream — dspy's `streamify` over one predictor's
/// output. Non-text events pass silently; the field's tokens come out in order, on the model's own
/// boundaries.
pub fn stream_field<'a>(
    events: impl Stream<Item = Result<LmStreamEvent>> + Send + 'a,
    field: &str,
) -> impl Stream<Item = Result<FieldChunk>> + Send + 'a {
    let mut listener = FieldListener::new(field);
    events.filter_map(move |event| {
        let out = match event {
            Ok(LmStreamEvent::Delta {
                delta: LmDelta::TextDelta { text },
                ..
            }) => listener.push(&text).map(Ok),
            // A stream that stops without the closing marker still owes the caller what it held.
            Ok(LmStreamEvent::End { .. }) => listener.finish().map(Ok),
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

    fn chunks(listener: &mut FieldListener, deltas: &[&str]) -> Vec<String> {
        deltas
            .iter()
            .filter_map(|delta| listener.push(delta))
            .map(|chunk| chunk.text)
            .collect()
    }

    /// The chunks a caller is handed are the model's own tokens, one per delta.
    ///
    /// The stream is upstream's, recorded from gpt-4o-mini in
    /// `test_stream_listener_returns_correct_chunk_chat_adapter`, and so is the expected list. This
    /// is the assertion the older listener failed: it buffered a fixed number of *characters* and
    /// answered `["To", " ge", "t to", " the o", …]`, which concatenates to the same text and
    /// renders as words split down the middle.
    #[test]
    fn the_chunks_are_dspys_own_token_boundaries() {
        let mut listener = FieldListener::new("answer");
        let streamed = chunks(
            &mut listener,
            &[
                "[[",
                " ##",
                " answer",
                " ##",
                " ]]\n\n",
                "To",
                " get",
                " to",
                " the",
                " other",
                " side",
                " of",
                " the",
                " dinner",
                " plate",
                "!\n\n[[ ##",
                " completed",
                " ##",
                " ]]",
            ],
        );
        assert_eq!(
            streamed,
            [
                "To", " get", " to", " the", " other", " side", " of", " the", " dinner", " plate",
                "!"
            ]
        );
    }

    /// The chunk that carries the closing marker is the last, and says so — dspy's
    /// `is_last_chunk`, which is how a caller knows to stop rendering rather than to wait.
    #[test]
    fn the_closing_marker_marks_the_last_chunk() {
        let mut listener = FieldListener::new("answer");
        let seen: Vec<FieldChunk> = [
            "[[ ## answer ## ]]\n",
            "Par",
            "is",
            "\n\n[[ ## completed ## ]]",
        ]
        .iter()
        .filter_map(|delta| listener.push(delta))
        .collect();
        let text: String = seen.iter().map(|chunk| chunk.text.as_str()).collect();
        assert_eq!(text, "Paris");
        let last = seen.last().expect("a chunk");
        assert!(last.is_last);
        // Empty, and upstream says why in its own test: "An empty chunk with is_last_chunk=True is
        // emitted to properly mark field end". The delta carrying the closing marker holds nothing
        // else, and the field's text has already gone out on its own token boundaries.
        assert_eq!(last.text, "");
        assert!(
            seen[..seen.len() - 1].iter().all(|chunk| !chunk.is_last),
            "only the closing chunk is the last one"
        );
    }

    /// The field's text comes out between its marker and the next, whatever else surrounds it —
    /// the reasoning section before it is discarded, the completed marker after it closes it.
    #[test]
    fn a_field_streams_out_between_its_marker_and_the_next() {
        let mut listener = FieldListener::new("answer");
        let streamed = chunks(
            &mut listener,
            &[
                "[[ ## reasoning ## ]]\nbecause the sky",
                " scatters blue",
                "\n\n[[ ## answer ## ]]\n",
                "Par",
                "is",
                "\n\n[[ ## completed ## ]]",
            ],
        )
        .concat();
        assert_eq!(
            streamed, "Paris",
            "only the answer field, no markers, no trailing newline"
        );
    }

    /// A marker split across two deltas is still recognised — the reason the watcher holds the
    /// deltas rather than deciding on each alone.
    #[test]
    fn a_marker_split_across_deltas_is_still_found() {
        let mut listener = FieldListener::new("answer");
        let streamed = chunks(
            &mut listener,
            &[
                "[[ ## ans",
                "wer ## ]]\nBer",
                "lin",
                "\n\n[[ ## completed ## ]]",
            ],
        )
        .concat();
        assert_eq!(
            streamed, "Berlin",
            "the marker split across two deltas still opened the field"
        );
    }

    /// A reply that arrives whole — a cache hit, or a model whose chunk is the entire answer —
    /// opens and closes the field in one delta, and streams nothing.
    #[test]
    fn a_reply_that_arrives_whole_streams_nothing() {
        let mut listener = FieldListener::new("answer");
        let streamed = chunks(
            &mut listener,
            &["[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]"],
        );
        assert!(streamed.is_empty(), "got: {streamed:?}");
    }

    /// A stream that stops while something is still held owes the caller that much — dspy's
    /// `finalize`. The trailing `[` was buffered because it could have been the closing marker
    /// forming; once the stream ends it is just text.
    #[test]
    fn a_stream_that_ends_early_flushes_what_it_held() {
        let mut listener = FieldListener::new("answer");
        let sent = chunks(&mut listener, &["[[ ## answer ## ]]\n", "Paris", "["]);
        assert_eq!(sent, ["Paris"], "the bracket is held, not sent");
        let last = listener.finish().expect("the buffered tail");
        assert_eq!(last.text, "[");
        assert!(last.is_last);
    }

    /// And a stream whose tokens all went out as they arrived has nothing left to finalize, which
    /// is the ordinary case: a token that cannot be the marker forming is released at once.
    #[test]
    fn a_stream_that_held_nothing_finalizes_to_nothing() {
        let mut listener = FieldListener::new("answer");
        let sent = chunks(&mut listener, &["[[ ## answer ## ]]\n", "Par", "is"]);
        assert_eq!(sent, ["Par", "is"]);
        assert!(listener.finish().is_none());
    }

    #[tokio::test]
    async fn stream_field_pulls_the_field_from_an_event_stream() {
        let events = vec![
            Ok(LmStreamEvent::Start { model: None }),
            Ok(LmStreamEvent::delta(
                0,
                LmDelta::text("[[ ## answer ## ]]\n"),
            )),
            Ok(LmStreamEvent::delta(0, LmDelta::text("Par"))),
            Ok(LmStreamEvent::delta(0, LmDelta::text("is"))),
            Ok(LmStreamEvent::delta(
                0,
                LmDelta::text("\n\n[[ ## completed ## ]]"),
            )),
            Ok(LmStreamEvent::end()),
        ];
        let streamed: String = stream_field(stream::iter(events), "answer")
            .map(|piece| piece.expect("a piece").text)
            .collect()
            .await;
        assert_eq!(streamed, "Paris");
    }
}
