//! Streaming one field of a JSON reply — dspy's `StreamListener` over `JSONAdapter`.
//!
//! The shape is the ChatAdapter listener's and the two rules that matter are different. A field
//! opens on `"name":` rather than a marker, and it *closes* when the accumulated object has given a
//! second key a value — there is no closing sentinel to watch for, so upstream partial-parses what
//! it has after every delta. [`partial`](super::partial) answers exactly that question, and is held
//! to `jiter`'s own answers prefix by prefix.
//!
//! One consequence worth stating because it looks like a bug: the chunks carry their quotes. dspy
//! streams the field's *raw JSON text*, so the first chunk of `"answer": "To get…"` is `"To` and
//! the last is `!"`. A listener that stripped them would be nicer and would not be upstream.

use std::collections::VecDeque;

use super::partial::{is_complete, keys_with_values};
use super::{FieldChunk, held};

/// How many deltas may sit in the buffer while they could still be closing the object.
const BUFFERED_DELTAS: usize = 10;

/// The character that could be the field's key starting — upstream's `start_indicator`.
const INDICATOR: char = '"';

/// The prefixes a buffer may end with while a JSON value could still be closing — upstream's
/// `end_pattern_prefixes` for this adapter.
const END_PREFIXES: [&str; 4] = ["\"", "\",", "\" ", "\"}"];

/// Watches a JSON reply for one field, handing back its text as the deltas complete it.
pub struct JsonFieldListener {
    field: String,
    start: String,
    /// Deltas seen since the first that could be opening the key, held until they either complete
    /// it or prove they cannot.
    opening: Vec<String>,
    /// Deltas of the field's own text, held only while the object could still be closing.
    pending: VecDeque<String>,
    /// The object as far as it has been written, opened with a `{` this adds itself so that the
    /// *next* key is detectable — upstream does the same, for the same reason.
    accumulated: String,
    started: bool,
    ended: bool,
}

impl JsonFieldListener {
    pub fn new(field: &str) -> Self {
        Self {
            field: field.to_owned(),
            start: format!("\"{field}\":"),
            opening: Vec::new(),
            pending: VecDeque::new(),
            accumulated: String::new(),
            started: false,
            ended: false,
        }
    }

    /// Feed one delta; get whatever of the field it now completes, if any.
    pub fn push(&mut self, delta: &str) -> Option<FieldChunk> {
        if self.ended || delta.is_empty() {
            return None;
        }
        let text = match self.started {
            true => delta.to_owned(),
            // No cache-hit shortcut here: upstream guards that branch with
            // `not isinstance(settings.adapter, JSONAdapter)`, because a whole object in one delta
            // is read by the ordinary end detection below.
            false => self.opening(delta)?,
        };
        self.emit(text)
    }

    /// The stream ended without the object closing. dspy's `finalize`.
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

    /// Look for `"field":` across deltas, answering with whatever follows it once it has arrived.
    fn opening(&mut self, delta: &str) -> Option<String> {
        if self.opening.is_empty() {
            // Upstream buffers the first delta that could be opening the field and answers nothing
            // for it, testing only once a second arrives. That matters even when the whole
            // identifier is inside this one delta: the trim below is applied to the *joined* text,
            // so acting a delta early drops a different amount of leading space.
            if delta.contains(INDICATOR) {
                self.opening.push(delta.to_owned());
            }
            return None;
        }
        self.opening.push(delta.to_owned());
        let concat: String = self.opening.concat();
        if let Some(at) = concat.find(&self.start) {
            self.started = true;
            self.opening.clear();
            // The opening brace is this crate's, as it is upstream's: without it the accumulated
            // text is not an object and nothing can be parsed out of it at all.
            self.accumulated.push('{');
            self.accumulated.push_str(&self.start);
            return Some(concat[at + self.start.len()..].trim_start().to_owned());
        }
        if !super::ends_with_prefix_of(concat.trim(), &self.start) {
            self.opening.clear();
        }
        None
    }

    /// Hold the delta, then hand back whatever the object has settled.
    fn emit(&mut self, delta: String) -> Option<FieldChunk> {
        if delta.is_empty() {
            return None;
        }
        self.pending.push_back(delta.clone());
        let mut token = match could_close(held(&self.pending).trim()) {
            false => self.flush(),
            true if self.pending.len() > BUFFERED_DELTAS => {
                self.pending.pop_front().unwrap_or_default()
            }
            true => String::new(),
        };
        self.accumulated.push_str(&delta);

        // The object closed: everything up to its last brace is the field's, and nothing follows.
        if self.accumulated.trim_end().ends_with('}') && is_complete(self.accumulated.trim_end()) {
            self.ended = true;
            let last = self.flush();
            let upto = last.rfind('}').unwrap_or(last.len());
            token.push_str(&last[..upto]);
            return Some(FieldChunk {
                text: token,
                is_last: true,
            });
        }
        // Or a second key was given a value, which is the next field starting.
        if let Some(next) = self.next_field() {
            self.ended = true;
            let last = self.flush();
            let upto = last.find(&next).unwrap_or(last.len());
            token.push_str(&last[..upto]);
        }
        (!token.is_empty() || self.ended).then_some(FieldChunk {
            text: token,
            is_last: self.ended,
        })
    }

    /// The name of the next key to have been given a value, if one has — upstream's
    /// `len(parsed) > 1` and the search for the first key that is not ours.
    fn next_field(&self) -> Option<String> {
        let keys = keys_with_values(&self.accumulated)?;
        (keys.len() > 1)
            .then(|| keys.into_iter().find(|key| *key != self.field))
            .flatten()
    }

    /// Everything still held. Unlike the ChatAdapter listener there is nothing to cut at: the
    /// caller above decides where the field ends, from the object rather than from a sentinel.
    fn flush(&mut self) -> String {
        self.pending.drain(..).collect()
    }
}

/// Whether the buffer could still be closing the object — upstream's `_could_form_end_identifier`
/// for this adapter.
fn could_close(buffered: &str) -> bool {
    END_PREFIXES.iter().any(|prefix| buffered.ends_with(prefix)) || buffered.contains('}')
}
