//! Provider-neutral streaming: a byte stream framed and mapped into typed [`LmStreamEvent`]s.
//!
//! Every streaming provider does the same thing over a different wire — connect, read bytes as
//! they arrive, split them into frames, turn each frame into events, and close with the usage the
//! stream reported. Only two things differ: the byte sequence that separates frames (a blank line
//! for the Server-Sent Events OpenAI and Anthropic speak, a newline for ollama's line-delimited
//! JSON) and how one frame maps to events. Those are the [`Framing`] a provider supplies; the
//! rest is here, so a provider's streaming is its mapping and nothing more.

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;

use anyhow::{Result, anyhow};
use bytes::Bytes;
use futures_util::{Stream, StreamExt, stream};

use crate::lm::LmUsage;
use crate::lm::api::{LmResponse, LmStreamEvent};

/// What one frame of the wire contributes.
pub(super) struct Framed {
    pub events: Vec<LmStreamEvent>,
    /// This frame closes the stream — OpenAI's `[DONE]`, ollama's `"done": true`, Anthropic's
    /// `message_stop`. The [`End`](LmStreamEvent::End) is emitted after it, carrying the usage.
    pub done: bool,
    /// The whole reply, when the closing frame carried one rather than leaving it to be reassembled
    /// from the deltas — the Responses API's `response.completed`. It rides the [`End`] as the
    /// authoritative answer, which is what makes a streamed reply match a non-streamed one exactly.
    pub response: Option<Box<LmResponse>>,
}

impl Framed {
    pub(super) fn of(events: Vec<LmStreamEvent>) -> Self {
        Self {
            events,
            done: false,
            response: None,
        }
    }

    pub(super) fn closing(events: Vec<LmStreamEvent>) -> Self {
        Self {
            events,
            done: true,
            response: None,
        }
    }

    /// A closing frame that carries the whole reply, used as-is rather than reassembled.
    pub(super) fn complete(events: Vec<LmStreamEvent>, response: LmResponse) -> Self {
        Self {
            events,
            done: true,
            response: Some(Box::new(response)),
        }
    }
}

/// The state a frame threads across the stream: the usage it accumulates for the final
/// [`End`](LmStreamEvent::End), and the part index the main content sits at — pushed forward by a
/// reasoning part that streams ahead of the text, so the two do not collide at index zero.
#[derive(Default)]
pub(super) struct StreamState {
    pub usage: Option<LmUsage>,
    pub content_offset: usize,
}

/// How a provider frames its stream: the frame separator, and the map from one frame's text to
/// the events it carries (threading [`StreamState`] as it goes).
pub(super) struct Framing {
    pub separator: &'static [u8],
    pub frame: fn(&str, &mut StreamState) -> Framed,
}

type Connect<'h> = Pin<Box<dyn Future<Output = reqwest::Result<reqwest::Response>> + Send + 'h>>;
type Bytes_<'h> = Pin<Box<dyn Stream<Item = reqwest::Result<Bytes>> + Send + 'h>>;

/// Where the stream is: waiting on the response, reading its body, or finished.
enum Phase<'h> {
    Connecting(Connect<'h>),
    Reading(Bytes_<'h>),
    Ended,
}

struct Live<'h> {
    phase: Phase<'h>,
    label: String,
    model: String,
    framing: Framing,
    buffer: Vec<u8>,
    pending: VecDeque<Result<LmStreamEvent>>,
    state: StreamState,
    response: Option<Box<LmResponse>>,
}

/// The typed events of one streaming call: [`Start`](LmStreamEvent::Start) once the response
/// arrives, each frame's events as bytes come in, then [`End`](LmStreamEvent::End) with the usage.
///
/// `connect` is the request future rather than a live response, so the connection is part of the
/// stream — the first poll opens it — and the whole thing borrows only what the future does.
pub(super) fn events<'h>(
    connect: impl Future<Output = reqwest::Result<reqwest::Response>> + Send + 'h,
    label: String,
    model: String,
    framing: Framing,
) -> impl Stream<Item = Result<LmStreamEvent>> + Send + 'h {
    let live = Live {
        phase: Phase::Connecting(Box::pin(connect)),
        label,
        model,
        framing,
        buffer: Vec::new(),
        pending: VecDeque::new(),
        state: StreamState::default(),
        response: None,
    };
    stream::unfold(live, |mut live| async move {
        loop {
            if let Some(event) = live.pending.pop_front() {
                return Some((event, live));
            }
            // The phase is taken by value so a new one can be set without borrowing across it;
            // `Ended` is the resting value it keeps unless a live phase is put back.
            match std::mem::replace(&mut live.phase, Phase::Ended) {
                Phase::Ended => return None,
                Phase::Connecting(future) => connected(&mut live, future.await),
                Phase::Reading(mut body) => match body.next().await {
                    Some(Ok(chunk)) => {
                        live.buffer.extend_from_slice(chunk.as_ref());
                        if !drain(&mut live) {
                            live.phase = Phase::Reading(body);
                        }
                    }
                    Some(Err(error)) => live.fail(error.to_string()),
                    // A body that ends without its closing frame still gets an end.
                    None => live.close(),
                },
            }
        }
    })
}

/// The response, or the error that stands in for it.
fn connected(live: &mut Live, response: reqwest::Result<reqwest::Response>) {
    match response {
        Ok(response) if response.status().is_success() => {
            live.pending.push_back(Ok(LmStreamEvent::Start {
                model: Some(live.model.clone()),
            }));
            live.phase = Phase::Reading(Box::pin(response.bytes_stream()));
        }
        Ok(response) => live.fail(format!("{} {}", live.label, response.status())),
        Err(error) => live.fail(format!("{} streaming request failed: {error}", live.label)),
    }
}

/// Turn every complete frame the buffer now holds into events. Answers whether the stream closed.
fn drain(live: &mut Live) -> bool {
    while let Some(frame) = next_frame(&mut live.buffer, live.framing.separator) {
        let framed = (live.framing.frame)(&frame, &mut live.state);
        live.pending.extend(framed.events.into_iter().map(Ok));
        if let Some(response) = framed.response {
            live.response = Some(response);
        }
        if framed.done {
            live.close();
            return true;
        }
    }
    false
}

/// The next complete frame, taken out of the buffer, or `None` while one is still arriving.
fn next_frame(buffer: &mut Vec<u8>, separator: &[u8]) -> Option<String> {
    let at = buffer
        .windows(separator.len())
        .position(|window| window == separator)?;
    let frame = String::from_utf8_lossy(&buffer[..at]).into_owned();
    buffer.drain(..at + separator.len());
    Some(frame)
}

impl Live<'_> {
    /// End the stream with the usage it reported, and the whole reply if a frame carried one.
    fn close(&mut self) {
        self.pending.push_back(Ok(LmStreamEvent::End {
            usage: self.state.usage.take(),
            cost: None,
            response: self.response.take(),
        }));
        self.phase = Phase::Ended;
    }

    fn fail(&mut self, message: String) {
        self.pending.push_back(Err(anyhow!(message)));
        self.phase = Phase::Ended;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lm::api::LmDelta;

    /// A frame reader that hands back whole frames and keeps a partial one for the next bytes.
    #[test]
    fn frames_are_taken_whole_and_a_partial_one_waits() {
        let mut buffer = b"a\n\nbc\n\nd".to_vec();
        assert_eq!(next_frame(&mut buffer, b"\n\n").as_deref(), Some("a"));
        assert_eq!(next_frame(&mut buffer, b"\n\n").as_deref(), Some("bc"));
        assert_eq!(
            next_frame(&mut buffer, b"\n\n"),
            None,
            "d is still arriving"
        );
        assert_eq!(buffer, b"d", "the partial frame is kept");
    }

    /// The mapping and the closing frame drive an end with the accumulated usage.
    #[test]
    fn a_closing_frame_ends_the_stream_with_its_usage() {
        fn frame(text: &str, state: &mut StreamState) -> Framed {
            match text {
                "done" => Framed::closing(Vec::new()),
                "usage" => {
                    state.usage = Some(LmUsage::counted(3, 4));
                    Framed::of(Vec::new())
                }
                other => Framed::of(vec![LmStreamEvent::delta(0, LmDelta::text(other))]),
            }
        }
        let mut live = Live {
            phase: Phase::Ended,
            label: "test".to_owned(),
            model: "m".to_owned(),
            framing: Framing {
                separator: b"\n",
                frame,
            },
            buffer: b"hi\nusage\ndone\n".to_vec(),
            pending: VecDeque::new(),
            state: StreamState::default(),
            response: None,
        };
        assert!(drain(&mut live), "the done frame closed it");

        let events: Vec<_> = live.pending.into_iter().map(|e| e.unwrap()).collect();
        assert_eq!(events[0], LmStreamEvent::delta(0, LmDelta::text("hi")));
        assert!(matches!(
            events.last(),
            Some(LmStreamEvent::End { usage: Some(u), .. }) if u.total() == Some(7)
        ));
    }
}
