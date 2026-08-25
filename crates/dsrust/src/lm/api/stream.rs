//! `LMStream` — events as they arrive, with the assembled reply waiting at the end.

use anyhow::{Result, anyhow};

use super::builder::LmOutputBuilder;
use super::event::LmStreamEvent;
use super::request::LmRequest;
use super::response::LmResponse;

/// Upstream keeps a synchronous and an asynchronous class that differ only in how they iterate.
/// One type over an iterator covers both here, since an async source becomes an iterator of events
/// before it reaches this.
///
/// ```
/// use dsrust::lm::api::{LmDelta, LmRequest, LmStream, LmStreamEvent};
///
/// let request = LmRequest::new("openai/gpt-4o-mini", vec![]);
/// let events = vec![LmStreamEvent::Delta {
///     output_index: 0,
///     part_index: 0,
///     delta: LmDelta::TextDelta { text: "Par".to_owned() },
/// }];
/// // One type over an iterator covers dspy's synchronous and asynchronous classes both, since an
/// // async source becomes an iterator of events before it reaches here.
/// let stream = LmStream::new(request, events.into_iter());
/// assert_eq!(stream.request.model, "openai/gpt-4o-mini");
/// ```
pub struct LmStream<E> {
    pub request: LmRequest,
    events: E,
    builder: LmOutputBuilder,
    result: Option<LmResponse>,
    failed: Option<String>,
}

impl<E: Iterator<Item = LmStreamEvent>> LmStream<E> {
    pub fn new(request: LmRequest, events: E) -> Self {
        Self {
            request,
            events,
            builder: LmOutputBuilder::new(),
            result: None,
            failed: None,
        }
    }

    /// The reply, once the stream has run to its end.
    pub fn result(&self) -> Result<&LmResponse> {
        if let Some(failed) = &self.failed {
            return Err(anyhow!(failed.clone()));
        }
        self.result
            .as_ref()
            .ok_or_else(|| anyhow!("the stream has not completed yet"))
    }

    /// Run to the end and answer with the reply, for a caller that wants the result rather than
    /// the events.
    pub fn collect(mut self) -> Result<LmResponse> {
        for _ in self.by_ref() {}
        self.result().cloned()
    }
}

/// A failure is remembered rather than yielded, so a caller draining the events still learns
/// about it from [`LmStream::result`] instead of finding a silently short reply.
impl<E: Iterator<Item = LmStreamEvent>> Iterator for LmStream<E> {
    type Item = LmStreamEvent;

    fn next(&mut self) -> Option<Self::Item> {
        let event = self.events.next()?;
        match self.builder.apply(event.clone()) {
            Ok(Some(response)) => self.result = Some(response),
            Ok(None) => {}
            Err(error) => self.failed = Some(error.to_string()),
        }
        Some(event)
    }
}

#[cfg(test)]
mod tests {
    use super::super::delta::LmDelta;
    use super::super::message::LmMessage;
    use super::super::part::LmPart;
    use super::*;

    fn request() -> LmRequest {
        LmRequest::new(
            "openai/gpt-4o",
            vec![LmMessage::user(vec![LmPart::text("Why?")])],
        )
    }

    fn events() -> Vec<LmStreamEvent> {
        vec![
            LmStreamEvent::Start {
                model: Some("openai/gpt-4o".to_owned()),
            },
            LmStreamEvent::delta(0, LmDelta::text("Par")),
            LmStreamEvent::delta(0, LmDelta::text("is")),
            LmStreamEvent::end(),
        ]
    }

    #[test]
    fn every_event_reaches_the_caller_and_the_reply_waits_at_the_end() {
        let mut stream = LmStream::new(request(), events().into_iter());
        let seen: Vec<LmStreamEvent> = stream.by_ref().collect();

        assert_eq!(seen.len(), 4, "nothing is swallowed on the way through");
        assert_eq!(stream.result().expect("completed").first_text(), "Paris");
    }

    #[test]
    fn the_reply_is_not_available_until_the_stream_ends() {
        let mut stream = LmStream::new(request(), events().into_iter());
        stream.next();
        assert!(stream.result().is_err(), "still mid-flight");
    }

    #[test]
    fn collecting_runs_to_the_end_and_answers_with_the_reply() {
        let response = LmStream::new(request(), events().into_iter())
            .collect()
            .expect("completes");
        assert_eq!(response.first_text(), "Paris");
    }

    /// Draining the events must not turn a failed stream into a short one that looks complete.
    #[test]
    fn a_failure_surfaces_from_the_result_rather_than_being_lost() {
        let events = vec![
            LmStreamEvent::delta(0, LmDelta::text("partial")),
            LmStreamEvent::Error {
                error: "upstream refused".to_owned(),
            },
        ];
        let mut stream = LmStream::new(request(), events.into_iter());
        for _ in stream.by_ref() {}

        let failure = stream.result().expect_err("the stream failed");
        assert!(failure.to_string().contains("refused"), "got {failure}");
    }
}
