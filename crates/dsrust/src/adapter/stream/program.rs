//! Running a whole program and watching one of its fields arrive — dspy's `streamify`.
//!
//! The listeners in this module's siblings read a *model's* deltas. This is the layer above: it
//! runs a [`Module`], taps whatever model that module reaches, and hands back a stream that yields
//! the watched field's chunks as they arrive and finally the prediction.
//!
//! **The tap is a model, not a hook in `Module`.** dspy does the same — its listeners see chunks
//! because the LM publishes them, not because the program was rewritten — and it is what keeps
//! this working for a module nobody anticipated: a caller's own `Module`, several predictors deep,
//! needs no change to be watched. [`lm::context_model`](crate::lm::context_model) scopes the tap
//! for the duration of the run and puts the original model back, so nothing outside the stream
//! sees it.

use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow};
use futures_channel::mpsc;
use futures_util::{Stream, StreamExt};

use super::FieldChunk;
use crate::example::{Example, Prediction};
use crate::lm::api;
use crate::lm::{ChatModel, DynChatModel};
use crate::module::Module;

/// What a streamed run hands back, in the order it happens.
///
/// dspy yields the same two kinds from `streamify` — `StreamResponse` while the program runs, then
/// the `Prediction` — and a caller tells them apart the same way.
#[derive(Debug, Clone)]
pub enum Streamed {
    /// One piece of the watched field, on the model's own token boundary.
    Chunk(FieldChunk),
    /// The program's answer, once it is done. Always last, and always exactly one.
    Answer(Prediction),
}

/// The model a streamed run installs in place of the real one: it asks the real one for its events,
/// publishes each delta, and hands back the answer the events carried.
struct Tap {
    inner: Arc<dyn DynChatModel>,
    deltas: Mutex<mpsc::UnboundedSender<String>>,
}

impl ChatModel for Tap {
    async fn forward(&self, request: &api::LmRequest) -> Result<api::LmResponse> {
        let mut events = self.inner.forward_stream_dyn(request);
        let mut answered = None;
        while let Some(event) = events.next().await {
            match event? {
                api::LmStreamEvent::Delta {
                    delta: api::LmDelta::TextDelta { text },
                    ..
                } => {
                    // A closed receiver means the caller stopped reading the stream. That is not a
                    // failure of the call, so the run goes on and the answer is still returned.
                    // `unbounded_send` rather than the `Sink` method: it cannot block, so nothing
                    // is awaited while the lock is held.
                    let _ = self
                        .deltas
                        .lock()
                        .expect("not poisoned")
                        .unbounded_send(text);
                }
                // The answer rides on `End`, both on a provider's own stream and on the one-delta
                // form a model that cannot stream reports.
                api::LmStreamEvent::End { response, .. } => {
                    answered = response.map(|boxed| *boxed);
                }
                api::LmStreamEvent::Error { error } => return Err(anyhow!(error)),
                _ => {}
            }
        }
        answered.ok_or_else(|| anyhow!("the model's stream ended without an answer"))
    }

    fn capabilities(&self) -> impl Future<Output = crate::lm::Capabilities> + Send {
        self.inner.capabilities_dyn()
    }

    fn native_reasoning_usable(&self) -> bool {
        self.inner.native_reasoning_usable_dyn()
    }

    fn native_citations_usable(&self) -> bool {
        self.inner.native_citations_usable_dyn()
    }
}

/// Run `module` and watch one field of its replies arrive — dspy's
/// `streamify(program, stream_listeners=[StreamListener(signature_field_name=…)])`.
///
/// The chunks come out as the model produces them, on its own token boundaries, and the prediction
/// comes last. A model that cannot stream reports its answer whole, so a scripted double still
/// yields one chunk and then the answer — which is how upstream tests its own listeners.
///
/// ```no_run
/// # use dsrust::{Example, Module, adapter::stream::{Streamed, streamify}};
/// # use futures_util::StreamExt;
/// # async fn watch(program: impl Module, inputs: Example) -> anyhow::Result<()> {
/// let mut running = std::pin::pin!(streamify(&program, "answer", inputs));
/// while let Some(next) = running.next().await {
///     match next? {
///         Streamed::Chunk(chunk) => print!("{}", chunk.text),
///         Streamed::Answer(answer) => println!("\n{:?}", answer.get("answer")),
///     }
/// }
/// # Ok(()) }
/// ```
pub fn streamify<'a, M: Module + ?Sized>(
    module: &'a M,
    field: &str,
    inputs: Example,
) -> impl Stream<Item = Result<Streamed>> + 'a {
    let (sender, deltas) = mpsc::unbounded();
    let tap = Arc::new(Tap {
        inner: crate::lm::global::current().unwrap_or_else(|_| Arc::new(NoModel)),
        deltas: Mutex::new(sender),
    });
    let running = crate::lm::global::context_model(crate::lm::global::client(), tap)
        .run(module.forward(inputs));
    watched(Box::pin(running), deltas, super::FieldListener::new(field))
}

/// The two halves interleaved: the deltas as they are published, then the answer.
///
/// A hand-written stream rather than a spawned task, because this crate names no runtime — the
/// module's future is polled by whoever polls this, which is also what keeps the scoped model
/// correct, since [`Scope::run`](crate::lm::Scope::run) enters on each poll.
fn watched<'a>(
    running: std::pin::Pin<Box<dyn Future<Output = Result<Prediction>> + 'a>>,
    mut deltas: mpsc::UnboundedReceiver<String>,
    mut listener: super::FieldListener,
) -> impl Stream<Item = Result<Streamed>> + 'a {
    // Held in an `Option` so that finishing it *drops* it. The future owns the scope, which owns
    // the tap, which owns the sender — so until it is dropped the channel stays open and the
    // receiver never reports the end. Keeping the finished future around would hang the stream.
    let mut running = Some(running);
    let mut answered: Option<Result<Prediction>> = None;
    futures_util::stream::poll_fn(move |context| {
        use std::task::Poll;
        loop {
            // Whatever the model has already published comes out first, so a chunk is never held
            // back behind the rest of the run.
            match deltas.poll_next_unpin(context) {
                Poll::Ready(Some(text)) => {
                    if let Some(chunk) = listener.push(&text) {
                        return Poll::Ready(Some(Ok(Streamed::Chunk(chunk))));
                    }
                    continue;
                }
                Poll::Ready(None) => {
                    // The tap is gone, so the run is over: flush the listener, then the answer.
                    if let Some(chunk) = listener.finish() {
                        return Poll::Ready(Some(Ok(Streamed::Chunk(chunk))));
                    }
                    return Poll::Ready(answered.take().map(|answer| answer.map(Streamed::Answer)));
                }
                Poll::Pending => {}
            }
            let Some(future) = running.as_mut() else {
                // Finished and dropped, so the channel above is closed and will report the end.
                return Poll::Pending;
            };
            match future.as_mut().poll(context) {
                Poll::Ready(answer) => {
                    answered = Some(answer);
                    // Dropping it closes the channel, so the branch above drains and then ends.
                    running = None;
                    continue;
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    })
}

/// Stands in when nothing is configured, so `streamify` reports that rather than panicking.
struct NoModel;

impl ChatModel for NoModel {
    async fn forward(&self, _request: &api::LmRequest) -> Result<api::LmResponse> {
        Err(anyhow!(
            "no model configured; call lm::configure(...) first"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lm::dummy::DummyLM;
    use crate::predict::Predict;
    use crate::signature::{InField, OutField, Signature};

    fn signature() -> Signature {
        Signature {
            instructions: "Answer the question.".into(),
            inputs: vec![InField {
                name: "question".into(),
                ..Default::default()
            }],
            outputs: vec![OutField {
                name: "answer".into(),
                ..Default::default()
            }],
        }
    }

    /// A model that streams its reply in pieces, which is what upstream's own streaming tests use
    /// — their doubles are async generators yielding delta after delta, not a `DummyLM`.
    struct Piecewise(Vec<&'static str>);

    impl ChatModel for Piecewise {
        async fn forward(&self, _request: &api::LmRequest) -> Result<api::LmResponse> {
            Ok(api::LmResponse::text(self.0.concat()))
        }

        fn forward_stream<'a>(
            &'a self,
            _request: &'a api::LmRequest,
        ) -> impl Stream<Item = Result<api::LmStreamEvent>> + Send + 'a {
            let mut events: Vec<Result<api::LmStreamEvent>> =
                vec![Ok(api::LmStreamEvent::Start { model: None })];
            events.extend(
                self.0
                    .iter()
                    .map(|piece| Ok(api::LmStreamEvent::delta(0, api::LmDelta::text(*piece)))),
            );
            events.push(Ok(api::LmStreamEvent::End {
                usage: None,
                cost: None,
                response: Some(Box::new(api::LmResponse::text(self.0.concat()))),
            }));
            futures_util::stream::iter(events)
        }
    }

    /// A whole program run, watched: the field's text arrives in pieces, then the prediction.
    ///
    /// The pieces are the model's own, which is the whole point — the chunks a caller renders are
    /// the boundaries the model produced, not a regrouping of them.
    #[tokio::test]
    async fn a_run_yields_its_field_then_its_answer() {
        let lm = std::sync::Arc::new(Piecewise(vec![
            "[[ ## answer ## ]]\n",
            "Par",
            "is",
            "\n\n[[ ## completed ## ]]",
        ]));
        let program = Predict::from_signature(signature());
        let watched = crate::lm::global::context_model(reqwest::Client::new(), lm).run(async {
            let mut seen = Vec::new();
            let mut running = std::pin::pin!(streamify(
                &program,
                "answer",
                crate::input! { question: "where?" }
            ));
            while let Some(next) = running.next().await {
                seen.push(next.expect("a streamed item"));
            }
            seen
        });
        let seen = watched.await;

        let chunks: Vec<&str> = seen
            .iter()
            .filter_map(|item| match item {
                Streamed::Chunk(chunk) => Some(chunk.text.as_str()),
                _ => None,
            })
            .collect();
        // The model's own boundaries, not a regrouping of them — and the empty last chunk is
        // dspy's way of marking the field's end, which the listener goldens pin.
        assert_eq!(chunks, ["Par", "is", ""]);
        let last_chunk = seen
            .iter()
            .filter_map(|item| match item {
                Streamed::Chunk(chunk) => Some(chunk),
                _ => None,
            })
            .next_back()
            .expect("a chunk");
        assert!(last_chunk.is_last, "the closing chunk says so");

        let answer = seen
            .last()
            .and_then(|item| match item {
                Streamed::Answer(answer) => Some(answer),
                _ => None,
            })
            .expect("the answer comes last");
        assert_eq!(answer.get("answer").and_then(|v| v.as_str()), Some("Paris"));
    }

    /// And it ends. The run's future owns the scope, which owns the tap, which owns the sender —
    /// so a stream that kept the finished future alive would wait on a channel nothing will ever
    /// close. This test hangs rather than fails if that regresses, which is why it is bounded.
    #[tokio::test]
    async fn a_finished_run_closes_the_stream() {
        let lm = std::sync::Arc::new(DummyLM::new([crate::example! { answer: "Berlin" }]));
        let program = Predict::from_signature(signature());
        let bounded = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            crate::lm::global::context_model(reqwest::Client::new(), lm).run(async {
                streamify(&program, "answer", crate::input! { question: "where?" })
                    .count()
                    .await
            }),
        );
        let count = bounded.await.expect("the stream ended rather than hanging");
        assert!(count > 0, "the run yielded nothing at all");
    }
}
