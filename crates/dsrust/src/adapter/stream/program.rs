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
use super::status::{Announcing, StatusMessages};
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
    /// What the program is doing, when a [`StatusMessages`] was given words for the stage —
    /// dspy's `StatusMessage`, which rides the same stream as the chunks.
    Status(String),
    /// The program's answer, once it is done. Always last, and always exactly one.
    Answer(Prediction),
}

/// What the tap and the announcer publish onto the one channel a run reads.
enum Tapped {
    Delta(String),
    Status(String),
}

/// The model a streamed run installs in place of the real one: it asks the real one for its events,
/// publishes each delta, and hands back the answer the events carried.
struct Tap {
    inner: Arc<dyn DynChatModel>,
    deltas: Mutex<mpsc::UnboundedSender<Tapped>>,
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
                        .unbounded_send(Tapped::Delta(text));
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
///         Streamed::Status(saying) => println!("[{saying}]"),
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
    Watching::new(module, field).run(inputs)
}

/// A streamed run being set up — the optional half of dspy's `streamify`, whose extras are keyword
/// arguments it has no Rust spelling for.
pub struct Watching<'a, M: ?Sized> {
    module: &'a M,
    field: String,
    messages: Option<Arc<dyn StatusMessages>>,
}

impl<'a, M: Module + ?Sized> Watching<'a, M> {
    pub fn new(module: &'a M, field: &str) -> Self {
        Self {
            module,
            field: field.to_owned(),
            messages: None,
        }
    }

    /// Say what the program is doing as it goes — dspy's
    /// `streamify(program, status_message_provider=…)`. Without this a run yields only the field's
    /// text and its answer, which is upstream's default too.
    pub fn saying(mut self, messages: Arc<dyn StatusMessages>) -> Self {
        self.messages = Some(messages);
        self
    }

    /// Run it.
    pub fn run(self, inputs: Example) -> impl Stream<Item = Result<Streamed>> + 'a {
        let (sender, published) = mpsc::unbounded();
        let tap = Arc::new(Tap {
            inner: crate::lm::global::current().unwrap_or_else(|_| Arc::new(NoModel)),
            deltas: Mutex::new(sender.clone()),
        });
        let announcing = self.messages.map(|messages| {
            let sink = Mutex::new(sender);
            Arc::new(Announcing {
                messages,
                sink: move |message: String| {
                    let _ = sink
                        .lock()
                        .expect("not poisoned")
                        .unbounded_send(Tapped::Status(message));
                },
            }) as Arc<dyn crate::Callback>
        });
        let scoped = crate::lm::global::context_model(crate::lm::global::client(), tap);
        let asking = self.module.forward(inputs);
        // The announcer is scoped for the run rather than registered process-wide, so it neither
        // silences a caller's own watchers nor outlives the stream — upstream appends to the list
        // it inherited, inside a context, for both reasons.
        let running: std::pin::Pin<Box<dyn Future<Output = Result<Prediction>> + 'a>> =
            match announcing {
                Some(announcing) => {
                    Box::pin(scoped.run(crate::callback::watched_by(announcing).run(asking)))
                }
                None => Box::pin(scoped.run(asking)),
            };
        watched(running, published, super::FieldListener::new(&self.field))
    }
}

/// The two halves interleaved: the deltas as they are published, then the answer.
///
/// A hand-written stream rather than a spawned task, because this crate names no runtime — the
/// module's future is polled by whoever polls this, which is also what keeps the scoped model
/// correct, since [`Scope::run`](crate::lm::Scope::run) enters on each poll.
fn watched<'a>(
    running: std::pin::Pin<Box<dyn Future<Output = Result<Prediction>> + 'a>>,
    mut published: mpsc::UnboundedReceiver<Tapped>,
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
            match published.poll_next_unpin(context) {
                Poll::Ready(Some(Tapped::Status(message))) => {
                    return Poll::Ready(Some(Ok(Streamed::Status(message))));
                }
                Poll::Ready(Some(Tapped::Delta(text))) => {
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

    /// A run that was given words says what it is doing, on the same stream as the field's text.
    ///
    /// Only the two tool stages have default wording upstream, so a program with no tools is
    /// silent until a caller overrides a stage — which is what this does.
    #[tokio::test]
    async fn a_run_can_say_what_it_is_doing() {
        struct Narrating;
        impl super::super::StatusMessages for Narrating {
            fn module_start(&self, module: &str, _inputs: &Example) -> Option<String> {
                Some(format!("{module} started"))
            }
            fn lm_start(&self, _request: &api::LmRequest) -> Option<String> {
                Some("calling the model".to_owned())
            }
        }

        let lm = std::sync::Arc::new(Piecewise(vec![
            "[[ ## answer ## ]]\n",
            "Paris",
            "\n\n[[ ## completed ## ]]",
        ]));
        let program = Predict::from_signature(signature());
        let seen = crate::lm::global::context_model(reqwest::Client::new(), lm)
            .run(async {
                Watching::new(&program, "answer")
                    .saying(std::sync::Arc::new(Narrating))
                    .run(crate::input! { question: "where?" })
                    .collect::<Vec<_>>()
                    .await
            })
            .await;

        let said: Vec<String> = seen
            .iter()
            .filter_map(|item| match item {
                Ok(Streamed::Status(message)) => Some(message.clone()),
                _ => None,
            })
            .collect();
        assert!(
            said.iter().any(|message| message.ends_with("started")),
            "the module stage was announced: {said:?}"
        );
        assert!(
            said.iter().any(|message| message == "calling the model"),
            "the lm stage was announced: {said:?}"
        );
    }

    /// And a run given no words says nothing, which is upstream's default: four of its six stages
    /// return `None` until a caller fills them in.
    #[tokio::test]
    async fn a_run_given_no_words_is_silent() {
        let lm = std::sync::Arc::new(Piecewise(vec![
            "[[ ## answer ## ]]\n",
            "Paris",
            "\n\n[[ ## completed ## ]]",
        ]));
        let program = Predict::from_signature(signature());
        let seen = crate::lm::global::context_model(reqwest::Client::new(), lm)
            .run(async {
                streamify(&program, "answer", crate::input! { question: "where?" })
                    .collect::<Vec<_>>()
                    .await
            })
            .await;
        assert!(
            !seen
                .iter()
                .any(|item| matches!(item, Ok(Streamed::Status(_)))),
            "nothing was announced"
        );
    }
}
