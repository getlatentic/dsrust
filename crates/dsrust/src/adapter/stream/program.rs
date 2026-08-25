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
    /// One piece of a watched field, on the model's own token boundary, and which field of which
    /// predictor it belongs to.
    Chunk(StreamedField),
    /// What the program is doing, when a [`StatusMessages`] was given words for the stage —
    /// dspy's `StatusMessage`, which rides the same stream as the chunks.
    Status(String),
    /// The program's answer, once it is done. Always last, and always exactly one.
    Answer(Prediction),
}

/// One piece of a watched field — dspy's `StreamResponse`, whose four fields these are.
///
/// The naming matters once more than one field is watched: two predictors streaming at once are
/// told apart by `predictor`, and a caller rendering into separate places needs `field`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamedField {
    /// dspy's `predict_name`: the predictor this field belongs to, as
    /// [`named_predictors`](crate::Module::named_predictors) names it.
    pub predictor: String,
    /// dspy's `signature_field_name`.
    pub field: String,
    /// dspy's `chunk`.
    pub text: String,
    /// dspy's `is_last_chunk`.
    pub is_last: bool,
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
/// # async fn watch(mut program: impl Module, inputs: Example) -> anyhow::Result<()> {
/// let mut running = std::pin::pin!(streamify(&mut program, "answer", inputs));
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
    module: &'a mut M,
    field: &str,
    inputs: Example,
) -> impl Stream<Item = Result<Streamed>> + 'a {
    Watching::new(module, field).run(inputs)
}

/// A streamed run being set up — the optional half of dspy's `streamify`, whose extras are keyword
/// arguments it has no Rust spelling for.
pub struct Watching<'a, M: ?Sized> {
    module: &'a M,
    /// Each watched field with the predictor that answers it, resolved up front so a failure to
    /// match is reported before the program runs rather than as silence while it does.
    fields: Result<Vec<(String, String)>>,
    messages: Option<Arc<dyn StatusMessages>>,
}

impl<'a, M: Module + ?Sized> Watching<'a, M> {
    /// Watch `field`, on whichever predictor of `module` answers it.
    ///
    /// Takes `&mut` because finding that predictor means walking the program's predictors, and
    /// this crate's walk hands out `&mut Signature` — upstream's `named_predictors` is read-only
    /// and needs no such thing. The run itself only reads the module.
    pub fn new(module: &'a mut M, field: &str) -> Self {
        let matched = matching(module, &[field.to_owned()]);
        Self {
            module,
            fields: matched,
            messages: None,
        }
    }

    /// Watch several fields at once — dspy's `stream_listeners=[…]`.
    ///
    /// Each is matched to its own predictor, so two of them streaming in one program are told
    /// apart by [`StreamedField::predictor`].
    pub fn all(module: &'a mut M, fields: &[&str]) -> Self {
        let asked: Vec<String> = fields.iter().map(|field| (*field).to_owned()).collect();
        let matched = matching(module, &asked);
        Self {
            module,
            fields: matched,
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
        // A field that matched no predictor, or more than one, is reported as the stream's first
        // and only item — upstream raises before the program runs, and this is where a stream says
        // the same thing.
        let watching = match self.fields {
            Ok(fields) => fields,
            Err(refused) => {
                return futures_util::future::Either::Left(futures_util::stream::once(
                    std::future::ready(Err(refused)),
                ));
            }
        };
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
        let listeners = watching
            .into_iter()
            .map(|(field, predictor)| Listening {
                listener: super::FieldListener::new(&field),
                predictor,
                field,
            })
            .collect();
        futures_util::future::Either::Right(watched(running, published, listeners))
    }
}

/// The two halves interleaved: the deltas as they are published, then the answer.
///
/// A hand-written stream rather than a spawned task, because this crate names no runtime — the
/// module's future is polled by whoever polls this, which is also what keeps the scoped model
/// correct, since [`Scope::run`](crate::lm::Scope::run) enters on each poll.
/// Which predictor answers each watched field — dspy's `find_predictor_for_stream_listeners`.
///
/// Both refusals are upstream's, wording included, and both are worth having: a field on two
/// predictors cannot be attributed, and a field on none would watch a program that never produces
/// it and report nothing, which reads as a model that said nothing rather than as a mistake.
fn matching<M: Module + ?Sized>(
    module: &mut M,
    fields: &[String],
) -> Result<Vec<(String, String)>> {
    let mut answered: Vec<(String, Option<String>)> =
        fields.iter().map(|field| (field.clone(), None)).collect();
    for predictor in module.named_predictors() {
        for output in &predictor.signature.outputs {
            let Some((field, found)) = answered.iter_mut().find(|(field, _)| *field == output.name)
            else {
                continue;
            };
            if found.is_some() {
                return Err(anyhow!(
                    "Signature field {field} is not unique in the program, cannot automatically \
                     determine which predictor to use for streaming. Please specify the predictor \
                     to listen to."
                ));
            }
            *found = Some(predictor.name.clone());
        }
    }
    answered
        .into_iter()
        .map(|(field, found)| match found {
            Some(predictor) => Ok((field, predictor)),
            None => Err(anyhow!(
                "Signature field {field} is not a field of any predictor in the program, cannot \
                 automatically determine which predictor to use for streaming. Please verify your \
                 field name or specify the predictor to listen to."
            )),
        })
        .collect()
}

/// One listener with the names its chunks carry.
struct Listening {
    listener: super::FieldListener,
    predictor: String,
    field: String,
}

fn watched<'a>(
    running: std::pin::Pin<Box<dyn Future<Output = Result<Prediction>> + 'a>>,
    mut published: mpsc::UnboundedReceiver<Tapped>,
    mut listeners: Vec<Listening>,
) -> impl Stream<Item = Result<Streamed>> + 'a {
    // What one delta produced but the stream has not handed out yet: a delta can complete a chunk
    // for more than one listener, and a stream yields one item at a time.
    let mut ready: std::collections::VecDeque<StreamedField> = std::collections::VecDeque::new();
    // Held in an `Option` so that finishing it *drops* it. The future owns the scope, which owns
    // the tap, which owns the sender — so until it is dropped the channel stays open and the
    // receiver never reports the end. Keeping the finished future around would hang the stream.
    let mut running = Some(running);
    let mut answered: Option<Result<Prediction>> = None;
    futures_util::stream::poll_fn(move |context| {
        use std::task::Poll;
        loop {
            if let Some(chunk) = ready.pop_front() {
                return Poll::Ready(Some(Ok(Streamed::Chunk(chunk))));
            }
            // Whatever the model has already published comes out first, so a chunk is never held
            // back behind the rest of the run.
            match published.poll_next_unpin(context) {
                Poll::Ready(Some(Tapped::Status(message))) => {
                    return Poll::Ready(Some(Ok(Streamed::Status(message))));
                }
                Poll::Ready(Some(Tapped::Delta(text))) => {
                    // Every listener sees every delta. Upstream routes each to the predictor its
                    // field belongs to; the matching has already made that field unique across the
                    // program, so a listener only ever fires on the reply carrying its own marker
                    // — the same result, at the cost upstream's `allow_reuse` note describes.
                    for watching in &mut listeners {
                        if let Some(chunk) = watching.listener.push(&text) {
                            ready.push_back(StreamedField {
                                predictor: watching.predictor.clone(),
                                field: watching.field.clone(),
                                text: chunk.text,
                                is_last: chunk.is_last,
                            });
                        }
                    }
                    continue;
                }
                Poll::Ready(None) => {
                    // The tap is gone, so the run is over: flush each listener, then the answer.
                    for watching in &mut listeners {
                        if let Some(chunk) = watching.listener.finish() {
                            ready.push_back(StreamedField {
                                predictor: watching.predictor.clone(),
                                field: watching.field.clone(),
                                text: chunk.text,
                                is_last: chunk.is_last,
                            });
                        }
                    }
                    if let Some(chunk) = ready.pop_front() {
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
        let mut program = Predict::from_signature(signature());
        let watched = crate::lm::global::context_model(reqwest::Client::new(), lm).run(async {
            let mut seen = Vec::new();
            let mut running = std::pin::pin!(streamify(
                &mut program,
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
        let mut program = Predict::from_signature(signature());
        let bounded = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            crate::lm::global::context_model(reqwest::Client::new(), lm).run(async {
                streamify(&mut program, "answer", crate::input! { question: "where?" })
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
        let mut program = Predict::from_signature(signature());
        let seen = crate::lm::global::context_model(reqwest::Client::new(), lm)
            .run(async {
                Watching::new(&mut program, "answer")
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
        let mut program = Predict::from_signature(signature());
        let seen = crate::lm::global::context_model(reqwest::Client::new(), lm)
            .run(async {
                streamify(&mut program, "answer", crate::input! { question: "where?" })
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

    /// Two fields on two predictors, told apart by the predictor that answered each — dspy's
    /// `predict_name`, which is why `StreamResponse` carries it.
    #[tokio::test]
    async fn two_watched_fields_carry_the_predictor_that_answered_them() {
        /// A two-step program: one predictor answers, a second judges.
        struct Pair {
            answering: Predict,
            judging: Predict,
        }

        impl Module for Pair {
            fn forward<'b>(
                &'b self,
                inputs: Example,
            ) -> std::pin::Pin<Box<dyn Future<Output = Result<Prediction>> + Send + 'b>>
            {
                Box::pin(async move {
                    let answered = self.answering.forward(inputs).await?;
                    let judged = self.judging.forward(crate::input! { answer: "x" }).await?;
                    Ok(Prediction::new(judged.example, answered.raw))
                })
            }

            fn named_predictors(&mut self) -> Vec<crate::module::NamedPredictor<'_>> {
                let mut all = self.answering.named_predictors();
                all[0].name = "answering".to_owned();
                let mut judging = self.judging.named_predictors();
                judging[0].name = "judging".to_owned();
                all.extend(judging);
                all
            }
        }

        let judged = Signature {
            instructions: "Judge it.".into(),
            inputs: vec![InField {
                name: "answer".into(),
                ..Default::default()
            }],
            outputs: vec![OutField {
                name: "judgement".into(),
                ..Default::default()
            }],
        };
        let mut program = Pair {
            answering: Predict::from_signature(signature()),
            judging: Predict::from_signature(judged),
        };

        let lm = std::sync::Arc::new(Piecewise(vec![
            "[[ ## answer ## ]]\n",
            "Paris",
            "\n\n[[ ## completed ## ]]",
        ]));
        let seen = crate::lm::global::context_model(reqwest::Client::new(), lm)
            .run(async {
                Watching::all(&mut program, &["answer", "judgement"])
                    .run(crate::input! { question: "where?" })
                    .collect::<Vec<_>>()
                    .await
            })
            .await;

        let named: Vec<(String, String)> = seen
            .iter()
            .filter_map(|item| match item {
                Ok(Streamed::Chunk(chunk)) if !chunk.text.is_empty() => {
                    Some((chunk.predictor.clone(), chunk.field.clone()))
                }
                _ => None,
            })
            .collect();
        assert!(
            named
                .iter()
                .all(|(predictor, field)| predictor == "answering" && field == "answer"),
            "only the answering predictor streamed, and it is named: {named:?}"
        );
    }

    /// A field no predictor answers is refused before the program runs, in upstream's words —
    /// watching it would report nothing and read as a model that said nothing.
    #[tokio::test]
    async fn a_field_no_predictor_answers_is_refused() {
        let mut program = Predict::from_signature(signature());
        let refused = Watching::new(&mut program, "nonesuch")
            .run(crate::input! { question: "where?" })
            .collect::<Vec<_>>()
            .await;
        let error = refused
            .into_iter()
            .next()
            .expect("a refusal")
            .expect_err("a refusal");
        assert!(
            format!("{error}").contains("is not a field of any predictor in the program"),
            "got: {error}"
        );
    }

    /// And a field two predictors answer is refused too, because nothing could say which one a
    /// chunk came from.
    #[tokio::test]
    async fn a_field_two_predictors_answer_is_refused() {
        struct Twice(Predict, Predict);

        impl Module for Twice {
            fn forward<'b>(
                &'b self,
                inputs: Example,
            ) -> std::pin::Pin<Box<dyn Future<Output = Result<Prediction>> + Send + 'b>>
            {
                Box::pin(self.0.forward(inputs))
            }

            fn named_predictors(&mut self) -> Vec<crate::module::NamedPredictor<'_>> {
                let mut all = self.0.named_predictors();
                all.extend(self.1.named_predictors());
                all
            }
        }

        let mut program = Twice(
            Predict::from_signature(signature()),
            Predict::from_signature(signature()),
        );
        let refused = Watching::new(&mut program, "answer")
            .run(crate::input! { question: "where?" })
            .collect::<Vec<_>>()
            .await;
        let error = refused
            .into_iter()
            .next()
            .expect("a refusal")
            .expect_err("a refusal");
        assert!(
            format!("{error}").contains("is not unique in the program"),
            "got: {error}"
        );
    }
}
