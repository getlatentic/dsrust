//! What every model implements, in the two forms a Rust caller needs.
//!
//! dspy has one `BaseLM` because Python needs no second shape to store it behind a pointer.
//! `ChatModel` returns `impl Future`, which is what makes it pleasant to implement and impossible
//! to make into a trait object, so `DynChatModel` is the object-safe half and the blanket impl
//! below gives it to every model for free. The global configuration stores the second form, which
//! is what lets a test install a scripted model the way dspy installs a `DummyLM`.

use anyhow::Result;
use futures_util::{Stream, StreamExt};

use super::{Capabilities, api};

/// The object-safe form of [`ChatModel`], so a model can be stored behind a pointer.
///

/// The events a model that cannot stream reports: the answer, whole, as one delta.
///
/// What upstream's scripted models give a listener, and what the defaults below hand back so that
/// *every* model has a streaming form — a listener watching a double is how dspy tests its own.
fn one_shot(answered: Result<api::LmResponse>) -> Vec<Result<api::LmStreamEvent>> {
    let answered = match answered {
        Ok(answered) => answered,
        Err(error) => return vec![Err(error)],
    };
    let mut events = vec![Ok(api::LmStreamEvent::Start { model: None })];
    let text = answered.first_text();
    if !text.is_empty() {
        events.push(Ok(api::LmStreamEvent::delta(0, api::LmDelta::text(text))));
    }
    // The whole answer rides on `End`, as it does on a provider's own stream. Without it a caller
    // watching the events has no way back to the response, which is what anything tapping a model
    // — rather than merely displaying it — needs.
    events.push(Ok(api::LmStreamEvent::End {
        usage: answered.spend(),
        cost: None,
        response: Some(Box::new(answered)),
    }));
    events
}

/// `ChatModel` returns `impl Future`, which is ergonomic to implement and impossible to make
/// into a trait object. Every `ChatModel` gets this for free through the blanket impl below,
/// and the global configuration stores this form — which is what lets a test install a
/// scripted model the way dspy installs a `DummyLM`.
pub trait DynChatModel: Send + Sync {
    /// The object-safe form of [`ChatModel::forward`] — the typed 3.3 boundary behind a pointer,
    /// which is how a module reaching its model through `dyn DynChatModel` asks it.
    fn forward_dyn<'a>(
        &'a self,
        request: &'a api::LmRequest,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<api::LmResponse>> + Send + 'a>>;

    /// The object-safe form of [`ChatModel::forward_stream`] — the events of one call behind a
    /// pointer, which is what lets a listener watch a model it only holds as `dyn`.
    ///
    /// Defaulted, so adding it broke no one: a model written against this trait directly — a test
    /// double, usually — keeps compiling and reports its answer whole. Everything implementing
    /// [`ChatModel`] gets the real form through the blanket impl below.
    fn forward_stream_dyn<'a>(
        &'a self,
        request: &'a api::LmRequest,
    ) -> std::pin::Pin<Box<dyn Stream<Item = Result<api::LmStreamEvent>> + Send + 'a>> {
        Box::pin(
            futures_util::stream::once(self.forward_dyn(request))
                .flat_map(|answered| futures_util::stream::iter(one_shot(answered))),
        )
    }

    /// The object-safe form of [`ChatModel::capabilities`].
    fn capabilities_dyn(&self)
    -> std::pin::Pin<Box<dyn Future<Output = Capabilities> + Send + '_>>;

    /// The object-safe form of [`ChatModel::native_reasoning_usable`] — the `_dyn` name keeps it from
    /// clashing with the inherent one on a model that implements both.
    fn native_reasoning_usable_dyn(&self) -> bool;

    /// The object-safe form of [`ChatModel::dump_state`].
    fn dump_state_dyn(&self) -> Option<serde_json::Map<String, serde_json::Value>>;

    /// The object-safe form of [`ChatModel::native_citations_usable`].
    fn native_citations_usable_dyn(&self) -> bool;
}

impl<T: ChatModel + Send + Sync> DynChatModel for T {
    /// dspy `on_lm_start`/`on_lm_end` happens here, because this is the only place it cannot be
    /// forgotten.
    ///
    /// Upstream decorates `BaseLM.__call__`, so every model it ships *and* every model a caller
    /// writes fires the point. `ChatModel::forward` is a required method and an implementor could
    /// always leave the span out of it; this blanket impl has no second version, and it is the
    /// boundary every module crosses to reach its model. A scripted double is watched here for the
    /// same reason upstream's `DummyLM` is.
    fn forward_dyn<'a>(
        &'a self,
        request: &'a api::LmRequest,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<api::LmResponse>> + Send + 'a>> {
        Box::pin(async move {
            let watch = crate::observe::lm_shown(request, self.callbacks());
            crate::observe::watching(watch, self.forward(request)).await
        })
    }

    fn forward_stream_dyn<'a>(
        &'a self,
        request: &'a api::LmRequest,
    ) -> std::pin::Pin<Box<dyn Stream<Item = Result<api::LmStreamEvent>> + Send + 'a>> {
        Box::pin(self.forward_stream(request))
    }

    fn capabilities_dyn(
        &self,
    ) -> std::pin::Pin<Box<dyn Future<Output = Capabilities> + Send + '_>> {
        Box::pin(self.capabilities())
    }

    fn native_reasoning_usable_dyn(&self) -> bool {
        ChatModel::native_reasoning_usable(self)
    }

    fn native_citations_usable_dyn(&self) -> bool {
        ChatModel::native_citations_usable(self)
    }

    fn dump_state_dyn(&self) -> Option<serde_json::Map<String, serde_json::Value>> {
        ChatModel::dump_state(self)
    }
}

/// The typed 3.3 model boundary: dspy's `forward(request: LMRequest) -> LMResponse`.
///
/// The one seam every model implements — a provider-backed [`LM`](super::LM), a `Cached` wrapper, the
/// scripted doubles a test installs — and the one method a module calls to reach its model. Unit
/// tests script it with canned replies while production speaks to real providers through [`LM`](super::LM).
pub trait ChatModel {
    fn forward<'a>(
        &'a self,
        request: &'a api::LmRequest,
    ) -> impl Future<Output = Result<api::LmResponse>> + Send + 'a;

    /// The same call, as the events arrive — dspy's `stream=True`.
    ///
    /// Defaulted to asking [`forward`](Self::forward) and reporting its whole answer as one delta,
    /// which is what a model that cannot stream gives you and what upstream's own scripted models
    /// give a listener. A provider that can stream overrides it; [`LM`](crate::LM) does.
    ///
    /// It exists on the trait rather than only on `LM` because everything built above streaming
    /// needs *any* model to have a streaming form — a listener watching a scripted double is how
    /// upstream tests its own, and a double with no streaming form could not be watched at all.
    fn forward_stream<'a>(
        &'a self,
        request: &'a api::LmRequest,
    ) -> impl Stream<Item = Result<api::LmStreamEvent>> + Send + 'a {
        futures_util::stream::once(self.forward(request))
            .flat_map(|answered| futures_util::stream::iter(one_shot(answered)))
    }

    /// dspy `BaseLM.__call__`: ask this model directly, with no signature and no adapter.
    ///
    /// The other half of upstream's pair. [`forward`](Self::forward) is the hook a provider
    /// implements; this is the entry a caller uses, and it normalises the input the way upstream's
    /// `__call__` does — through `LMRequest.from_call`, which is
    /// [`LmRequest::from_items`](api::LmRequest::from_items) here.
    ///
    /// ```no_run
    /// # use dsrust::{Assistant, ChatModel, LM, User, items};
    /// # async fn ask() -> anyhow::Result<()> {
    /// let lm = LM::new("openai/gpt-4o-mini")?;
    ///
    /// let answered = lm.call(["What is the capital of France?"]).await?;
    /// let next = lm.call(items![answered, User(["And of Belgium?"])]).await?;
    /// # let _ = next;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// [`items!`](crate::items) is what makes a mixed run of arguments one array: a Rust array holds
    /// one type, so `[answered.into(), turn.into()]` cannot infer the target.
    ///
    /// Defaulted, so a provider gets it by implementing `forward` and nothing else — which is why
    /// no model can be missing it, the same reason upstream decorates `__call__` on the base class.
    ///
    /// Neither this nor [`forward`](Self::forward) names an HTTP client, as dspy's do not. The call
    /// goes out on the configured one, which is where
    /// [`configure_with_client`](crate::configure_with_client) puts a caller's own.
    fn call(
        &self,
        items: impl IntoIterator<Item = impl Into<api::LmItem>>,
    ) -> impl Future<Output = Result<api::LmResponse>> + Send
    where
        Self: Sized + Send + Sync,
    {
        // The model name is the provider's own; every request this crate builds leaves it for
        // whichever wire answers, as an adapter-built one does.
        let request = api::LmRequest::from_items("", items);
        // Through `forward_dyn` and not `forward`, so asking a model directly is watched exactly as
        // asking it through a module is. Upstream decorates `__call__` — this method — so a `call`
        // that reached the provider hook instead would leave the one entry named after the decorated
        // method as the one entry with no point on it. It did, until
        // `asking_a_model_directly_fires_the_lm_point` said so.
        async move {
            let request = request?;
            self.forward_dyn(&request).await
        }
    }

    /// What this model states about itself in a saved program's `lm` block.
    ///
    /// dspy's `BaseLM.dump_state`, and defaulted for the same reason upstream's has a default: a
    /// model whose settings are not reconstructible from a dict — a scripted double, a caller's own
    /// wrapper — has nothing honest to say, and saying nothing is what dspy writes as `null`. [`LM`]
    /// overrides it; see [`saved::dump`](super::saved::dump).
    ///
    /// [`LM`]: super::LM
    fn dump_state(&self) -> Option<serde_json::Map<String, serde_json::Value>> {
        None
    }

    /// Watchers attached to this model rather than to the process — dspy's
    /// `dspy.LM("gpt-4o-mini", callbacks=[…])`, its second documented way to register one.
    ///
    /// They are told about this model's calls in addition to whatever
    /// [`configure_callbacks`](crate::configure_callbacks) registered, which is upstream's
    /// `settings.get("callbacks", []) + instance.callbacks`. Defaulted to none, so a provider
    /// implementing `forward` and nothing else carries no list — and read in
    /// [`DynChatModel::forward_dyn`], the one place a model's call cannot avoid.
    fn callbacks(&self) -> &[std::sync::Arc<dyn crate::Callback>] {
        &[]
    }

    /// What this model can be asked for natively. Nothing, unless the implementor says otherwise
    /// — the same default dspy's `BaseLM` carries, and for the same reason.
    ///
    /// Asynchronous because the honest answer is not always a lookup: an ollama server is asked
    /// what a model can do, exactly as litellm asks it, and that is a request like any other.
    fn capabilities(&self) -> impl Future<Output = Capabilities> + Send {
        std::future::ready(Capabilities::default())
    }

    /// Whether this model answers with citations on its own channel — dspy's
    /// `Citations.adapt_to_native_lm_feature`, whose test is `lm.model.startswith("anthropic/")`.
    ///
    /// A model that does takes the `Citations` output field *out* of the rendered prompt and fills
    /// it from the citations it attached to its own text blocks. Off by default, as it is for every
    /// provider but one upstream.
    fn native_citations_usable(&self) -> bool {
        false
    }

    /// Whether native reasoning is usable over this model's current path — dspy's model-specific
    /// caveat in `Reasoning.adapt_to_native_lm_feature`, kept where the model itself is known.
    ///
    /// On by default, since a model that reasons at all reasons over its own path. dspy turns it off
    /// for the gpt-5 family on the chat-completions route, where litellm 1.79.0 never returns the
    /// reasoning content (its issue #14748); the Responses API is unaffected. A model that reports
    /// `false` keeps the reasoning field rendered as prose instead of asking for it natively.
    fn native_reasoning_usable(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lm::dummy::DummyLM;
    use futures_util::StreamExt;

    /// A model that cannot stream still has a streaming form, and reports its answer whole.
    ///
    /// This is the point of defaulting it rather than requiring it: everything built above
    /// streaming needs *any* model to be watchable, and upstream tests its own listeners against
    /// scripted doubles for exactly that reason. A double with no streaming form could not be
    /// watched at all.
    #[tokio::test]
    async fn a_model_that_cannot_stream_reports_its_answer_whole() {
        let lm = DummyLM::new([crate::example! { answer: "Paris" }]);
        let request = api::LmRequest::new("dummy", vec![api::LmMessage::user(["where?"])]);
        let events: Vec<api::LmStreamEvent> = lm
            .forward_stream(&request)
            .map(|event| event.expect("an event"))
            .collect()
            .await;

        assert!(matches!(
            events.first(),
            Some(api::LmStreamEvent::Start { .. })
        ));
        assert!(matches!(
            events.last(),
            Some(api::LmStreamEvent::End { .. })
        ));
        let streamed: String = events
            .iter()
            .filter_map(|event| match event {
                api::LmStreamEvent::Delta {
                    delta: api::LmDelta::TextDelta { text },
                    ..
                } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert!(streamed.contains("Paris"), "got: {streamed}");
    }

    /// And it is reachable behind a pointer, which is how a listener holds it.
    #[tokio::test]
    async fn the_dyn_form_streams_the_same_thing() {
        let lm: std::sync::Arc<dyn DynChatModel> =
            std::sync::Arc::new(DummyLM::new([crate::example! { answer: "Berlin" }]));
        let request = api::LmRequest::new("dummy", vec![api::LmMessage::user(["where?"])]);
        let streamed: String = lm
            .forward_stream_dyn(&request)
            .filter_map(|event| {
                std::future::ready(match event {
                    Ok(api::LmStreamEvent::Delta {
                        delta: api::LmDelta::TextDelta { text },
                        ..
                    }) => Some(text),
                    _ => None,
                })
            })
            .collect()
            .await;
        assert!(streamed.contains("Berlin"), "got: {streamed}");
    }
}
