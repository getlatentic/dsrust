//! What every model implements, in the two forms a Rust caller needs.
//!
//! dspy has one `BaseLM` because Python needs no second shape to store it behind a pointer.
//! `ChatModel` returns `impl Future`, which is what makes it pleasant to implement and impossible
//! to make into a trait object, so `DynChatModel` is the object-safe half and the blanket impl
//! below gives it to every model for free. The global configuration stores the second form, which
//! is what lets a test install a scripted model the way dspy installs a `DummyLM`.

use anyhow::Result;

use super::{Capabilities, api};

/// The object-safe form of [`ChatModel`], so a model can be stored behind a pointer.
///
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

    /// The object-safe form of [`ChatModel::capabilities`].
    fn capabilities_dyn(&self)
    -> std::pin::Pin<Box<dyn Future<Output = Capabilities> + Send + '_>>;

    /// The object-safe form of [`ChatModel::native_reasoning_usable`] — the `_dyn` name keeps it from
    /// clashing with the inherent one on a model that implements both.
    fn native_reasoning_usable_dyn(&self) -> bool;
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

    fn capabilities_dyn(
        &self,
    ) -> std::pin::Pin<Box<dyn Future<Output = Capabilities> + Send + '_>> {
        Box::pin(self.capabilities())
    }

    fn native_reasoning_usable_dyn(&self) -> bool {
        ChatModel::native_reasoning_usable(self)
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
        Self: Sync,
    {
        // The model name is the provider's own; every request this crate builds leaves it for
        // whichever wire answers, as an adapter-built one does.
        let request = api::LmRequest::from_items("", items);
        async move { self.forward(&request).await }
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
