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
        http: &'a reqwest::Client,
        request: &'a api::LmRequest,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<api::LmResponse>> + Send + 'a>>;

    /// The object-safe form of [`ChatModel::capabilities`].
    fn capabilities_dyn<'a>(
        &'a self,
        http: &'a reqwest::Client,
    ) -> std::pin::Pin<Box<dyn Future<Output = Capabilities> + Send + 'a>>;

    /// The object-safe form of [`ChatModel::native_reasoning_usable`] — the `_dyn` name keeps it from
    /// clashing with the inherent one on a model that implements both.
    fn native_reasoning_usable_dyn(&self) -> bool;
}

impl<T: ChatModel + Send + Sync> DynChatModel for T {
    fn forward_dyn<'a>(
        &'a self,
        http: &'a reqwest::Client,
        request: &'a api::LmRequest,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<api::LmResponse>> + Send + 'a>> {
        Box::pin(self.forward(http, request))
    }

    fn capabilities_dyn<'a>(
        &'a self,
        http: &'a reqwest::Client,
    ) -> std::pin::Pin<Box<dyn Future<Output = Capabilities> + Send + 'a>> {
        Box::pin(self.capabilities(http))
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
        http: &'a reqwest::Client,
        request: &'a api::LmRequest,
    ) -> impl Future<Output = Result<api::LmResponse>> + Send + 'a;

    /// What this model can be asked for natively. Nothing, unless the implementor says otherwise
    /// — the same default dspy's `BaseLM` carries, and for the same reason.
    ///
    /// Asynchronous because the honest answer is not always a lookup: an ollama server is asked
    /// what a model can do, exactly as litellm asks it, and that is a request like any other.
    fn capabilities<'a>(
        &'a self,
        _http: &'a reqwest::Client,
    ) -> impl Future<Output = Capabilities> + Send + 'a {
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
