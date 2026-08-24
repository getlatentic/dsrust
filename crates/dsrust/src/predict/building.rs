//! Putting a `Predict` together: the constructors, and the seams an optimizer sets.
//!
//! dspy builds one by calling the class and mutating attributes afterwards. The same settings live
//! here as builder methods, and they are the ones an optimizer reaches for — `set_lm` and
//! `with_config` are what let `BestOfN` run the same module against calls that differ, and
//! `demos` is what a bootstrap writes its result into.

use std::marker::PhantomData;

use anyhow::Result;

use super::{Dynamic, Predict};
use crate::adapter::{Adapter, ChatAdapter};
use crate::example::Example;
use crate::lm::{DynChatModel, Sampling, global};
use crate::signature::{Signature, SignatureSpec};

impl<S> Predict<S> {
    /// The same module, told which task it asks. The signature and demos are unchanged; only
    /// what a call answers with follows from the type.
    pub(crate) fn into_task<T>(self) -> Predict<T> {
        Predict {
            callbacks: Vec::new(),
            spec: PhantomData,
            signature: self.signature,
            adapter: self.adapter,
            demos: self.demos,
            lm: self.lm,
            config: self.config,
            hint: self.hint,
            feedback_retry: self.feedback_retry,
        }
    }

    /// Re-ask with the error attached when a reply parses but does not validate.
    ///
    /// Off by default because dspy does not do it: upstream raises `AdapterParseError` and
    /// re-asks through `JSONAdapter`, so the second call carries the JSON adapter's prompt rather
    /// than a sentence naming the rejection. Turning this on trades that fidelity for a recovery
    /// that keeps the original wire format.
    pub fn feedback_retry(mut self) -> Self {
        self.feedback_retry = true;
        self
    }

    /// Ask this model rather than the configured one. dspy's `set_lm`.
    /// Watch *this* predictor — dspy's `dspy.Predict("q -> a", callbacks=[cb])`.
    ///
    /// The handlers here are told about this module's points on top of the process-wide ones, in
    /// upstream's order: configured first, then the instance's. Without it a handler can filter by
    /// the module's kind and by `CallId::parent`, but two predictors of the same signature look
    /// alike.
    pub fn callbacks(
        mut self,
        callbacks: impl IntoIterator<Item = std::sync::Arc<dyn crate::callback::Callback>>,
    ) -> Self {
        self.callbacks = callbacks.into_iter().collect();
        self
    }

    pub fn set_lm(mut self, lm: std::sync::Arc<dyn DynChatModel>) -> Self {
        self.lm = Some(lm);
        self
    }

    /// Ask for the reply to be sampled this way rather than at the provider's defaults.
    ///
    /// dspy reaches the same setting through `lm.copy(temperature=...)`, which needs a model to
    /// copy; this is per call, so a module that defers to the configured model can still vary
    /// how it is asked.
    pub fn config(mut self, config: Sampling) -> Self {
        self.config = config;
        self
    }

    /// dspy's `get_lm`: the model this module asks, or nothing if it defers to the configured
    /// one. An optimizer reads it to copy the settings it is about to vary.
    pub fn lm(&self) -> Option<&std::sync::Arc<dyn DynChatModel>> {
        self.lm.as_ref()
    }

    /// The model one call should ask: this module's own, or the configured default.
    ///
    /// A module carrying its own model runs where none is configured, which is what `BestOfN` and
    /// `Refine` rely on when they hand a predictor a scripted one.
    pub(crate) fn asking(&self) -> Result<std::sync::Arc<dyn DynChatModel>> {
        match &self.lm {
            Some(lm) => Ok(lm.clone()),
            None => global::current(),
        }
    }

    /// Show the model these solved examples before the request.
    pub fn demos(mut self, demos: impl IntoIterator<Item = Example>) -> Self {
        self.demos = demos.into_iter().collect();
        self
    }

    /// Send this module's prompts through a different wire format. Any [`Adapter`] works,
    /// including one a caller writes: dspy chooses its adapter the same way.
    pub fn adapter(mut self, adapter: impl Adapter + 'static) -> Self {
        self.adapter = Box::new(adapter);
        self
    }
}

impl Predict<Dynamic> {
    /// A module for a signature held as field names. `Predict!("question -> answer")` is the
    /// spelling to reach for; this is what it expands to.
    pub fn from_signature(signature: Signature) -> Self {
        Self {
            callbacks: Vec::new(),
            spec: PhantomData,
            lm: None,
            config: Sampling::default(),
            hint: None,
            feedback_retry: false,
            signature,
            adapter: Box::new(ChatAdapter::default()),
            demos: Vec::new(),
        }
    }

    /// dspy `Predict("email -> sentiment")`: declare the task by naming its fields.
    ///
    /// The shortest way to a working module. A field with no type is a string, which is what
    /// makes the untyped spelling useful for a first pass; `Predict::task` takes a derived
    /// signature when the types matter.
    pub fn parse(signature: &str) -> Result<Self> {
        Ok(Self::from_signature(signature.parse()?))
    }

    /// The module for a derived signature, which is `Predict::<Task>::new()` reached from the
    /// untyped name.
    pub fn task<S: SignatureSpec + Send + Sync>() -> Predict<S> {
        Predict::<S>::new()
    }
}
