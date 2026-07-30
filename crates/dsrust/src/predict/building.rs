//! Putting a `Predict` together: the constructors, and the seams an optimizer sets.
//!
//! dspy builds one by calling the class and mutating attributes afterwards. The same settings live
//! here as builder methods, and they are the ones an optimizer reaches for — `with_lm` and
//! `with_config` are what let `BestOfN` run the same module against calls that differ, and
//! `with_demos` is what a bootstrap writes its result into.

use std::marker::PhantomData;

use anyhow::Result;

use super::{Dynamic, Predict};
use crate::adapter::{Adapter, ChatAdapter};
use crate::example::Example;
use crate::lm::{DynChatModel, LmConfig, global};
use crate::signature::{Signature, SignatureSpec};

impl<S> Predict<S> {
    /// The same module, told which task it asks. The signature and demos are unchanged; only
    /// what a call answers with follows from the type.
    pub(crate) fn into_task<T>(self) -> Predict<T> {
        Predict {
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
    pub fn with_feedback_retry(mut self) -> Self {
        self.feedback_retry = true;
        self
    }

    /// Ask this model rather than the configured one. dspy's `set_lm`.
    pub fn with_lm(mut self, lm: std::sync::Arc<dyn DynChatModel>) -> Self {
        self.lm = Some(lm);
        self
    }

    /// Ask for the reply to be sampled this way rather than at the provider's defaults.
    ///
    /// dspy reaches the same setting through `lm.copy(temperature=...)`, which needs a model to
    /// copy; this is per call, so a module that defers to the configured model can still vary
    /// how it is asked.
    pub fn with_config(mut self, config: LmConfig) -> Self {
        self.config = config;
        self
    }

    /// The config this module asks for. An optimizer reads it to vary one field and leave
    /// the rest alone.
    pub fn config(&self) -> &LmConfig {
        &self.config
    }

    /// dspy's `get_lm`: the model this module asks, or nothing if it defers to the configured
    /// one. An optimizer reads it to copy the settings it is about to vary.
    pub fn lm(&self) -> Option<&std::sync::Arc<dyn DynChatModel>> {
        self.lm.as_ref()
    }

    /// The model and client one call should use: this module's own, or the configured default.
    ///
    /// A module with its own model needs only a client, not the global model behind it, so it
    /// runs where none is configured — which is what `BestOfN` and `Refine` rely on when they
    /// hand a predictor a scripted model.
    pub(crate) fn asking(&self) -> Result<(reqwest::Client, std::sync::Arc<dyn DynChatModel>)> {
        match &self.lm {
            Some(lm) => Ok((global::client(), lm.clone())),
            None => global::current(),
        }
    }

    /// Show the model these solved examples before the request.
    pub fn with_demos(mut self, demos: impl IntoIterator<Item = Example>) -> Self {
        self.demos = demos.into_iter().collect();
        self
    }

    /// Send this module's prompts through a different wire format. Any [`Adapter`] works,
    /// including one a caller writes: dspy chooses its adapter the same way.
    pub fn with_adapter(mut self, adapter: impl Adapter + 'static) -> Self {
        self.adapter = Box::new(adapter);
        self
    }
}

impl Predict<Dynamic> {
    /// A module for a signature held as field names. `predict!("question -> answer")` is the
    /// spelling to reach for; this is what it expands to.
    pub fn from_signature(signature: Signature) -> Self {
        Self {
            spec: PhantomData,
            lm: None,
            config: LmConfig::default(),
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
