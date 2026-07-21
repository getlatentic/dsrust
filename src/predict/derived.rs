//! A derived signature's typed call path: the task's inputs struct in, its outputs struct back.
//!
//! The untyped modules answer with a `Prediction`, which is every field the model returned and
//! nothing about what they mean. A derived task knows both, so it answers with its own outputs
//! and `result.answer` keeps meaning the field rather than a lookup that might miss.

use std::marker::PhantomData;

use anyhow::{Context, Result};
use serde::de::DeserializeOwned;
use serde_json::Value;

use super::{Feedback, Predict, Validated};
use crate::adapter::Adapter;
use crate::example::{Example, Prediction};
use crate::lm::{DynChatModel, global};
use crate::module::{Ask, Module, NamedPredictor, TraceStep};
use crate::signature::SignatureSpec;

/// A [`Predict`] bound to a derived signature: the inputs struct in, the outputs struct back,
/// through the same adapter-fallback and feedback-retry path.
pub struct TypedPredict<S: SignatureSpec> {
    pub(super) predict: Predict,
    pub(super) spec: PhantomData<S>,
}

impl<S: SignatureSpec> TypedPredict<S> {
    /// Send this module's prompts through a different wire format; see
    /// [`Predict::with_adapter`].
    pub fn with_adapter(mut self, adapter: impl Adapter + 'static) -> Self {
        self.predict = self.predict.with_adapter(adapter);
        self
    }

    /// Ask through the globally configured LM; see [`crate::lm::configure`].
    pub async fn call(&self, inputs: &S::Inputs) -> Result<S::Outputs> {
        let (http, lm) = global::current()?;
        self.call_with(&http, lm.as_ref(), inputs).await
    }

    /// Ask through an explicit client and model: the per-call override, and the seam tests
    /// script with a canned [`ChatModel`](crate::lm::ChatModel).
    pub async fn call_with(
        &self,
        http: &reqwest::Client,
        lm: &dyn DynChatModel,
        inputs: &S::Inputs,
    ) -> Result<S::Outputs> {
        typed_task::<S>(&self.predict, http, lm, inputs, std::convert::identity).await
    }
}

/// The typed tail shared by [`TypedPredict`] and [`TypedChainOfThought`]: deserialize the
/// validated reply into the task's outputs, giving a shape mismatch one feedback retry that
/// carries the serde error — the typed paths' fourth possible provider call. A second
/// failure of any kind is final. `shape` trims module-owned fields (chain-of-thought's
/// `reasoning`) before deserializing.
pub(crate) async fn typed_task<S: SignatureSpec>(
    predict: &Predict,
    http: &reqwest::Client,
    lm: &dyn DynChatModel,
    inputs: &S::Inputs,
    shape: fn(Value) -> Value,
) -> Result<S::Outputs> {
    typed_pairs::<S>(predict, http, lm, S::input_pairs(inputs), shape).await
}

/// [`typed_task`] reached with the fields already named, which is what an `Example` carries.
pub(crate) async fn typed_pairs<S: SignatureSpec>(
    predict: &Predict,
    http: &reqwest::Client,
    lm: &dyn DynChatModel,
    pairs: Vec<(&str, Value)>,
    shape: fn(Value) -> Value,
) -> Result<S::Outputs> {
    let Validated { raw, value } = predict.call_with_inputs(http, lm, &pairs).await?;
    let error = match typed::<S::Outputs>(shape(value)) {
        Ok(outputs) => return Ok(outputs),
        Err(error) => error,
    };
    tracing::warn!(%error, "retrying with shape feedback");
    let feedback = Feedback {
        previous: raw,
        error: format!("{error:#}"),
    };
    let (_, value) = predict.feedback_ask(http, lm, &pairs, &feedback).await?;
    typed(shape(value))
}

pub(crate) fn typed<T: DeserializeOwned>(value: Value) -> Result<T> {
    serde_json::from_value(value).context("validated reply did not fit the requested type")
}

/// A derived task answers with its own outputs, so `result.answer` keeps meaning the field.
impl<S: SignatureSpec + Send + Sync> Ask for TypedPredict<S>
where
    S::Outputs: Send,
{
    type Answer = S::Outputs;

    fn ask<'a>(
        &'a self,
        inputs: Example,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<S::Outputs>> + Send + 'a>> {
        Box::pin(async move {
            let (http, lm) = global::current()?;
            let pairs: Vec<(&str, Value)> = inputs
                .fields()
                .map(|(name, value)| (name, value.clone()))
                .collect();
            typed_pairs::<S>(
                &self.predict,
                &http,
                lm.as_ref(),
                pairs,
                std::convert::identity,
            )
            .await
        })
    }
}

/// A typed module is a module: same walk, same trace, so an optimizer reaches a derived
/// signature exactly as it reaches a declared one.
///
/// Without this a program written the idiomatic way — a struct and a derive — could be asked
/// but never compiled, which is the half of DSPy that makes the other half worth having.
impl<S: SignatureSpec + Send + Sync> Module for TypedPredict<S> {
    fn forward<'a>(
        &'a self,
        inputs: Example,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<Prediction>> + Send + 'a>> {
        self.predict.forward(inputs)
    }

    fn forward_traced<'a>(
        &'a self,
        inputs: Example,
        trace: &'a mut Vec<TraceStep>,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<Prediction>> + Send + 'a>> {
        self.predict.forward_traced(inputs, trace)
    }

    fn named_predictors(&mut self) -> Vec<NamedPredictor<'_>> {
        self.predict.named_predictors()
    }
}
