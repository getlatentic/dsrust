//! A derived signature's typed call path: the task's inputs struct in, its outputs struct back.
//!
//! The untyped modules answer with a `Prediction`, which is every field the model returned and
//! nothing about what they mean. A derived task knows both, so it answers with its own outputs
//! and `result.answer` keeps meaning the field rather than a lookup that might miss.

use anyhow::Result;
use serde::de::DeserializeOwned;
use serde_json::Value;

use super::{Dynamic, Feedback, Predict};
use crate::adapter::Input;
use crate::error::Explained;
use crate::example::{Example, Prediction};
use crate::lm::{DynChatModel, LmUsage};
use crate::module::Ask;
use crate::signature::SignatureSpec;

/// A [`Predict`] bound to a derived signature: the inputs struct in, the outputs struct back,
/// through the same adapter-fallback and feedback-retry path.
/// The same [`Predict`], told which task it is asking. dspy has one `Predict` class for both
/// ways of declaring a signature, and so does this: the type parameter is what a call's answer
/// follows from, not a second module.
pub type TypedPredict<S> = Predict<S>;

#[allow(clippy::new_without_default)] // dspy has `Predict(QA)` and no `Default`; so does this
impl<S: SignatureSpec + Send + Sync> Predict<S> {
    /// The module for this task, which is dspy's `Predict(QA)`.
    pub fn new() -> Self {
        Predict::<Dynamic>::from_signature(S::signature()).into_task()
    }

    /// Ask through the globally configured LM; see [`crate::lm::configure`].
    pub async fn call_inputs(&self, inputs: &S::Inputs) -> Result<S::Outputs> {
        let lm = self.asking()?;
        self.call_inputs_with(lm.as_ref(), inputs).await
    }

    /// Ask through an explicit client and model: the per-call override, and the seam tests
    /// script with a canned [`ChatModel`](crate::lm::ChatModel).
    pub async fn call_inputs_with(
        &self,
        lm: &dyn DynChatModel,
        inputs: &S::Inputs,
    ) -> Result<S::Outputs> {
        typed_task::<S, _>(self, lm, inputs, std::convert::identity).await
    }
}

/// The typed tail shared by [`TypedPredict`] and [`TypedChainOfThought`]: deserialize the
/// validated reply into the task's outputs, giving a shape mismatch one feedback retry that
/// carries the serde error — the typed paths' fourth possible provider call. A second
/// failure of any kind is final. `shape` trims module-owned fields (chain-of-thought's
/// `reasoning`) before deserializing.
pub(crate) async fn typed_task<S: SignatureSpec, P>(
    predict: &Predict<P>,
    lm: &dyn DynChatModel,
    inputs: &S::Inputs,
    shape: fn(Value) -> Value,
) -> Result<S::Outputs> {
    typed_pairs::<S, _>(predict, lm, S::input_pairs(inputs), shape).await
}

/// [`typed_task`] reached with the fields already named, which is what an `Example` carries.
pub(crate) async fn typed_pairs<S: SignatureSpec, P>(
    predict: &Predict<P>,
    lm: &dyn DynChatModel,
    pairs: Vec<Input<'_>>,
    shape: fn(Value) -> Value,
) -> Result<S::Outputs> {
    typed_prediction::<S, _>(predict, lm, pairs, shape)
        .await
        .and_then(|answered| answered.typed::<S::Outputs>())
}

/// The same reply, as the `Prediction` the module point closes with.
///
/// A typed call answers with the caller's own struct, and `on_module_end` takes a `Prediction` —
/// so a typed path that converted here and threw the rest away could not be watched at all, and
/// two spellings of `Predict` were invisible to every callback until this split them.
pub(crate) async fn typed_prediction<S: SignatureSpec, P>(
    predict: &Predict<P>,
    lm: &dyn DynChatModel,
    pairs: Vec<Input<'_>>,
    shape: fn(Value) -> Value,
) -> Result<Prediction> {
    let validated = predict
        .call_with_inputs(lm, &pairs, &super::Steering::default())
        .await?;
    let shaped = shape(validated.value);
    let error = match typed::<S::Outputs>(shaped.clone()) {
        Ok(_) => {
            return answered(shaped, validated.raw, validated.siblings, validated.usage);
        }
        Err(error) => error,
    };
    tracing::warn!(%error, "retrying with shape feedback");
    let feedback = Feedback {
        previous: validated.raw,
        error: format!("{error:#}"),
    };
    let (raw, value, usage) = predict
        .feedback_ask(lm, &pairs, &feedback, &super::Steering::default())
        .await?;
    let shaped = shape(value);
    // Checked here rather than left to the caller, so a second bad shape carries the same
    // explained error it always did — and stops, rather than asking a third time.
    typed::<S::Outputs>(shaped.clone())?;
    answered(shaped, raw, Vec::new(), usage)
}

/// The same shape [`Predict::forward_with_steering`] answers with, so a typed call and a value-level
/// one close their point with the same thing.
fn answered(
    value: Value,
    raw: String,
    siblings: Vec<Example>,
    usage: Option<LmUsage>,
) -> Result<Prediction> {
    let mut candidates = vec![super::prediction_example(&value)];
    candidates.extend(siblings);
    Ok(Prediction::from_candidates(&candidates, raw)?.set_lm_usage(usage))
}

pub(crate) fn typed<T: DeserializeOwned>(value: Value) -> Result<T> {
    serde_json::from_value(value).explain("validated reply did not fit the requested type")
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
            let lm = self.asking()?;
            // The point, as the value-level path opens it: dspy decorates `Module.__call__` for
            // every module, so a typed spelling that skipped this was a module no callback saw.
            let span =
                crate::observe::module_shown("Predict", &inputs, crate::Module::callbacks(self));
            let pairs: Vec<Input<'_>> = inputs
                .fields()
                .map(|(name, value)| Input::new(name, value.clone()))
                .collect();
            crate::observe::watching(
                span,
                typed_prediction::<S, _>(self, lm.as_ref(), pairs, std::convert::identity),
            )
            .await?
            .typed::<S::Outputs>()
        })
    }
}
