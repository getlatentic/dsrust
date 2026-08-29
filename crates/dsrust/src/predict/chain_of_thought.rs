use std::marker::PhantomData;

use anyhow::Result;

use crate::adapter::Input;
use serde::de::DeserializeOwned;
use serde_json::Value;

use super::Predict;
use super::derived::{typed, typed_task};
use crate::example::{Example, Prediction};
use crate::lm::{DynChatModel, global};
use crate::module::{Ask, Module, NamedPredictor, TraceStep};
use crate::signature::{OutField, Signature, SignatureSpec};

/// dspy.ChainOfThought: the same signature with a leading `reasoning` field. The model puts
/// its thinking there; the caller receives only the signature's own fields.
pub struct ChainOfThought {
    predict: Predict,
}

impl ChainOfThought {
    /// The signature this module actually asks with, `reasoning` and all.
    ///
    /// dspy reaches the same thing as `cot.predict.signature`. What it answers with is not the
    /// signature handed in: this module prepends a field before asking.
    pub fn signature(&self) -> &Signature {
        &self.predict.signature
    }

    /// A module for a signature held as field names, matching
    /// [`Predict::from_signature`](super::Predict::from_signature).
    /// dspy `ChainOfThought("question -> answer")`: the task named by its fields.
    ///
    /// `ChainOfThought!` checks a spelling written in the source while the caller compiles;
    /// this is for a signature that is only a string at run time.
    ///
    /// `ChainOfThought!("question -> answer")` is the same thing checked while the caller compiles;
    /// this is the form for a signature that is only a string at run time.
    ///
    /// ```
    /// # fn wrapper() -> anyhow::Result<()> {
    /// let module = dsrust::ChainOfThought::parse("question -> answer")?;
    /// # let _ = module;
    /// # Ok(())
    /// # }
    /// ```
    pub fn parse(signature: &str) -> anyhow::Result<Self> {
        Ok(Self::from_signature(signature.parse()?))
    }

    pub fn from_signature(signature: Signature) -> Self {
        Self::rationale(signature, OutField::default())
    }

    /// dspy's `ChainOfThought(signature, rationale_field=…)`: reason through a field of the
    /// caller's own rather than the plain one.
    ///
    /// The *name* is always `reasoning` however the field is spelled, because upstream prepends
    /// under that name and only takes the description, kind and closed set from what it is given.
    /// A default field is the plain case, and carries no description: dspy sets the
    /// `${reasoning}` sentinel and suppresses it when rendering, so the field reaches a prompt as
    /// its name alone. The "Let's think step by step" prefix it once had was removed upstream in
    /// PR #8822, and prose here would be a line dspy never sends.
    pub fn rationale(mut signature: Signature, rationale: OutField) -> Self {
        signature.outputs.insert(
            0,
            OutField {
                name: "reasoning".into(),
                ..rationale
            },
        );
        Self {
            predict: Predict::<super::Dynamic>::from_signature(signature),
        }
    }

    /// Ask through this model rather than the globally configured one — the per-module override,
    /// and the seam a composed module uses to point its children at one model.
    pub fn set_lm(mut self, lm: std::sync::Arc<dyn crate::lm::DynChatModel>) -> Self {
        self.predict = self.predict.set_lm(lm);
        self
    }

    /// The module for a derived signature; its caller speaks the signature's own types.
    pub fn task<S: SignatureSpec>() -> TypedChainOfThought<S> {
        TypedChainOfThought {
            cot: ChainOfThought::from_signature(S::signature()),
            spec: PhantomData,
        }
    }

    /// Ask through the globally configured LM; see [`crate::lm::configure`].
    pub async fn call(&self, input: &str) -> Result<Value> {
        let lm = global::current()?;
        self.call_with(lm.as_ref(), input).await
    }

    /// Ask through an explicit client and model: the per-call override, and the seam tests
    /// script with a canned [`ChatModel`](crate::lm::ChatModel).
    pub async fn call_with(&self, lm: &dyn DynChatModel, input: &str) -> Result<Value> {
        Ok(without_reasoning(self.predict.call_with(lm, input).await?))
    }

    /// The validated reply as a caller-owned struct instead of loose JSON.
    pub async fn call_typed<T: DeserializeOwned>(&self, input: &str) -> Result<T> {
        typed(self.call(input).await?)
    }

    /// [`Self::call_typed`] through an explicit client and model.
    pub async fn call_typed_with<T: DeserializeOwned>(
        &self,
        lm: &dyn DynChatModel,
        input: &str,
    ) -> Result<T> {
        typed(self.call_with(lm, input).await?)
    }
}

/// A [`ChainOfThought`] bound to a derived signature, mirroring
/// [`TypedPredict`](super::TypedPredict).
pub struct TypedChainOfThought<S: SignatureSpec> {
    cot: ChainOfThought,
    spec: PhantomData<S>,
}

impl<S: SignatureSpec> TypedChainOfThought<S> {
    /// Ask through the globally configured LM; see [`crate::lm::configure`].
    pub async fn call_inputs(&self, inputs: &S::Inputs) -> Result<S::Outputs> {
        let lm = global::current()?;
        self.call_inputs_with(lm.as_ref(), inputs).await
    }

    /// Ask through an explicit client and model: the per-call override, and the seam tests
    /// script with a canned [`ChatModel`](crate::lm::ChatModel).
    pub async fn call_inputs_with(
        &self,
        lm: &dyn DynChatModel,
        inputs: &S::Inputs,
    ) -> Result<S::Outputs> {
        typed_task::<S, _>(&self.cot.predict, lm, inputs, without_reasoning).await
    }
}

fn without_reasoning(mut value: Value) -> Value {
    if let Some(map) = value.as_object_mut() {
        map.remove("reasoning");
    }
    value
}

/// A [`ChainOfThought`] is one predictor too: its reasoning field is part of the signature it
/// asks with, so an optimizer rewriting demos reaches the same place.
impl Module for ChainOfThought {
    fn forward<'a>(
        &'a self,
        inputs: Example,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<Prediction>> + Send + 'a>> {
        Box::pin(async move {
            let span = crate::observe::module_shown("ChainOfThought", &inputs, self.callbacks());
            crate::observe::watching(span, self.predict.forward(inputs)).await
        })
    }

    fn forward_traced<'a>(
        &'a self,
        inputs: Example,
        trace: &'a mut Vec<TraceStep>,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<Prediction>> + Send + 'a>> {
        self.predict.forward_traced(inputs, trace)
    }

    /// One predictor, named the way dspy names it.
    ///
    /// `Predict` calls its own `self`; upstream's `ChainOfThought` holds a `Predict` on an
    /// attribute called `predict` and `named_predictors` reports the attribute name. The name
    /// reaches a prompt through `Refine`'s module description, and keys demos during a compile,
    /// so the two have to agree.
    fn named_predictors(&mut self) -> Vec<NamedPredictor<'_>> {
        self.predict
            .named_predictors()
            .into_iter()
            .map(|predictor| NamedPredictor {
                name: "predict".to_owned(),
                ..predictor
            })
            .collect()
    }
}

/// The [`TypedPredict`](super::TypedPredict) reasoning: a derived signature stays reachable by
/// an optimizer whichever module drives it.
impl<S: SignatureSpec + Send + Sync> Module for TypedChainOfThought<S> {
    fn forward<'a>(
        &'a self,
        inputs: Example,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<Prediction>> + Send + 'a>> {
        self.cot.forward(inputs)
    }

    fn forward_traced<'a>(
        &'a self,
        inputs: Example,
        trace: &'a mut Vec<TraceStep>,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<Prediction>> + Send + 'a>> {
        self.cot.forward_traced(inputs, trace)
    }

    fn named_predictors(&mut self) -> Vec<NamedPredictor<'_>> {
        self.cot.named_predictors()
    }
}

#[cfg(test)]
mod tests {
    /// dspy's `rationale_field`: a caller reasons through a field of their own. Its description
    /// and kind are taken; its name is not, because upstream prepends under `reasoning` whatever
    /// it is handed.
    #[test]
    fn a_caller_can_supply_the_rationale_field() {
        let cot = ChainOfThought::rationale(
            "question -> answer".parse().expect("a signature"),
            OutField {
                name: "ignored".into(),
                desc: "work through it aloud".into(),
                ..Default::default()
            },
        );

        let reasoning = &cot.signature().outputs[0];
        assert_eq!(
            reasoning.name, "reasoning",
            "the name is upstream's, not the caller's"
        );
        assert_eq!(reasoning.desc, "work through it aloud");
    }

    /// dspy's second argument, `rationale_field_type`, travels as the field's kind — so the one
    /// constructor carries both. The ledger maps them to this method and the claim needs backing.
    #[test]
    fn the_rationale_carries_its_declared_type() {
        let cot = ChainOfThought::rationale(
            "question -> answer".parse().expect("a signature"),
            OutField {
                kind: crate::signature::FieldKind::Int,
                ..Default::default()
            },
        );
        assert_eq!(
            cot.signature().outputs[0].kind,
            crate::signature::FieldKind::Int
        );
    }

    /// And the plain case says nothing, which is what dspy's suppressed sentinel amounts to.
    #[test]
    fn the_plain_rationale_carries_no_description() {
        let cot =
            ChainOfThought::from_signature("question -> answer".parse().expect("a signature"));
        assert_eq!(cot.signature().outputs[0].desc, "");
    }

    use super::*;
    use crate::predict::scripted::{Pick, RoomTask, Scripted, room_inputs, signature};
    use serde_json::json;

    #[test]
    fn chain_of_thought_leads_with_reasoning_and_strips_it() {
        let cot = ChainOfThought::from_signature(signature());
        let sig = &cot.predict.signature;
        assert_eq!(sig.outputs[0].name, "reasoning");
        assert_eq!(sig.schema()["required"][0], json!("reasoning"));

        let value = json!({ "reasoning": "…", "color": "red", "why": "calm" });
        assert_eq!(
            without_reasoning(value),
            json!({ "color": "red", "why": "calm" })
        );
    }

    #[tokio::test]
    async fn chain_of_thought_strips_reasoning_from_the_typed_path() {
        let reply = "[[ ## reasoning ## ]]\nthinking\n\n[[ ## color ## ]]\nred\n\n[[ ## why ## ]]\ncalm\n\n[[ ## completed ## ]]";
        let lm = Scripted::new(&[reply]);
        let cot = ChainOfThought::from_signature(signature());
        let value = cot.call_with(&lm, "draft it").await.expect("valid reply");
        assert_eq!(value, json!({ "color": "red", "why": "calm" }));

        let lm = Scripted::new(&[reply]);
        let pick: Pick = cot
            .call_typed_with(&lm, "draft it")
            .await
            .expect("deserializes");
        assert_eq!(pick.color, "red");
    }

    #[tokio::test]
    async fn a_typed_chain_of_thought_strips_reasoning_before_deserializing() {
        let reply = "[[ ## reasoning ## ]]\nthinking\n\n[[ ## color ## ]]\nblue\n\n[[ ## why ## ]]\nfresh\n\n[[ ## completed ## ]]";
        let lm = Scripted::new(&[reply]);
        let outputs = RoomTask::chain_of_thought()
            .call_inputs_with(&lm, &room_inputs())
            .await
            .expect("valid reply");
        assert_eq!(outputs.color, "blue");
        assert_eq!(outputs.why, "fresh");
    }
}

crate::asks_with_a_prediction!(ChainOfThought);

/// The [`TypedPredict`](super::TypedPredict) answer: a derived task keeps its own outputs.
impl<S: SignatureSpec + Send + Sync> Ask for TypedChainOfThought<S>
where
    S::Outputs: Send,
{
    type Answer = S::Outputs;

    fn ask<'a>(
        &'a self,
        inputs: Example,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<S::Outputs>> + Send + 'a>> {
        Box::pin(async move {
            let lm = global::current()?;
            let pairs: Vec<Input<'_>> = inputs
                .fields()
                .map(|(name, value)| Input::new(name, value.clone()))
                .collect();
            super::derived::typed_pairs::<S, _>(
                &self.cot.predict,
                lm.as_ref(),
                pairs,
                without_reasoning,
            )
            .await
        })
    }
}
