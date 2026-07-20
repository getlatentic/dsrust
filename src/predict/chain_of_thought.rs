use std::marker::PhantomData;

use anyhow::Result;
use serde::de::DeserializeOwned;
use serde_json::Value;

use super::{Predict, typed, typed_task};
use crate::example::{Example, Prediction};
use crate::lm::{DynChatModel, global};
use crate::module::{Module, NamedPredictor};
use crate::signature::{OutField, Signature, SignatureSpec};

/// dspy.ChainOfThought: the same signature with a leading `reasoning` field. The model puts
/// its thinking there; the caller receives only the signature's own fields.
pub struct ChainOfThought {
    predict: Predict,
}

impl ChainOfThought {
    pub fn new(mut signature: Signature) -> Self {
        signature.outputs.insert(
            0,
            OutField {
                name: "reasoning".into(),
                desc: "think step by step about the request before the other fields".into(),
                ..Default::default()
            },
        );
        Self {
            predict: Predict::new(signature),
        }
    }

    /// The module for a derived signature; its caller speaks the signature's own types.
    pub fn task<S: SignatureSpec>() -> TypedChainOfThought<S> {
        TypedChainOfThought {
            cot: ChainOfThought::new(S::signature()),
            spec: PhantomData,
        }
    }

    /// Ask through the globally configured LM; see [`crate::lm::configure`].
    pub async fn call(&self, input: &str) -> Result<Value> {
        let (http, lm) = global::current()?;
        self.call_with(&http, lm.as_ref(), input).await
    }

    /// Ask through an explicit client and model: the per-call override, and the seam tests
    /// script with a canned [`ChatModel`](crate::lm::ChatModel).
    pub async fn call_with(
        &self,
        http: &reqwest::Client,
        lm: &dyn DynChatModel,
        input: &str,
    ) -> Result<Value> {
        Ok(without_reasoning(
            self.predict.call_with(http, lm, input).await?,
        ))
    }

    /// The validated reply as a caller-owned struct instead of loose JSON.
    pub async fn call_typed<T: DeserializeOwned>(&self, input: &str) -> Result<T> {
        typed(self.call(input).await?)
    }

    /// [`Self::call_typed`] through an explicit client and model.
    pub async fn call_typed_with<T: DeserializeOwned>(
        &self,
        http: &reqwest::Client,
        lm: &dyn DynChatModel,
        input: &str,
    ) -> Result<T> {
        typed(self.call_with(http, lm, input).await?)
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
        typed_task::<S>(&self.cot.predict, http, lm, inputs, without_reasoning).await
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
        self.predict.forward(inputs)
    }

    fn named_predictors(&mut self) -> Vec<NamedPredictor<'_>> {
        self.predict.named_predictors()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::predict::scripted::{Pick, RoomTask, Scripted, room_inputs, signature};
    use serde_json::json;

    #[test]
    fn chain_of_thought_leads_with_reasoning_and_strips_it() {
        let cot = ChainOfThought::new(signature());
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
        let cot = ChainOfThought::new(signature());
        let value = cot
            .call_with(&reqwest::Client::new(), &lm, "draft it")
            .await
            .expect("valid reply");
        assert_eq!(value, json!({ "color": "red", "why": "calm" }));

        let lm = Scripted::new(&[reply]);
        let pick: Pick = cot
            .call_typed_with(&reqwest::Client::new(), &lm, "draft it")
            .await
            .expect("deserializes");
        assert_eq!(pick.color, "red");
    }

    #[tokio::test]
    async fn a_typed_chain_of_thought_strips_reasoning_before_deserializing() {
        let reply = "[[ ## reasoning ## ]]\nthinking\n\n[[ ## color ## ]]\nblue\n\n[[ ## why ## ]]\nfresh\n\n[[ ## completed ## ]]";
        let lm = Scripted::new(&[reply]);
        let outputs = RoomTask::chain_of_thought()
            .call_with(&reqwest::Client::new(), &lm, &room_inputs())
            .await
            .expect("valid reply");
        assert_eq!(outputs.color, "blue");
        assert_eq!(outputs.why, "fresh");
    }
}
