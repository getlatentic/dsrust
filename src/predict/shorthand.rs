//! Asking a one-input signature with one string, and getting a struct back.
//!
//! dspy has no equivalent — Python's `**kwargs` makes `predict(question="…")` short enough already.
//! A Rust caller would otherwise build an `Example` for a signature with a single input, so these
//! name that input themselves and hand the reply back as JSON or, typed, as the caller's own struct.

use anyhow::Result;
use serde::de::DeserializeOwned;
use serde_json::Value;

use super::{Dynamic, Predict, Steering, typed};
use crate::adapter::Input;
use crate::lm::DynChatModel;

impl Predict<Dynamic> {
    /// Ask through the globally configured LM; see [`crate::lm::configure`].
    pub async fn call(&self, input: &str) -> Result<Value> {
        let (http, lm) = self.asking()?;
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
        let name = self
            .signature
            .inputs
            .first()
            .map_or("request", |f| f.name.as_str());
        Ok(self
            .call_with_inputs(
                http,
                lm,
                &[Input::new(name, Value::String(input.to_owned()))],
                &Steering::default(),
            )
            .await?
            .value)
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
