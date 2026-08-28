//! A double whose one predictor answers twice per example.
//!
//! Its own file because it is the only double that produces a *repeated* predictor, which is the
//! shape `collapse` exists for and the shape every other double here is built to avoid.

use std::pin::Pin;

use anyhow::Result;
use serde_json::{Value, json};

use super::answer;
use crate::example::{Example, Prediction};
use crate::lm::Sampling;
use crate::module::{Module, NamedPredictor};
use crate::signature::Signature;

/// One predictor called twice per example, which is the only shape that reaches
/// [`collapse`](super::earned::collapse).
///
/// Both calls take the same `question` and differ in `hop`, as a multi-hop program's do — so the
/// two demos share the question's value the way upstream's share the object, and the pickle memo
/// this is here to exercise has something to back-reference.
pub(crate) struct TwoHop {
    saved_lm_0: Option<std::sync::Arc<dyn crate::lm::DynChatModel>>,
    signature: Signature,
    config: Sampling,
    hint: Option<String>,
    pub(crate) demos: Vec<Example>,
}

impl TwoHop {
    pub(crate) fn new() -> Self {
        Self {
            saved_lm_0: None,
            signature: Signature::single_input("Answer.", Vec::new()),
            config: Sampling::default(),
            hint: None,
            demos: Vec::new(),
        }
    }

    fn hop(&self, question: &str, hop: &str) -> crate::module::TraceStep {
        crate::module::TraceStep {
            predictor: "gen".to_owned(),
            signature: self.signature.clone(),
            inputs: Example::new([("question", json!(question)), ("hop", json!(hop))])
                .with_inputs(["question", "hop"]),
            outputs: Example::new([("answer", json!(format!("{hop} answer")))]),
        }
    }
}

impl Module for TwoHop {
    fn forward<'a>(
        &'a self,
        inputs: Example,
    ) -> Pin<Box<dyn Future<Output = Result<Prediction>> + Send + 'a>> {
        Box::pin(async move {
            let question = inputs
                .get("question")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            Ok(Prediction::new(
                Example::new([("answer", json!(answer(&question, true)))]),
                "raw",
            ))
        })
    }

    fn forward_traced<'a>(
        &'a self,
        inputs: Example,
        trace: &'a mut Vec<crate::module::TraceStep>,
    ) -> Pin<Box<dyn Future<Output = Result<Prediction>> + Send + 'a>> {
        Box::pin(async move {
            let question = inputs
                .get("question")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            trace.push(self.hop(&question, "first"));
            trace.push(self.hop(&question, "second"));
            Ok(Prediction::new(
                Example::new([("answer", json!(answer(&question, true)))]),
                "raw",
            ))
        })
    }

    fn named_predictors(&mut self) -> Vec<NamedPredictor<'_>> {
        vec![NamedPredictor {
            name: "gen".to_owned(),
            signature: &mut self.signature,
            demos: &mut self.demos,
            config: &mut self.config,
            hint: &mut self.hint,
            lm: &mut self.saved_lm_0,
        }]
    }
}
