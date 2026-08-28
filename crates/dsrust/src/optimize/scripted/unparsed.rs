//! A double whose predictor answers with text no adapter can read.
//!
//! The only shape that reaches `bootstrap_trace_data`'s failure arms, and both of them: `parsed`
//! empty is the constant arm that keeps the example, and `parsed` holding a declared field is the
//! graded arm that upstream drops.

use std::pin::Pin;

use anyhow::Result;
use serde_json::{Value, json};

use crate::adapter::parse::FieldMismatch;
use crate::example::{Example, Prediction};
use crate::lm::Sampling;
use crate::module::{Module, NamedPredictor, TraceStep};
use crate::signature::{InField, OutField, Signature};

pub(crate) struct Unparsed {
    saved_lm_0: Option<std::sync::Arc<dyn crate::lm::DynChatModel>>,
    signature: Signature,
    config: Sampling,
    hint: Option<String>,
    pub(crate) demos: Vec<Example>,
    /// What the adapter managed to read before giving up — dspy's `parsed_result`.
    parsed: Value,
    completion: String,
}

impl Unparsed {
    /// A reply the adapter could read nothing from, which is upstream's constant arm.
    pub(crate) fn reading_nothing(completion: &str) -> Self {
        Self::new(json!({}), completion)
    }

    /// A reply carrying one of the declared fields, which is the arm upstream drops.
    pub(crate) fn reading_one_field(completion: &str) -> Self {
        Self::new(json!({ "answer": "half" }), completion)
    }

    fn new(parsed: Value, completion: &str) -> Self {
        let signature = Signature {
            // dspy's default for `question -> answer, note`, so the structure instruction the
            // reflective record carries is the one upstream rendered.
            instructions: "Given the fields `question`, produce the fields `answer`, `note`."
                .to_owned(),
            inputs: vec![InField {
                name: "question".to_owned(),
                ..Default::default()
            }],
            outputs: ["answer", "note"]
                .map(|name| OutField {
                    name: name.to_owned(),
                    ..Default::default()
                })
                .to_vec(),
        };
        Self {
            saved_lm_0: None,
            signature,
            config: Sampling::default(),
            hint: None,
            demos: Vec::new(),
            parsed,
            completion: completion.to_owned(),
        }
    }

    fn refusal(&self) -> anyhow::Error {
        anyhow::Error::new(FieldMismatch {
            parsed: self.parsed.clone(),
            adapter_name: "JSONAdapter".to_owned(),
            lm_response: self.completion.clone(),
            expected_fields: self
                .signature
                .outputs
                .iter()
                .map(|f| f.name.clone())
                .collect(),
            message: None,
            reports_parsed: true,
            signature: self.signature.clone(),
        })
    }
}

impl Module for Unparsed {
    fn forward<'a>(
        &'a self,
        _inputs: Example,
    ) -> Pin<Box<dyn Future<Output = Result<Prediction>> + Send + 'a>> {
        Box::pin(async move { Err(self.refusal()) })
    }

    fn forward_traced<'a>(
        &'a self,
        _inputs: Example,
        _trace: &'a mut Vec<TraceStep>,
    ) -> Pin<Box<dyn Future<Output = Result<Prediction>> + Send + 'a>> {
        // Nothing is recorded before the refusal: the call that failed never produced a step, and
        // the failure step is appended by the *collector*, not by the program.
        Box::pin(async move { Err(self.refusal()) })
    }

    fn named_predictors(&mut self) -> Vec<NamedPredictor<'_>> {
        vec![NamedPredictor {
            name: "p".to_owned(),
            signature: &mut self.signature,
            demos: &mut self.demos,
            config: &mut self.config,
            hint: &mut self.hint,
            lm: &mut self.saved_lm_0,
        }]
    }
}
