//! The data a program is measured against, and the data it produces.
//!
//! dspy's `Example` is a dict with one extra idea: some of its keys are inputs and the rest are
//! labels. That split is what lets an evaluator hand `example.inputs()` to a program and score
//! the result against `example.labels()`, and it is what an optimizer needs to bootstrap demos.
//! Python expresses it with attribute access on a dict; the same idea in Rust is a named type
//! with the split made explicit, so a caller cannot forget to declare it.

use std::collections::BTreeSet;

use anyhow::{Result, anyhow};
use serde_json::Value;

use crate::lm::LmUsage;

use crate::adapter::python_json::format_value;

/// One labelled example: field values, plus which of those fields are inputs.
///
/// Field order is preserved, because prompts render fields in signature order and a stable
/// order keeps generated prompts reproducible.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Example {
    fields: Vec<(String, Value)>,
    /// `None` until [`Self::with_inputs`] declares the split. dspy raises rather than guess
    /// when this is unset, and so does this: an evaluator that silently scored against an
    /// empty input set would report a meaningless number instead of a mistake.
    input_keys: Option<BTreeSet<String>>,
}

impl Example {
    pub fn new(fields: impl IntoIterator<Item = (impl Into<String>, Value)>) -> Self {
        Self {
            fields: fields
                .into_iter()
                .map(|(name, value)| (name.into(), value))
                .collect(),
            input_keys: None,
        }
    }

    /// Declare which fields are inputs; everything else becomes a label. Mirrors
    /// `Example.with_inputs`, and is required before [`Self::inputs`] or [`Self::labels`]
    /// mean anything.
    pub fn with_inputs<S: Into<String>>(mut self, keys: impl IntoIterator<Item = S>) -> Self {
        self.input_keys = Some(keys.into_iter().map(Into::into).collect());
        self
    }

    pub fn get(&self, name: &str) -> Option<&Value> {
        self.fields
            .iter()
            .find(|(field, _)| field == name)
            .map(|(_, value)| value)
    }

    pub fn set(&mut self, name: impl Into<String>, value: Value) {
        let name = name.into();
        match self.fields.iter_mut().find(|(field, _)| *field == name) {
            Some((_, slot)) => *slot = value,
            None => self.fields.push((name, value)),
        }
    }

    pub fn fields(&self) -> impl Iterator<Item = (&str, &Value)> {
        self.fields
            .iter()
            .map(|(name, value)| (name.as_str(), value))
    }

    pub fn is_input(&self, name: &str) -> bool {
        self.input_keys
            .as_ref()
            .is_some_and(|keys| keys.contains(name))
    }

    /// Only the declared input fields, for handing to a program.
    ///
    /// Errors when the split was never declared, matching dspy's `ValueError`. Returning an
    /// empty set instead would let an evaluator score a program that received nothing.
    pub fn inputs(&self) -> Result<Example> {
        let keys = self.declared()?;
        Ok(self.subset(|name| keys.contains(name)))
    }

    /// Everything not declared an input: the expected answer to score against.
    pub fn labels(&self) -> Result<Example> {
        let keys = self.declared()?;
        Ok(self.subset(|name| !keys.contains(name)))
    }

    fn declared(&self) -> Result<&BTreeSet<String>> {
        self.input_keys.as_ref().ok_or_else(|| {
            anyhow!("inputs have not been set for this example; call with_inputs first")
        })
    }

    fn subset(&self, keep: impl Fn(&str) -> bool) -> Example {
        Example {
            fields: self
                .fields
                .iter()
                .filter(|(name, _)| keep(name))
                .cloned()
                .collect(),
            input_keys: self.input_keys.clone(),
        }
    }

    pub fn len(&self) -> usize {
        self.fields.len()
    }

    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// The field values as prompt-ready strings, in declaration order. A JSON string renders
    /// as its contents rather than a quoted literal, matching how dspy formats field values.
    pub fn rendered(&self) -> Vec<(String, String)> {
        self.fields
            .iter()
            .map(|(name, value)| (name.clone(), format_value(value)))
            .collect()
    }
}

/// Build an [`Example`] from named fields, the way `dspy.Example(question=..., answer=...)`
/// reads in Python. Values go through `serde_json::json!`, so a literal, a variable, or a
/// nested structure all work.
///
/// ```
/// let example = dsrs::example! { question: "Why is the sky blue?", answer: "Scattering." }
///     .with_inputs(["question"]);
/// assert_eq!(example.labels().unwrap().len(), 1);
/// ```
/// The inputs of one call, each field named where its value goes.
///
/// ```
/// # async fn wrapper(haiku: impl dsrs::Module) -> anyhow::Result<()> {
/// let out = haiku.forward(dsrs::input! { subject: "computer science" }).await?;
/// # Ok(()) }
/// ```
///
/// Every field is an input, which is what asking a module means and what separates this from
/// `example!`: a trainset row carries labels beside its inputs and has to say which are which,
/// while a call carries only what it is asking about.
#[macro_export]
macro_rules! input {
    ($($name:ident : $value:expr),* $(,)?) => {
        $crate::example! { $($name: $value),* }
            .with_inputs([$(stringify!($name)),*])
    };
}

#[macro_export]
macro_rules! example {
    ($($name:ident : $value:expr),* $(,)?) => {
        $crate::Example::new([
            $((stringify!($name), $crate::__macro_support::json!($value))),*
        ])
    };
}

/// What a module returns: the parsed output fields, plus what produced them.
///
/// dspy's `Prediction` is an `Example` carrying the raw completions alongside the parsed
/// values. Keeping the raw reply is not decoration — an evaluator reports it when a metric
/// fails, and an optimizer needs it to judge a trace.
#[derive(Debug, Clone, PartialEq)]
pub struct Prediction {
    pub example: Example,
    /// The model's reply exactly as it arrived, before parsing.
    pub raw: String,
    /// What every call behind this answer cost together, or nothing when no model reported it —
    /// which is what a scripted model reports, and what a provider omitting the block reports.
    ///
    /// dspy reaches the same number through `Prediction.get_lm_usage()`, filled only while
    /// `track_usage` is set. It is unconditional here because it arrives on the response either
    /// way, so there is no setting for it to be switched off by and no ambient state to read.
    pub usage: Option<LmUsage>,
}

impl Prediction {
    pub fn new(example: Example, raw: impl Into<String>) -> Self {
        Self {
            example,
            raw: raw.into(),
            usage: None,
        }
    }

    /// What the calls behind this answer cost.
    pub fn with_usage(mut self, usage: Option<LmUsage>) -> Self {
        self.usage = usage;
        self
    }

    pub fn get(&self, name: &str) -> Option<&Value> {
        self.example.get(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn qa() -> Example {
        Example::new([
            ("question", json!("Why is the sky blue?")),
            ("answer", json!("Rayleigh scattering.")),
        ])
    }

    #[test]
    fn with_inputs_splits_the_fields_into_inputs_and_labels() {
        let example = qa().with_inputs(["question"]);
        assert_eq!(example.inputs().unwrap().len(), 1);
        assert_eq!(example.labels().unwrap().len(), 1);
        assert_eq!(
            example.inputs().unwrap().get("question").unwrap(),
            &json!("Why is the sky blue?")
        );
        assert_eq!(
            example.labels().unwrap().get("answer").unwrap(),
            &json!("Rayleigh scattering.")
        );
    }

    #[test]
    fn an_undeclared_example_refuses_to_split() {
        // Checked against dspy 3.2.1, which raises ValueError here. Answering with an empty
        // input set would hand a program nothing and still score it.
        let example = qa();
        assert!(example.inputs().is_err());
        assert!(example.labels().is_err());
    }

    #[test]
    fn with_inputs_leaves_the_original_undeclared() {
        // dspy's with_inputs copies; mutating in place would surprise a caller reusing a
        // trainset example.
        let base = qa();
        let _marked = base.clone().with_inputs(["question"]);
        assert!(base.inputs().is_err());
    }

    #[test]
    fn marking_a_field_that_does_not_exist_is_allowed_and_yields_nothing() {
        // dspy tolerates this rather than validating against the field list.
        let example = qa().with_inputs(["missing"]);
        assert!(example.inputs().unwrap().is_empty());
        assert_eq!(example.labels().unwrap().len(), 2);
    }

    #[test]
    fn field_order_survives_every_operation() {
        let example = qa().with_inputs(["question"]);
        let names: Vec<&str> = example.fields().map(|(name, _)| name).collect();
        assert_eq!(names, ["question", "answer"]);
        let labels = example.labels().unwrap();
        let label_names: Vec<&str> = labels.fields().map(|(name, _)| name).collect();
        assert_eq!(label_names, ["answer"]);
    }

    #[test]
    fn setting_a_field_replaces_rather_than_duplicates() {
        let mut example = qa();
        example.set("answer", json!("Scattering."));
        assert_eq!(example.len(), 2);
        assert_eq!(example.get("answer").unwrap(), &json!("Scattering."));
    }

    #[test]
    fn rendering_leaves_strings_unquoted_and_json_intact() {
        let example = Example::new([
            ("note", json!("hello")),
            ("count", json!(3)),
            ("tags", json!(["a", "b"])),
        ]);
        assert_eq!(
            example.rendered(),
            vec![
                ("note".to_owned(), "hello".to_owned()),
                ("count".to_owned(), "3".to_owned()),
                ("tags".to_owned(), r#"["a", "b"]"#.to_owned()),
            ]
        );
    }

    #[test]
    fn a_prediction_keeps_the_reply_that_produced_it() {
        let prediction = Prediction::new(qa(), "[[ ## answer ## ]]\nRayleigh scattering.");
        assert_eq!(
            prediction.get("answer").unwrap(),
            &json!("Rayleigh scattering.")
        );
        assert!(prediction.raw.contains("[[ ## answer ## ]]"));
    }
}

#[cfg(test)]
mod input_macro {
    /// A call's fields are all inputs; a trainset row's are not, until it says so.
    #[test]
    fn every_field_of_a_call_is_an_input() {
        let asking = crate::input! { subject: "computer science", tone: "wry" };
        assert_eq!(asking.inputs().expect("declared").fields().count(), 2);
        assert!(asking.labels().expect("declared").fields().next().is_none());

        let row = crate::example! { subject: "computer science", haiku: "silicon dreaming" };
        assert!(
            row.inputs().is_err(),
            "a row has not said which fields are inputs"
        );
    }
}
