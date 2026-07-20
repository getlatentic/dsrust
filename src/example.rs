//! The data a program is measured against, and the data it produces.
//!
//! dspy's `Example` is a dict with one extra idea: some of its keys are inputs and the rest are
//! labels. That split is what lets an evaluator hand `example.inputs()` to a program and score
//! the result against `example.labels()`, and it is what an optimizer needs to bootstrap demos.
//! Python expresses it with attribute access on a dict; the same idea in Rust is a named type
//! with the split made explicit, so a caller cannot forget to declare it.

use std::collections::BTreeSet;

use serde_json::Value;

/// One labelled example: field values, plus which of those fields are inputs.
///
/// Field order is preserved, because prompts render fields in signature order and a stable
/// order keeps generated prompts reproducible.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Example {
    fields: Vec<(String, Value)>,
    input_keys: BTreeSet<String>,
}

impl Example {
    pub fn new(fields: impl IntoIterator<Item = (impl Into<String>, Value)>) -> Self {
        Self {
            fields: fields
                .into_iter()
                .map(|(name, value)| (name.into(), value))
                .collect(),
            input_keys: BTreeSet::new(),
        }
    }

    /// Declare which fields are inputs; everything else becomes a label. Mirrors
    /// `Example.with_inputs`, and is required before [`Self::inputs`] or [`Self::labels`]
    /// mean anything.
    pub fn with_inputs<S: Into<String>>(mut self, keys: impl IntoIterator<Item = S>) -> Self {
        self.input_keys = keys.into_iter().map(Into::into).collect();
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
        self.input_keys.contains(name)
    }

    /// Only the declared input fields, for handing to a program.
    pub fn inputs(&self) -> Example {
        self.subset(|name| self.input_keys.contains(name))
    }

    /// Everything not declared an input: the expected answer to score against.
    pub fn labels(&self) -> Example {
        self.subset(|name| !self.input_keys.contains(name))
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
            .map(|(name, value)| (name.clone(), render(value)))
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
/// assert_eq!(example.labels().len(), 1);
/// ```
#[macro_export]
macro_rules! example {
    ($($name:ident : $value:expr),* $(,)?) => {
        $crate::Example::new([
            $((stringify!($name), $crate::__macro_support::json!($value))),*
        ])
    };
}

/// dspy `format_field_value`: a string is itself, anything else is its JSON form. Quoting a
/// string would change what the model reads, and upstream is careful to avoid that.
pub fn render(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
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
}

impl Prediction {
    pub fn new(example: Example, raw: impl Into<String>) -> Self {
        Self {
            example,
            raw: raw.into(),
        }
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
        assert_eq!(example.inputs().len(), 1);
        assert_eq!(example.labels().len(), 1);
        assert_eq!(example.inputs().get("question").unwrap(), &json!("Why is the sky blue?"));
        assert_eq!(example.labels().get("answer").unwrap(), &json!("Rayleigh scattering."));
    }

    #[test]
    fn an_undeclared_example_has_no_inputs_and_labels_everything() {
        // dspy requires with_inputs before the split means anything; the same holds here, and
        // an evaluator that forgets it gets an empty input set rather than a wrong one.
        let example = qa();
        assert!(example.inputs().is_empty());
        assert_eq!(example.labels().len(), 2);
    }

    #[test]
    fn field_order_survives_every_operation() {
        let example = qa().with_inputs(["question"]);
        let names: Vec<&str> = example.fields().map(|(name, _)| name).collect();
        assert_eq!(names, ["question", "answer"]);
        let labels = example.labels();
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
                ("tags".to_owned(), r#"["a","b"]"#.to_owned()),
            ]
        );
    }

    #[test]
    fn a_prediction_keeps_the_reply_that_produced_it() {
        let prediction = Prediction::new(qa(), "[[ ## answer ## ]]\nRayleigh scattering.");
        assert_eq!(prediction.get("answer").unwrap(), &json!("Rayleigh scattering."));
        assert!(prediction.raw.contains("[[ ## answer ## ]]"));
    }
}
