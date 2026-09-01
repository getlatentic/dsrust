//! What each predictor in a composed program is called, against what dspy calls it.
//!
//! The name is not decoration: `load_state` indexes by it and a GEPA candidate is keyed by it, so
//! two programs of the same shape that disagree about it cannot exchange a saved state in either
//! direction.
//!
//! `#[derive(Module)]` used to *replace* a child's name with the field holding it. That agrees for
//! a leaf `Predict`, which calls itself `self`, and differs the moment a step is itself composed:
//! a `ChainOfThought` in `answer_generator` was `answer_generator` here and
//! `answer_generator.predict` upstream. Nothing caught it because no test held a derived module
//! containing a composed one — the shape is the whole difference.

use dsrust::{ChainOfThought, Example, Module, Predict, Prediction};
use serde_json::Value;

/// A step that is itself composed, so its own predictor has a name to be prefixed.
#[derive(dsrust::Module)]
struct Inner {
    step: ChainOfThought,
}

impl dsrust::Forward for Inner {
    async fn forward(&self, inputs: Example) -> anyhow::Result<Prediction> {
        self.step.forward(inputs).await
    }
}

#[derive(dsrust::Module)]
struct Composed {
    flat: Predict,
    cot: ChainOfThought,
    nested: Inner,
}

impl dsrust::Forward for Composed {
    async fn forward(&self, inputs: Example) -> anyhow::Result<Prediction> {
        self.flat.forward(inputs).await
    }
}

fn golden() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/conformance/module/predictor_names.json"
    );
    let text = std::fs::read_to_string(path).expect("the golden is committed");
    serde_json::from_str(&text).expect("it parses")
}

fn recorded(what: &str) -> Vec<String> {
    golden()["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .find(|case| case["what"].as_str() == Some(what))
        .unwrap_or_else(|| panic!("no case for {what}"))["names"]
        .as_array()
        .expect("names")
        .iter()
        .map(|name| name.as_str().expect("a name").to_owned())
        .collect()
}

fn names(program: &mut dyn Module) -> Vec<String> {
    program
        .named_predictors()
        .into_iter()
        .map(|predictor| predictor.name)
        .collect()
}

#[test]
fn a_bare_predictor_is_named_as_dspy_names_it() {
    let mut flat = Predict::parse("a -> b").expect("parses");
    assert_eq!(names(&mut flat), recorded("a bare Predict"));

    let mut cot = ChainOfThought!("a -> b");
    assert_eq!(names(&mut cot), recorded("a bare ChainOfThought"));
}

/// The case the old rule got wrong: a field holding something with a name of its own.
#[test]
fn a_composed_program_is_named_as_dspy_names_it() {
    let mut program = Composed {
        flat: Predict::parse("question -> query").expect("parses"),
        cot: ChainOfThought!("question, context -> answer"),
        nested: Inner {
            step: ChainOfThought!("a -> b"),
        },
    };
    assert_eq!(names(&mut program), recorded("a composed module"));
}
