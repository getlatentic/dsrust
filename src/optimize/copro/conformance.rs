//! COPRO against dspy's own, replaying an identical trace through both.
//!
//! Each case in `tests/conformance/optimize/copro.json` was produced by running dspy's COPRO
//! against a keyed `DummyLM` (see `scripts/generate_copro_fixture.py`). Here the crate's COPRO runs
//! against the same table, and every prompt it produces is compared to dspy's, in order, along with
//! the instruction it compiles to. A divergence is a bug in this crate until dspy is shown wrong —
//! it means the two optimizers made a different decision somewhere in the loop.

use std::sync::Arc;

use serde_json::Value;

use super::COPRO;
use crate::evaluate::exact_match;
use crate::example::Example;
use crate::predict::Predict;
use crate::signature::Signature;
use crate::DummyLM;
use crate::lm::dummy::Asked;

fn fixture() -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/conformance/optimize/copro.json");
    let text = std::fs::read_to_string(&path).expect("the copro golden is committed");
    serde_json::from_str(&text).expect("the golden parses")
}

/// An example built from a fixture object, declaring `question` as its one input where present so a
/// trainset row scores and a keyed answer does not.
fn example(object: &Value) -> Example {
    let fields = object
        .as_object()
        .expect("object")
        .iter()
        .map(|(name, value)| (name.clone(), value.clone()));
    let example = Example::new(fields);
    match object.get("question") {
        Some(_) => example.with_inputs(["question"]),
        None => example,
    }
}

/// The keyed model dspy was run against: each key answers whichever call carries it.
fn model(case: &Value) -> Arc<DummyLM> {
    let pairs = case["keyed"].as_array().expect("keyed").iter().map(|entry| {
        let key = entry["key"].as_str().expect("key").to_owned();
        (key, example(&entry["fields"]))
    });
    Arc::new(DummyLM::keyed(pairs))
}

fn student(case: &Value, model: Arc<DummyLM>) -> Predict {
    let mut signature: Signature = case["signature"].as_str().expect("signature").parse().expect("parses");
    signature.instructions = case["instruction"].as_str().expect("instruction").to_owned();
    Predict::from_signature(signature).with_lm(model)
}

/// Show the first difference rather than two walls of text.
fn assert_prompt(label: &str, expected: &str, actual: &str) {
    if expected == actual {
        return;
    }
    let at = expected
        .char_indices()
        .zip(actual.char_indices())
        .find(|((_, want), (_, got))| want != got)
        .map(|((index, _), _)| index)
        .unwrap_or_else(|| expected.len().min(actual.len()));
    panic!(
        "{label} diverges from dspy\n  first difference at byte {at}\n\n  expected: {:?}\n  actual:   {:?}\n",
        &expected[at.saturating_sub(40)..(at + 60).min(expected.len())],
        &actual[at.saturating_sub(40)..(at + 60).min(actual.len())],
    );
}

/// Every prompt the crate produced against the trace dspy recorded — a task call's system carries
/// the instruction in force, a depth call's user carries the attempts, so matching both in order is
/// matching the whole loop.
fn assert_calls(case: &Value, asked: &[Asked]) {
    let expected = case["calls"].as_array().expect("calls");
    assert_eq!(
        asked.len(),
        expected.len(),
        "case {:?} made {} calls, dspy made {}",
        case["instruction"],
        asked.len(),
        expected.len()
    );
    for (index, (got, want)) in asked.iter().zip(expected).enumerate() {
        assert_prompt(
            &format!("system of call {index}"),
            want["system"].as_str().expect("system"),
            &got.system,
        );
        assert_prompt(
            &format!("user of call {index}"),
            want["user"].as_str().expect("user"),
            got.last_message(),
        );
    }
}

#[tokio::test]
async fn copro_makes_the_decisions_dspy_makes() {
    let fixture = fixture();
    let cases = fixture["cases"].as_array().expect("cases");
    assert!(!cases.is_empty(), "the golden records no cases");
    for case in cases {
        let model = model(case);
        let trainset: Vec<Example> = case["trainset"].as_array().expect("trainset").iter().map(example).collect();
        let mut student = student(case, model.clone());

        COPRO::new(exact_match)
            .with_breadth(case["breadth"].as_u64().expect("breadth") as usize)
            .with_depth(case["depth"].as_u64().expect("depth") as usize)
            .with_prompt_model(model.clone())
            .compile(&mut student, &trainset)
            .await
            .expect("compiles");

        assert_calls(case, &model.asked());
        let compiled = case["final"][0].as_str().expect("a final instruction");
        assert_eq!(
            student.signature.instructions, compiled,
            "compiled instruction for case {:?}",
            case["instruction"]
        );
    }
}
