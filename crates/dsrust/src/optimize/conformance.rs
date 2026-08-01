//! The crate's optimizer held to dspy's own decisions.
//!
//! `teleprompt/test_bootstrap.py` is green and every one of its tests crosses into Rust — through
//! the *adapter*, while running dspy's Python optimizer. Nothing there reaches [`super::bootstrap`]
//! at all, and the crossing counter cannot say so: a crossing proves some Rust ran, not that the
//! Rust under test ran.
//!
//! This closes that. `scripts/generate_bootstrap_fixture.py` removes the model from upstream's
//! loop with a `DummyLM` answering from a fixed table, so a compile becomes a pure function of the
//! trainset, the metric and the configuration. Given the same three, the demos that survive here
//! must be the demos that survived there — same examples, same order, same count.

use std::collections::BTreeMap;

use serde_json::Value;

use super::BootstrapFewShot;
use super::scripted::{Answers, Pair, Solver, trainset};
use crate::evaluate::exact_match;
use crate::example::{Example, Prediction};

/// What dspy decided, recorded by running it. Regenerate with
/// `scripts/generate_bootstrap_fixture.py`.
fn golden() -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/conformance/optimize/bootstrap_few_shot.json");
    let text = std::fs::read_to_string(&path).expect("the bootstrap golden is committed");
    serde_json::from_str(&text).expect("the golden parses")
}

fn cases() -> Vec<Value> {
    golden()["cases"].as_array().expect("cases").clone()
}

fn field(value: &Value, name: &str) -> String {
    value[name].as_str().expect("a string field").to_owned()
}

/// A demo's question and answer, which together identify it among the trainset.
fn turn(demo: &Example) -> (String, String) {
    let read = |name: &str| {
        demo.get(name)
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned()
    };
    (read("question"), read("answer"))
}

fn expected_turns(case: &Value) -> Vec<(String, String)> {
    case["predictors"][0]["demos"]
        .as_array()
        .expect("demos")
        .iter()
        .map(|demo| (field(demo, "question"), field(demo, "answer")))
        .collect()
}

fn label(case: &Value) -> String {
    let read = |name: &str| case[name].as_u64().expect("a budget");
    format!(
        "max_bootstrapped_demos={}, max_labeled_demos={}, max_rounds={}, metric={}, threshold={}",
        read("max_bootstrapped_demos"),
        read("max_labeled_demos"),
        read("max_rounds"),
        field(case, "metric"),
        case["metric_threshold"]
    )
}

/// The fixture's `graded` metric: half credit for an answer that is wrong but present.
///
/// A score no threshold rejects and every threshold above it does. Without one it is read for
/// Python truth and succeeds; against a bar of 1.0 the same score fails. An exact-match metric
/// returns only 0.0 or 1.0, under which those two readings agree — so this is the metric that
/// tells them apart.
fn graded(example: &Example, prediction: &Prediction) -> f64 {
    if exact_match(example, prediction) == 1.0 {
        return 1.0;
    }
    match prediction.get("answer").and_then(Value::as_str) {
        Some(answer) if !answer.is_empty() => 0.5,
        _ => 0.0,
    }
}

/// Build the optimizer one case describes.
fn optimizer(case: &Value) -> BootstrapFewShot<fn(&Example, &Prediction) -> f64> {
    let budget = |name: &str| case[name].as_u64().expect("a budget") as usize;
    let metric: fn(&Example, &Prediction) -> f64 = match field(case, "metric").as_str() {
        "exact" => exact_match,
        "graded" => graded,
        other => panic!("the fixture names an unknown metric {other:?}"),
    };
    BootstrapFewShot {
        metric_threshold: case["metric_threshold"].as_f64(),
        max_bootstrapped_demos: budget("max_bootstrapped_demos"),
        max_labeled_demos: budget("max_labeled_demos"),
        max_rounds: budget("max_rounds"),
        ..BootstrapFewShot::new(metric)
    }
}

/// The comparison only means anything if both sides are given the same program to compile, and
/// the two descriptions live in different languages. Guard the pair rather than trusting them to
/// be edited together.
#[test]
fn the_fixture_describes_the_program_the_rust_side_runs() {
    let recorded: Vec<(String, String)> = golden()["trainset"]
        .as_array()
        .expect("a trainset")
        .iter()
        .map(|example| (field(example, "question"), field(example, "answer")))
        .collect();
    let ours: Vec<(String, String)> = trainset().iter().map(turn).collect();
    assert_eq!(
        ours, recorded,
        "the fixture's trainset has drifted from scripted.rs"
    );
}

/// Every budget upstream branches on, compared demo for demo.
///
/// The bootstrapped prefix is deterministic — the trainset walked in order, filtered by the
/// metric. The tail is drawn by `random.sample` over a shuffled validation set, so matching it at
/// all depends on [`super::rng`] reproducing CPython's generator: an approximation gives the right
/// count in the wrong order, which is the failure this whole comparison exists to detect.
#[tokio::test]
async fn keeps_the_demos_dspy_keeps() {
    for case in cases() {
        let mut student = Solver::new(Answers::Correctly);
        optimizer(&case)
            .compile(&mut student, &trainset())
            .await
            .expect("compile succeeds");

        let ours: Vec<(String, String)> = student.demos.iter().map(turn).collect();
        assert_eq!(ours, expected_turns(&case), "at {}", label(&case));
    }
}

/// Every field a demo carries. Which fields those are is the evidence once a program has more
/// than one predictor, because a demo the drafting half earned names a draft and one the
/// answering half earned names no question.
///
/// `augmented` is one of them, and used not to be: it was filtered out of dspy's side, so the
/// marker distinguishing an earned demo from a labelled one went uncompared for as long as the
/// crate did not set it.
fn fields(demo: &Example) -> BTreeMap<String, String> {
    demo.fields()
        .map(|(name, value)| (name.to_owned(), rendered(value)))
        .collect()
}

/// A field value as text, keeping a bool distinguishable from an absent string — `augmented` is a
/// bool on both sides, and mapping it through `as_str` would flatten it to the empty string and
/// compare equal to anything.
fn rendered(value: &Value) -> String {
    match value {
        Value::Bool(flag) => flag.to_string(),
        other => other.as_str().unwrap_or_default().to_owned(),
    }
}

fn expected_fields(demo: &Value) -> BTreeMap<String, String> {
    demo.as_object()
        .expect("a demo")
        .iter()
        .map(|(name, value)| (name.clone(), rendered(value)))
        .collect()
}

/// A pipeline's demos, per predictor, against dspy's.
///
/// This is what the single-predictor comparison cannot reach. Two decisions only exist once a
/// program has a second predictor: each is taught by its own calls, and `_train` rebinds
/// `raw_demos` to the sample it just drew, so the second draws from the first's sample rather
/// than from the validation set. Upstream's own answer shows the second's labelled tail in a
/// different order from the first's, which is the rebinding made visible.
#[tokio::test]
async fn a_pipeline_keeps_the_demos_dspy_keeps_for_each_predictor() {
    let recorded = golden();
    let cases = recorded["pair_cases"].as_array().expect("pair cases");
    assert!(!cases.is_empty(), "the fixture records no pipeline case");

    for case in cases {
        let mut student = Pair::new();
        optimizer(case)
            .compile(&mut student, &trainset())
            .await
            .expect("compile succeeds");

        let ours = [
            ("first", &student.first_demos),
            ("second", &student.second_demos),
        ];
        for (name, demos) in ours {
            let expected: Vec<BTreeMap<String, String>> = case["predictors"]
                .as_array()
                .expect("predictors")
                .iter()
                .find(|entry| field(entry, "predictor") == name)
                .unwrap_or_else(|| panic!("the fixture records no predictor {name:?}"))["demos"]
                .as_array()
                .expect("demos")
                .iter()
                .map(expected_fields)
                .collect();
            let actual: Vec<BTreeMap<String, String>> = demos.iter().map(fields).collect();
            assert_eq!(actual, expected, "predictor {name} at {}", label(case));
        }
    }
}

/// What each attempt was shown, which is where two decisions live that the compiled program
/// cannot report.
///
/// The teacher is primed with labelled demos before the walk, and the example being solved is
/// struck from that set for the length of its own call and put back afterwards. Both are undone
/// by the time a compile returns, so comparing only the result would pass with either one
/// missing — and did, until this landed.
#[tokio::test]
async fn shows_each_attempt_what_dspy_showed_it() {
    for case in cases() {
        let mut student = Solver::new(Answers::Correctly);
        optimizer(&case)
            .compile(&mut student, &trainset())
            .await
            .expect("compile succeeds");

        let ours: Vec<(String, Vec<String>)> = student
            .calls()
            .iter()
            .map(|call| {
                let demos = call.demos.iter().map(|demo| turn(demo).0).collect();
                (call.question.clone(), demos)
            })
            .collect();

        let expected: Vec<(String, Vec<String>)> = case["calls"]
            .as_array()
            .expect("calls")
            .iter()
            .map(|call| {
                let demos = call["demos"]
                    .as_array()
                    .expect("demos")
                    .iter()
                    .map(|question| question.as_str().expect("a question").to_owned())
                    .collect();
                (field(call, "question"), demos)
            })
            .collect();

        assert_eq!(ours, expected, "at {}", label(&case));
    }
}

/// A derived signature is compiled the same way a declared one is.
///
/// `TypedPredict` is the idiomatic way to write a task in Rust, and until it was a `Module` an
/// optimizer could not walk it: the ergonomic front door led away from the half of DSPy that
/// makes the other half worth having. Reaching `compile` at all is the assertion — it takes
/// `M: Module`, which a derived program could not satisfy.
#[tokio::test]
async fn a_derived_signature_can_be_compiled() {
    use std::sync::Arc;

    use crate::lm::global::install_for_test;
    use crate::module::Module;
    use crate::predict::Predict;
    use crate::signature::{InField, OutField, Signature, SignatureSpec};
    use crate::{DummyLM, example};

    struct Capital;

    impl SignatureSpec for Capital {
        type Inputs = ();
        type Outputs = serde_json::Map<String, Value>;

        fn signature() -> Signature {
            Signature {
                instructions: "Answer.".to_owned(),
                inputs: vec![InField {
                    name: "question".to_owned(),
                    ..Default::default()
                }],
                outputs: vec![OutField {
                    name: "answer".to_owned(),
                    ..Default::default()
                }],
            }
        }

        fn input_pairs(_: &Self::Inputs) -> Vec<crate::adapter::Input<'static>> {
            Vec::new()
        }
    }

    let _configured = install_for_test(Arc::new(DummyLM::keyed([
        ("France", example! { answer: "Paris" }),
        ("Germany", example! { answer: "Berlin" }),
    ])));

    let mut student = Predict::task::<Capital>();
    let solvable = vec![
        example! { question: "capital of France?", answer: "Paris" }.with_inputs(["question"]),
        example! { question: "capital of Germany?", answer: "Berlin" }.with_inputs(["question"]),
    ];

    let kept = BootstrapFewShot {
        max_labeled_demos: 0,
        ..BootstrapFewShot::new(exact_match)
    }
    .compile(&mut student, &solvable)
    .await
    .expect("a derived program compiles");

    assert_eq!(kept, 2, "both examples were solved and kept");
    let demos = student.named_predictors()[0].demos.len();
    assert_eq!(
        demos, 2,
        "the optimizer wrote demos into the derived program"
    );
}
