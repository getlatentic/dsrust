//! `DummyLM` driving real modules, which is what it exists for.
//!
//! Every test file here used to hand-roll a scripted model and write marker syntax by hand.
//! That tests the hand-written markers as much as the code. `DummyLM` takes field values and
//! renders them the way the adapter does, so a test says what the model *means*.

use std::sync::{Arc, Mutex};

use dsrust::lm::global;
use dsrust::signature::{FieldKind, OutField, Signature};
use dsrust::{DummyLM, Example, LabeledFewShot, Module, example};
use serde_json::Value;

/// The configured model is process-wide, so these tests take turns.
static GLOBAL_LM: Mutex<()> = Mutex::new(());

fn install(lm: Arc<DummyLM>) -> std::sync::MutexGuard<'static, ()> {
    let guard = GLOBAL_LM
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    global::configure_model(reqwest::Client::new(), lm);
    guard
}

fn signature() -> Signature {
    Signature::single_input(
        "Answer the question.",
        vec![OutField {
            name: "answer".into(),
            desc: "the answer".into(),
            ..Default::default()
        }],
    )
}

#[tokio::test]
async fn a_predict_answers_from_the_script() {
    let lm = Arc::new(DummyLM::new([example! { answer: "Paris" }]));
    let _guard = install(lm.clone());

    let predict = dsrust::predict::Predict::from_signature(signature());
    let prediction = predict
        .forward(example! { request: "capital of France?" }.with_inputs(["request"]))
        .await
        .expect("the scripted answer parses");

    assert_eq!(
        prediction.get("answer").and_then(Value::as_str),
        Some("Paris")
    );
    // The dummy kept the prompt, so a test can assert on what the model was actually shown.
    assert!(lm.asked()[0].last_message().contains("capital of France?"));
}

#[tokio::test]
async fn a_scripted_bool_survives_the_render_and_parse_round_trip() {
    // The dummy renders a field the way the adapter does, which spells a bool Python's way.
    // Whatever it writes, the parser on the other side has to read back.
    let bool_signature = Signature::single_input(
        "Decide.",
        vec![OutField {
            name: "sure".into(),
            desc: "sure about it".into(),
            kind: FieldKind::Bool,
            ..Default::default()
        }],
    );
    let lm = Arc::new(DummyLM::new([example! { sure: true }]));
    let _guard = install(lm.clone());

    let prediction = dsrust::predict::Predict::from_signature(bool_signature)
        .forward(example! { request: "is the sky blue?" }.with_inputs(["request"]))
        .await
        .expect("the scripted bool parses");

    assert_eq!(prediction.get("sure"), Some(&Value::Bool(true)));
}

#[tokio::test]
async fn a_compiled_program_shows_its_demos_to_the_model() {
    let lm = Arc::new(DummyLM::new([example! { answer: "Madrid" }]));
    let _guard = install(lm.clone());

    let trainset = vec![
        example! { request: "capital of France?", answer: "Paris" }.with_inputs(["request"]),
        example! { request: "capital of Germany?", answer: "Berlin" }.with_inputs(["request"]),
    ];
    let mut predict = dsrust::predict::Predict::from_signature(signature());
    LabeledFewShot::new(2).compile(&mut predict, &trainset);

    predict
        .forward(example! { request: "capital of Spain?" }.with_inputs(["request"]))
        .await
        .expect("the call succeeds");

    // Two demo pairs then the real ask: the compile reached the wire.
    assert_eq!(lm.asked()[0].turns.len(), 5);
}

#[tokio::test]
async fn a_keyed_dummy_suits_a_loop_whose_order_the_model_chooses() {
    let lm = Arc::new(DummyLM::keyed([
        ("France", example! { answer: "Paris" }),
        ("Spain", example! { answer: "Madrid" }),
    ]));
    let _guard = install(lm.clone());

    let predict = dsrust::predict::Predict::from_signature(signature());
    for (question, expected) in [
        ("capital of Spain?", "Madrid"),
        ("capital of France?", "Paris"),
    ] {
        let prediction = predict
            .forward(
                Example::new([("request", Value::String(question.into()))])
                    .with_inputs(["request"]),
            )
            .await
            .expect("keyed answer found");
        assert_eq!(
            prediction.get("answer").and_then(Value::as_str),
            Some(expected)
        );
    }
}

/// dspy `Prediction.from_completions`: a call asking for `n` answers holds all `n`, and its own
/// fields are the first — upstream's `{k: v[0] for k, v in completions.items()}`.
///
/// Measured against dspy 3.3.0b1 with the same script: `prediction.answer` is `red`,
/// `prediction.completions.answer` is `["red", "blue", "green"]`.
#[tokio::test]
async fn a_call_asking_for_several_answers_holds_all_of_them() {
    let lm = Arc::new(DummyLM::new([
        example! { answer: "red" },
        example! { answer: "blue" },
        example! { answer: "green" },
    ]));
    let _guard = install(lm.clone());

    let predict =
        dsrust::predict::Predict::from_signature(signature()).config(dsrust::lm::Sampling {
            completions: Some(3),
            ..Default::default()
        });
    let prediction = predict
        .forward(example! { request: "a colour?" }.with_inputs(["request"]))
        .await
        .expect("the scripted answers parse");

    // The prediction's own field is the first candidate, as upstream reads it.
    assert_eq!(
        prediction.get("answer").and_then(Value::as_str),
        Some("red")
    );

    let completions = prediction
        .completions
        .expect("three answers were asked for");
    assert_eq!(completions.len(), 3);
    let answers: Vec<&str> = completions
        .get("answer")
        .expect("the answer field")
        .iter()
        .filter_map(Value::as_str)
        .collect();
    assert_eq!(answers, ["red", "blue", "green"]);
}

/// A call asking for one answer carries no completions, which is every existing caller — upstream's
/// single-candidate `Prediction` reads the same way.
#[tokio::test]
async fn a_call_asking_for_one_answer_carries_no_completions() {
    let lm = Arc::new(DummyLM::new([example! { answer: "Paris" }]));
    let _guard = install(lm.clone());

    let prediction = dsrust::predict::Predict::from_signature(signature())
        .forward(example! { request: "capital of France?" }.with_inputs(["request"]))
        .await
        .expect("the scripted answer parses");

    assert!(prediction.completions.is_none());
}
