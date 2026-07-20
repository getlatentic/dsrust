//! `DummyLM` driving real modules, which is what it exists for.
//!
//! Every test file here used to hand-roll a scripted model and write marker syntax by hand.
//! That tests the hand-written markers as much as the code. `DummyLM` takes field values and
//! renders them the way the adapter does, so a test says what the model *means*.

use std::sync::{Arc, Mutex};

use dsrs::lm::global;
use dsrs::signature::{FieldKind, OutField, Signature};
use dsrs::{DummyLM, Example, LabeledFewShot, Module, example};
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
            kind: FieldKind::Str,
            values: None,
            schema: None,
        }],
    )
}

#[tokio::test]
async fn a_predict_answers_from_the_script() {
    let lm = Arc::new(DummyLM::new([example! { answer: "Paris" }]));
    let _guard = install(lm.clone());

    let predict = dsrs::predict::Predict::new(signature());
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
            values: None,
            schema: None,
        }],
    );
    let lm = Arc::new(DummyLM::new([example! { sure: true }]));
    let _guard = install(lm.clone());

    let prediction = dsrs::predict::Predict::new(bool_signature)
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
    let mut predict = dsrs::predict::Predict::new(signature());
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

    let predict = dsrs::predict::Predict::new(signature());
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
