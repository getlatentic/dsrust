//! An optimizer compiling a real `Predict`, end to end.
//!
//! The unit tests use a hand-written student. This uses the module a caller would actually
//! reach for, and checks the thing that matters: after compiling, the demos an optimizer chose
//! appear in the prompt the model receives. If they do not, the compile changed nothing.

use dsrust::adapter::Input;
use std::collections::VecDeque;
use std::sync::Mutex;

use anyhow::Result;
use dsrust::lm::ChatModel;
use dsrust::lm::api::{self, LmMessage};
use dsrust::signature::{OutField, Signature};
use dsrust::{Adapter, ChatAdapter, Example, LabeledFewShot, example};
use serde_json::json;

struct Recorder {
    replies: Mutex<VecDeque<String>>,
    calls: Mutex<Vec<Vec<LmMessage>>>,
}

impl Recorder {
    fn new(replies: &[&str]) -> Self {
        Self {
            replies: Mutex::new(replies.iter().map(|r| (*r).to_owned()).collect()),
            calls: Mutex::new(Vec::new()),
        }
    }
}

impl ChatModel for Recorder {
    async fn forward(&self, request: &api::LmRequest) -> Result<api::LmResponse> {
        self.calls
            .lock()
            .expect("not poisoned")
            .push(request.messages.clone());
        self.replies
            .lock()
            .expect("not poisoned")
            .pop_front()
            .map(api::LmResponse::text)
            .ok_or_else(|| anyhow::anyhow!("script exhausted"))
    }
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

fn trainset() -> Vec<Example> {
    vec![
        example! { request: "capital of France?", answer: "Paris" }.with_inputs(["request"]),
        example! { request: "capital of Germany?", answer: "Berlin" }.with_inputs(["request"]),
    ]
}

#[test]
fn compiling_a_predict_writes_demos_through_the_module_seam() {
    let mut predict = dsrust::predict::Predict::from_signature(signature());
    assert!(predict.demos.is_empty());

    LabeledFewShot::new(2).compile(&mut predict, &trainset());

    // The optimizer reached the predictor through `named_predictors` and wrote demos back.
    assert_eq!(predict.demos.len(), 2);
}

#[tokio::test]
async fn the_chosen_demos_reach_the_prompt() {
    let mut predict = dsrust::predict::Predict::from_signature(signature());
    LabeledFewShot::new(2).compile(&mut predict, &trainset());

    let lm = Recorder::new(&["[[ ## answer ## ]]\nMadrid\n\n[[ ## completed ## ]]"]);
    predict
        .call_with(&lm, "capital of Spain?")
        .await
        .expect("the call succeeds");

    // The messages the model received; the system prompt leads, as dspy's `format` returns it.
    let turns = &lm.calls.lock().expect("not poisoned")[0][1..];
    // Two demos become two user/assistant pairs, then the real ask: compiling changed what
    // the model sees, which is the entire point of an optimizer.
    assert_eq!(turns.len(), 5);
    assert_eq!(turns[0].role, "user");
    assert_eq!(turns[1].role, "assistant");
    assert!(turns[1].text().unwrap().contains("[[ ## answer ## ]]"));
    assert!(turns[4].text().unwrap().contains("capital of Spain?"));
}

#[test]
fn a_compiled_program_and_an_uncompiled_one_render_differently() {
    let bare = dsrust::predict::Predict::from_signature(signature());
    let mut compiled = dsrust::predict::Predict::from_signature(signature());
    LabeledFewShot::new(2).compile(&mut compiled, &trainset());

    let inputs = [Input::new("request", json!("capital of Spain?"))];
    let bare_turns = &ChatAdapter::default()
        .format(&signature(), &bare.demos, &inputs)
        .expect("renders")[1..];
    let compiled_turns = &ChatAdapter::default()
        .format(&signature(), &compiled.demos, &inputs)
        .expect("renders")[1..];

    assert_eq!(bare_turns.len(), 1);
    assert_eq!(compiled_turns.len(), 5);
}
