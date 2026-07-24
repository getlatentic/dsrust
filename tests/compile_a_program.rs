//! An optimizer compiling a real `Predict`, end to end.
//!
//! The unit tests use a hand-written student. This uses the module a caller would actually
//! reach for, and checks the thing that matters: after compiling, the demos an optimizer chose
//! appear in the prompt the model receives. If they do not, the compile changed nothing.

use dsrust::adapter::Input;
use std::collections::VecDeque;
use std::sync::Mutex;

use anyhow::Result;
use dsrust::lm::api::{self, Content, content_of};
use dsrust::lm::{ChatModel, ChatTurn, Role};
use dsrust::signature::{OutField, Signature};
use dsrust::{Adapter, ChatAdapter, Example, LabeledFewShot, example};
use serde_json::json;

struct Recorder {
    replies: Mutex<VecDeque<String>>,
    turns: Mutex<Vec<Vec<ChatTurn>>>,
}

impl Recorder {
    fn new(replies: &[&str]) -> Self {
        Self {
            replies: Mutex::new(replies.iter().map(|r| (*r).to_owned()).collect()),
            turns: Mutex::new(Vec::new()),
        }
    }
}

impl ChatModel for Recorder {
    async fn forward(
        &self,
        _http: &reqwest::Client,
        request: &api::LmRequest,
    ) -> Result<api::LmResponse> {
        self.turns
            .lock()
            .expect("not poisoned")
            .push(recorded_turns(request));
        self.replies
            .lock()
            .expect("not poisoned")
            .pop_front()
            .map(api::LmResponse::text)
            .ok_or_else(|| anyhow::anyhow!("script exhausted"))
    }
}

/// The non-system messages as the turns this test asserts on, each part collapsed to its prose.
fn recorded_turns(request: &api::LmRequest) -> Vec<ChatTurn> {
    request
        .messages
        .iter()
        .filter(|message| message.role != "system")
        .map(|message| ChatTurn {
            role: match message.role.as_str() {
                "assistant" => Role::Assistant,
                _ => Role::User,
            },
            content: content_of(&message.parts).unwrap_or_else(|_| Content::Text(String::new())),
        })
        .collect()
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
        .call_with(&reqwest::Client::new(), &lm, "capital of Spain?")
        .await
        .expect("the call succeeds");

    let turns = &lm.turns.lock().expect("not poisoned")[0];
    // Two demos become two user/assistant pairs, then the real ask: compiling changed what
    // the model sees, which is the entire point of an optimizer.
    assert_eq!(turns.len(), 5);
    assert_eq!(turns[0].role, Role::User);
    assert_eq!(turns[1].role, Role::Assistant);
    assert!(
        turns[1]
            .content
            .text()
            .unwrap()
            .contains("[[ ## answer ## ]]")
    );
    assert!(
        turns[4]
            .content
            .text()
            .unwrap()
            .contains("capital of Spain?")
    );
}

#[test]
fn a_compiled_program_and_an_uncompiled_one_render_differently() {
    let bare = dsrust::predict::Predict::from_signature(signature());
    let mut compiled = dsrust::predict::Predict::from_signature(signature());
    LabeledFewShot::new(2).compile(&mut compiled, &trainset());

    let inputs = [Input::new("request", json!("capital of Spain?"))];
    let (_, bare_turns) = ChatAdapter::default()
        .format(&signature(), &bare.demos, &inputs)
        .expect("renders");
    let (_, compiled_turns) = ChatAdapter::default()
        .format(&signature(), &compiled.demos, &inputs)
        .expect("renders");

    assert_eq!(bare_turns.len(), 1);
    assert_eq!(compiled_turns.len(), 5);
}
