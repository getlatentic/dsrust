//! `Evaluate` scoring a real `Predict`, not a stand-in closure.
//!
//! The unit tests cover the scoring arithmetic with hand-written programs. This covers the
//! join a caller actually makes: a devset of labelled examples, a module that renders prompts
//! and parses replies, and a metric over the two. If those three do not compose, an optimizer
//! built on top of them cannot either.

use std::collections::VecDeque;
use std::sync::Mutex;

use anyhow::Result;
use dsrust::lm::{ChatModel, api};
use dsrust::signature::{OutField, Signature};
use dsrust::{Evaluate, Example, Prediction, exact_match, example};
use serde_json::json;

/// Pops one canned reply per call, so a whole devset can be scripted in order.
struct Scripted {
    replies: Mutex<VecDeque<String>>,
}

impl Scripted {
    fn new(replies: &[&str]) -> Self {
        Self {
            replies: Mutex::new(replies.iter().map(|r| (*r).to_owned()).collect()),
        }
    }
}

impl ChatModel for Scripted {
    async fn forward(&self, _request: &api::LmRequest) -> Result<api::LmResponse> {
        self.replies
            .lock()
            .expect("not poisoned")
            .pop_front()
            .map(api::LmResponse::text)
            .ok_or_else(|| anyhow::anyhow!("the script ran out of replies"))
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

fn devset() -> Vec<Example> {
    vec![
        example! { request: "capital of France?", answer: "Paris" }.with_inputs(["request"]),
        example! { request: "capital of Germany?", answer: "Berlin" }.with_inputs(["request"]),
    ]
}

fn reply(answer: &str) -> String {
    format!("[[ ## answer ## ]]\n{answer}\n\n[[ ## completed ## ]]")
}

#[tokio::test]
async fn a_module_that_answers_correctly_scores_a_hundred() {
    let lm = Scripted::new(&[&reply("Paris"), &reply("Berlin")]);
    let predict = dsrust::predict::Predict::from_signature(signature());

    let evaluation = Evaluate::new(
        devset(),
        |example: Example| {
            let request = example
                .get("request")
                .and_then(|value| value.as_str())
                .unwrap_or_default()
                .to_owned();
            let predict = &predict;
            let lm = &lm;
            async move {
                let value = predict.call_with(lm, &request).await?;
                let answer = value["answer"].clone();
                Ok(Prediction::new(
                    Example::new([("answer", answer)]),
                    value.to_string(),
                ))
            }
        },
        exact_match,
    )
    .run()
    .await
    .expect("the run stays inside its error budget");

    // The percentage upstream reports, not the 0..1 mean the rows carry.
    assert_eq!(evaluation.score, 100.0);
    assert_eq!(evaluation.failure_count(), 0);
}

#[tokio::test]
async fn a_wrong_answer_scores_zero_and_keeps_the_reply_for_inspection() {
    let lm = Scripted::new(&[&reply("Lyon"), &reply("Berlin")]);
    let predict = dsrust::predict::Predict::from_signature(signature());

    let evaluation = Evaluate::new(
        devset(),
        |example: Example| {
            let request = example
                .get("request")
                .and_then(|value| value.as_str())
                .unwrap_or_default()
                .to_owned();
            let predict = &predict;
            let lm = &lm;
            async move {
                let value = predict.call_with(lm, &request).await?;
                Ok(Prediction::new(
                    Example::new([("answer", value["answer"].clone())]),
                    value.to_string(),
                ))
            }
        },
        exact_match,
    )
    .run()
    .await
    .expect("the run stays inside its error budget");

    assert_eq!(evaluation.score, 50.0);
    // The failing row keeps what the model said, which is the first thing to look at.
    let wrong = &evaluation.results[0];
    assert_eq!(wrong.score, 0.0);
    assert_eq!(
        wrong.prediction.as_ref().unwrap().get("answer"),
        Some(&json!("Lyon"))
    );
}
