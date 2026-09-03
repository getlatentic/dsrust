//! Every module a task was named on answers the same way.
//!
//! dspy is uniform without trying: each module returns a `Prediction` and `__getattr__` reads its
//! store, so `out.answer` is one line whichever module wrote it. Here the task arm of each module
//! macro carries the task, so `call!` answers with the task's own outputs — the same line after
//! `Predict!(QA)`, `ChainOfThought!(QA)` or `ReActV2!(QA, tools)`.
//!
//! The case that makes an agent different is at the bottom: a loop can finish having produced no
//! outputs at all, which `Predict` never does.

use std::sync::Arc;

use dsrust::lm::DynChatModel;
use dsrust::{DummyLM, Example, Module, ReActV2, Signature, Tool, call, tool};
use serde_json::json;

#[derive(Signature)]
/// Answer the question.
struct QA {
    #[input]
    question: String,
    #[output]
    answer: String,
}

#[tool]
/// Look one term up.
fn lookup(query: String) -> anyhow::Result<String> {
    Ok(format!("found {query}"))
}

fn agent(replies: Vec<Example>) -> dsrust::Typed<QA, ReActV2> {
    let tools: Vec<Box<dyn Tool>> = vec![Box::new(Lookup)];
    ReActV2!(QA, tools)
        .map(|agent| agent.set_lm(Arc::new(DummyLM::new(replies)) as Arc<dyn DynChatModel>))
}

fn turn(thought: &str, calls: serde_json::Value) -> Example {
    Example::new([("next_thought", json!(thought)), ("tool_calls", calls)])
}

/// The task spelling, the short call, and the field read off a struct rather than looked up.
#[tokio::test]
async fn an_agent_answers_with_the_task_it_was_given() {
    let agent = agent(vec![turn(
        "I can answer.",
        json!({ "tool_calls": [{ "name": "submit", "args": { "answer": "Paris" } }] }),
    )]);
    let out = call!(agent, question = "capital of France?")
        .await
        .expect("the loop runs");
    // The same line `Predict!(QA)` and `ChainOfThought!(QA)` answer with.
    assert_eq!(out.answer, "Paris");
}

/// A trajectory the task never declared is ignored rather than refused: an agent's `Prediction`
/// carries the loop's own record beside the answer, and the task only names `answer`.
#[tokio::test]
async fn the_loops_own_fields_do_not_get_in_the_way() {
    let agent = agent(vec![
        turn(
            "Look it up.",
            json!({ "tool_calls": [{ "name": "lookup", "args": { "query": "cats" } }] }),
        ),
        turn(
            "Now I can answer.",
            json!({ "tool_calls": [{ "name": "submit", "args": { "answer": "found cats" } }] }),
        ),
    ]);
    // The untyped module is still reachable, which is where the loop's own record lives: a typed
    // answer is the task's fields and has nowhere to put a trajectory.
    let untyped = agent.into_module();
    let prediction = untyped
        .forward(Example::new([("question", json!("cats"))]))
        .await
        .expect("the loop runs");
    assert!(prediction.get("history").is_some(), "the loop records one");
    assert_eq!(
        prediction.typed::<QAOutputs>().expect("the outputs").answer,
        "found cats"
    );
}

/// The case `Predict` never reaches: the loop finished, successfully, with no outputs at all —
/// it ran out of turns. dspy would raise `AttributeError` on `out.answer`; naming the reason is
/// more use than naming the absent field.
#[tokio::test]
async fn a_loop_that_produced_nothing_says_so() {
    let agent = agent(vec![
        turn("Thinking.", json!({ "tool_calls": [] })),
        turn("Still thinking.", json!({ "tool_calls": [] })),
        turn("Still thinking.", json!({ "tool_calls": [] })),
    ])
    .map(|agent| agent.max_iters(1));
    let refused = call!(agent, question = "cats")
        .await
        .expect_err("the loop produced no outputs");
    assert!(
        refused
            .to_string()
            .starts_with("the loop ended without producing the outputs"),
        "{refused}"
    );
}
