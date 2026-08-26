//! The ReAct loop actually running: think, call a tool, read the observation, finish.
//!
//! The unit tests check the signatures and the tool wrapper. This drives the whole episode
//! against a scripted model, which is the only way to see that observations reach the next
//! turn and that the budget stops a model that never finishes.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use dsrust::lm::{ChatModel, api, global};
use dsrust::signature::{OutField, Signature};
use dsrust::{FnTool, Module, ReAct, Tool, example, react::arg_str};
use serde_json::Value;

struct Scripted {
    replies: Mutex<VecDeque<String>>,
    seen: Mutex<Vec<String>>,
}

impl Scripted {
    fn new(replies: &[&str]) -> Self {
        Self {
            replies: Mutex::new(replies.iter().map(|r| (*r).to_owned()).collect()),
            seen: Mutex::new(Vec::new()),
        }
    }
    fn calls(&self) -> usize {
        self.seen.lock().expect("not poisoned").len()
    }
}

impl ChatModel for Scripted {
    async fn forward(&self, request: &api::LmRequest) -> Result<api::LmResponse> {
        let last = request
            .user_messages()
            .last()
            .and_then(|message| message["content"].as_str().map(str::to_owned))
            .unwrap_or_default();
        self.seen.lock().expect("not poisoned").push(last);
        self.replies
            .lock()
            .expect("not poisoned")
            .pop_front()
            .map(api::LmResponse::text)
            .ok_or_else(|| anyhow::anyhow!("script exhausted"))
    }
}

fn turn(thought: &str, tool: &str, args: &str) -> String {
    format!(
        "[[ ## next_thought ## ]]\n{thought}\n\n[[ ## next_tool_name ## ]]\n{tool}\n\n\
         [[ ## next_tool_args ## ]]\n{args}\n\n[[ ## completed ## ]]"
    )
}

/// dspy's extract pass is a `ChainOfThought`, so a reply that skips `reasoning` is missing a
/// field the signature demands and never reaches the caller.
fn answer(text: &str) -> String {
    format!(
        "[[ ## reasoning ## ]]\nthe trajectory says so\n\n\
         [[ ## answer ## ]]\n{text}\n\n[[ ## completed ## ]]"
    )
}

fn task() -> Signature {
    Signature::single_input(
        "Answer the question.",
        vec![OutField {
            name: "answer".into(),
            desc: "the answer".into(),
            ..Default::default()
        }],
    )
}

fn weather() -> Box<dyn Tool> {
    Box::new(FnTool::new(
        "get_weather",
        "look up the weather for a city",
        serde_json::json!({ "city": { "type": "string" } }),
        |args: &Value| {
            Ok(format!(
                "The weather in {} is sunny.",
                arg_str(args, "city")?
            ))
        },
    ))
}

/// The configured model is process-wide, so these tests take turns rather than racing to
/// overwrite each other's script.
static GLOBAL_LM: Mutex<()> = Mutex::new(());

/// A `Module` reaches its model through the global, matching dspy's settings, so a scripted
/// model is installed the same way a `DummyLM` would be.
fn configure(lm: Arc<Scripted>) -> std::sync::MutexGuard<'static, ()> {
    let guard = GLOBAL_LM
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    global::configure_model(reqwest::Client::new(), lm);
    guard
}

#[tokio::test]
async fn the_agent_calls_a_tool_then_finishes() {
    let lm = Arc::new(Scripted::new(&[
        &turn(
            "I should look up the weather",
            "get_weather",
            r#"{"city":"Tokyo"}"#,
        ),
        &turn("Now I can answer", "finish", "{}"),
        &answer("It is sunny in Tokyo."),
    ]));
    let _guard = configure(lm.clone());

    let react = ReAct::new(task(), vec![weather()]);
    let prediction = react
        .forward(example! { request: "What is the weather in Tokyo?" }.with_inputs(["request"]))
        .await
        .expect("the episode completes");

    assert_eq!(
        prediction.get("answer").and_then(Value::as_str),
        Some("It is sunny in Tokyo.")
    );
    // Two loop turns plus the extract pass.
    assert_eq!(lm.calls(), 3);

    // The observation from turn one reached turn two, which is the point of the trajectory.
    let seen = lm.seen.lock().expect("not poisoned");
    assert!(seen[1].contains("The weather in Tokyo is sunny."));
}

#[tokio::test]
async fn the_budget_stops_a_model_that_never_finishes() {
    // Without a cap this would loop against a paid provider forever.
    let never = turn("still thinking", "get_weather", r#"{"city":"Tokyo"}"#);
    let lm = Arc::new(Scripted::new(&[&never, &never, &never, &answer("gave up")]));
    let _guard = configure(lm.clone());

    let react = ReAct::new(task(), vec![weather()]).max_iters(3);
    react
        .forward(example! { request: "endless" }.with_inputs(["request"]))
        .await
        .expect("the episode ends at the budget");

    assert_eq!(
        lm.calls(),
        4,
        "three turns capped by max_iters, then extract"
    );
}

#[tokio::test]
async fn a_failing_tool_reports_into_the_trajectory_and_the_episode_continues() {
    let lm = Arc::new(Scripted::new(&[
        // No `city` argument: the tool errors.
        &turn("try it", "get_weather", "{}"),
        &turn("recovered", "finish", "{}"),
        &answer("recovered fine"),
    ]));
    let _guard = configure(lm.clone());

    let react = ReAct::new(task(), vec![weather()]);
    react
        .forward(example! { request: "recover" }.with_inputs(["request"]))
        .await
        .expect("a tool error does not end the episode");

    let seen = lm.seen.lock().expect("not poisoned");
    assert!(
        seen[1].contains("Execution error in get_weather"),
        "the model sees the error and can try something else"
    );
}
