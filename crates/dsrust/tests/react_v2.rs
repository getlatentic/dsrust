//! ReActV2 against dspy 3.3's `tests/predict/test_react_v2.py`, scenario for scenario.
//!
//! Each test scripts the same replies dspy's does and asserts the same outcome: what the agent
//! answered, how the loop terminated, and what the conversation `History` recorded — the tool-call
//! ids that pair a result to its call included, since that pairing is what a later turn replays.

use std::sync::{Arc, Mutex};

use dsrust::adapter::{ChatAdapter, History, ToolCalls};
use dsrust::lm::api::{self};
use dsrust::lm::{Capabilities, ChatModel, DynChatModel};
use dsrust::{DummyLM, Example, FnTool, Module, ReActV2, Tool};
use serde_json::{Value, json};

/// A tool that echoes what it looked up, dspy's `lookup` in these tests.
fn lookup() -> Box<dyn Tool> {
    Box::new(FnTool::new(
        "lookup",
        "look something up",
        json!({ "query": { "type": "string" } }),
        |args: &Value| {
            Ok(format!(
                "found {}",
                args["query"].as_str().unwrap_or_default()
            ))
        },
    ))
}

/// One scripted react turn: a thought and the calls to make, in the output order the per-turn
/// signature declares.
fn turn(next_thought: &str, tool_calls: Value) -> Example {
    Example::new([
        ("next_thought".to_owned(), json!(next_thought)),
        ("tool_calls".to_owned(), tool_calls),
    ])
}

fn agent(tools: Vec<Box<dyn Tool>>, lm: Arc<DummyLM>) -> ReActV2 {
    ReActV2::new("question -> answer".parse().expect("a signature"), tools)
        .with_lm(lm as Arc<dyn DynChatModel>)
}

fn history_of(prediction: &dsrust::Prediction) -> History {
    serde_json::from_value(prediction.get("history").cloned().expect("a history"))
        .expect("a History")
}

fn calls_in(message: &serde_json::Map<String, Value>) -> ToolCalls {
    serde_json::from_value(message["tool_calls"].clone()).expect("a ToolCalls")
}

/// dspy `test_react_v2_text_mock_lm_loop_records_inputs_once`.
#[tokio::test]
async fn a_text_loop_records_its_inputs_once_and_answers() {
    let lm = Arc::new(DummyLM::new([
        turn(
            "I should look this up.",
            json!({ "tool_calls": [{ "name": "lookup", "args": { "query": "cats" } }] }),
        ),
        turn(
            "I can answer now.",
            json!({ "tool_calls": [{ "name": "submit", "args": { "answer": "found cats" } }] }),
        ),
    ]));
    let prediction = agent(vec![lookup()], lm)
        .forward(Example::new([("question", json!("cats"))]))
        .await
        .expect("the loop runs");

    assert_eq!(
        prediction.get("answer").and_then(Value::as_str),
        Some("found cats")
    );
    assert_eq!(
        prediction.get("termination_reason").and_then(Value::as_str),
        Some("submit")
    );

    let history = history_of(&prediction);
    assert_eq!(
        history
            .messages
            .iter()
            .filter(|message| message.contains_key("question"))
            .count(),
        1,
        "the original inputs are recorded once, on the first turn"
    );
    let first = calls_in(&history.messages[0]);
    assert_eq!(first.tool_calls[0].id.as_deref(), Some("call_0_0"));
    assert!(
        !history.messages[0].contains_key("tool_call_results"),
        "results ride on the calls"
    );
    assert_eq!(
        first.tool_call_results.expect("results").tool_call_results[0]
            .call_id
            .as_deref(),
        Some("call_0_0")
    );
}

/// dspy `test_react_v2_continuation_omits_missing_original_inputs`: once recorded, the task's
/// inputs leave the current turn — they are in the replayed history, not asked again.
#[tokio::test]
async fn a_continuation_turn_omits_the_original_inputs() {
    let lm = Arc::new(DummyLM::new([
        turn(
            "I should look this up.",
            json!({ "tool_calls": [{ "name": "lookup", "args": { "query": "cats" } }] }),
        ),
        turn(
            "I can answer now.",
            json!({ "tool_calls": [{ "name": "submit", "args": { "answer": "found cats" } }] }),
        ),
    ]));
    let prediction = agent(vec![lookup()], lm.clone())
        .forward(Example::new([("question", json!("cats"))]))
        .await
        .expect("the loop runs");
    assert_eq!(
        prediction.get("answer").and_then(Value::as_str),
        Some("found cats")
    );

    let asked = lm.asked();
    let second = &asked[1];
    assert!(
        !second.last_message().contains("[[ ## question ## ]]"),
        "the current turn does not ask the question again: {}",
        second.last_message()
    );
    assert!(
        second.turns.iter().any(|turn| turn
            .content
            .text()
            .unwrap_or_default()
            .contains("[[ ## question ## ]]\ncats")),
        "but the replayed history still carries it"
    );
}

/// dspy `test_react_v2_text_mode_accepts_top_level_tool_arguments`: a call stated at the top level,
/// `{name, arguments}`, is read the same as a wrapped one.
#[tokio::test]
async fn a_top_level_tool_call_is_accepted() {
    let lm = Arc::new(DummyLM::new([
        turn(
            "I should look this up.",
            json!({ "name": "lookup", "arguments": { "query": "cats" } }),
        ),
        turn(
            "I can answer now.",
            json!({ "tool_calls": [{ "name": "submit", "args": { "answer": "found cats" } }] }),
        ),
    ]));
    let prediction = agent(vec![lookup()], lm)
        .with_adapter(ChatAdapter::default())
        .forward(Example::new([("question", json!("cats"))]))
        .await
        .expect("the loop runs");

    assert_eq!(
        prediction.get("answer").and_then(Value::as_str),
        Some("found cats")
    );
    let history = history_of(&prediction);
    let first = calls_in(&history.messages[0]);
    assert_eq!(
        first.tool_calls[0].args,
        *json!({ "query": "cats" }).as_object().unwrap()
    );
}

/// dspy `test_react_v2_text_mode_accepts_wrapped_submit_arguments`: `submit` stated as a wrapped
/// call with `arguments` submits the same as one with `args`.
#[tokio::test]
async fn a_wrapped_submit_call_is_accepted() {
    let lm = Arc::new(DummyLM::new([turn(
        "I can answer now.",
        json!({ "tool_calls": [{ "name": "submit", "arguments": { "answer": "done" } }] }),
    )]));
    let prediction = agent(vec![], lm)
        .with_adapter(ChatAdapter::default())
        .forward(Example::new([("question", json!("cats"))]))
        .await
        .expect("the loop runs");

    assert_eq!(
        prediction.get("answer").and_then(Value::as_str),
        Some("done")
    );
    assert_eq!(
        prediction.get("termination_reason").and_then(Value::as_str),
        Some("submit")
    );
}

/// dspy `test_react_v2_unknown_tool_observation_can_continue`: an unknown tool is an error
/// observation the agent reads and recovers from, not an abort.
#[tokio::test]
async fn an_unknown_tool_becomes_an_observation_and_the_loop_continues() {
    let lm = Arc::new(DummyLM::new([
        turn(
            "Try a missing tool.",
            json!({ "tool_calls": [{ "name": "missing_tool", "args": { "query": "cats" } }] }),
        ),
        turn(
            "Recover with a final answer.",
            json!({ "tool_calls": [{ "name": "submit", "args": { "answer": "done" } }] }),
        ),
    ]));
    let prediction = agent(vec![], lm)
        .forward(Example::new([("question", json!("cats"))]))
        .await
        .expect("the loop runs");

    let history = history_of(&prediction);
    let first_result = calls_in(&history.messages[0])
        .tool_call_results
        .expect("results")
        .tool_call_results
        .remove(0);
    assert!(first_result.is_error);
    assert_eq!(first_result.call_id.as_deref(), Some("call_0_0"));
    assert!(
        first_result
            .value
            .as_str()
            .unwrap_or_default()
            .contains("Unknown tool")
    );
    assert_eq!(
        prediction.get("answer").and_then(Value::as_str),
        Some("done")
    );
}

/// dspy `test_react_v2_accepts_serialized_history_input`: a history handed in as a plain mapping is
/// carried through and added to.
#[tokio::test]
async fn a_serialized_history_input_is_accepted() {
    let lm = Arc::new(DummyLM::new([turn(
        "I can answer.",
        json!({ "tool_calls": [{ "name": "submit", "args": { "answer": "done" } }] }),
    )]));
    let prediction = agent(vec![], lm)
        .forward(Example::new([(
            "history",
            json!({ "messages": [{ "question": "old" }] }),
        )]))
        .await
        .expect("the loop runs");

    assert_eq!(
        prediction.get("answer").and_then(Value::as_str),
        Some("done")
    );
    let history = history_of(&prediction);
    assert_eq!(
        history.messages[0],
        *json!({ "question": "old" }).as_object().unwrap()
    );
    assert!(history.messages.iter().all(|message| !message.is_empty()));
}

/// dspy `test_react_v2_forced_submit_on_empty_tool_calls`: a turn that makes no calls ends the
/// loop, and one last turn is forced to submit.
#[tokio::test]
async fn empty_tool_calls_force_a_final_submit() {
    let lm = Arc::new(DummyLM::new([
        turn("No action.", json!({ "tool_calls": [] })),
        turn(
            "Forced final.",
            json!({ "tool_calls": [{ "name": "submit", "args": { "answer": "forced" } }] }),
        ),
    ]));
    let prediction = agent(vec![], lm)
        .forward(Example::new([("question", json!("cats"))]))
        .await
        .expect("the loop runs");

    assert_eq!(
        prediction.get("answer").and_then(Value::as_str),
        Some("forced")
    );
    assert_eq!(
        prediction.get("termination_reason").and_then(Value::as_str),
        Some("forced_submit")
    );
}

// --- Native function calling: the provider calls the tools itself. -----------------------------

/// A model that calls tools of its own, one reply per turn, recording every request it was handed.
struct NativeToolLM {
    replies: Vec<api::LmResponse>,
    seen: Mutex<Vec<api::LmRequest>>,
}

impl NativeToolLM {
    fn new(replies: Vec<api::LmResponse>) -> Arc<Self> {
        Arc::new(Self {
            replies,
            seen: Mutex::new(Vec::new()),
        })
    }
}

impl ChatModel for NativeToolLM {
    async fn forward(
        &self,
        _http: &reqwest::Client,
        request: &api::LmRequest,
    ) -> anyhow::Result<api::LmResponse> {
        let mut seen = self.seen.lock().expect("not poisoned");
        let reply = self.replies[seen.len().min(self.replies.len() - 1)].clone();
        seen.push(request.clone());
        Ok(reply)
    }

    fn capabilities<'a>(
        &'a self,
        _http: &'a reqwest::Client,
    ) -> impl std::future::Future<Output = Capabilities> + Send + 'a {
        std::future::ready(Capabilities {
            function_calling: true,
            ..Default::default()
        })
    }
}

fn native_call(id: &str, name: &str, args: Value) -> api::LmResponse {
    api::LmResponse {
        outputs: vec![api::LmOutput {
            parts: vec![api::LmPart::ToolCall {
                id: Some(id.to_owned()),
                name: name.to_owned(),
                args: args.as_object().expect("an object").clone(),
                provider_data: Default::default(),
                metadata: Default::default(),
            }],
            finish_reason: Some("tool_calls".into()),
            ..Default::default()
        }],
        ..Default::default()
    }
}

/// A native reply that made no tool call — the turn that ends the loop into a forced submit.
fn no_call() -> api::LmResponse {
    api::LmResponse {
        outputs: vec![api::LmOutput {
            finish_reason: Some("stop".into()),
            ..Default::default()
        }],
        ..Default::default()
    }
}

/// A `tool` message in a request, by the id it answers.
fn tool_message_ids(request: &api::LmRequest) -> Vec<String> {
    request
        .messages
        .iter()
        .filter(|message| message.role == "tool")
        .filter_map(|message| {
            message.parts.iter().find_map(|part| match part {
                api::LmPart::ToolResult { call_id, .. } => call_id.clone(),
                _ => None,
            })
        })
        .collect()
}

/// dspy `test_react_v2_native_tool_loop_replays_tool_result_with_provider_id`: the provider's own
/// call ids survive into the history, and the next turn replays each result as a `tool` message
/// naming the call it answers.
#[tokio::test]
async fn a_native_loop_replays_results_with_the_providers_id() {
    let lm = NativeToolLM::new(vec![
        native_call("call_provider_1", "lookup", json!({ "query": "cats" })),
        native_call("call_submit", "submit", json!({ "answer": "found cats" })),
    ]);
    let prediction = ReActV2::new("question -> answer".parse().unwrap(), vec![lookup()])
        .with_adapter(ChatAdapter::default().with_native_function_calling(true))
        .with_lm(lm.clone() as Arc<dyn DynChatModel>)
        .forward(Example::new([("question", json!("cats"))]))
        .await
        .expect("the loop runs");

    assert_eq!(
        prediction.get("answer").and_then(Value::as_str),
        Some("found cats")
    );
    let history = history_of(&prediction);
    let first = calls_in(&history.messages[0]);
    assert_eq!(first.tool_calls[0].id.as_deref(), Some("call_provider_1"));
    assert!(!history.messages[0].contains_key("tool_call_results"));
    assert_eq!(
        first.tool_call_results.expect("results").tool_call_results[0]
            .call_id
            .as_deref(),
        Some("call_provider_1")
    );
    assert_eq!(
        tool_message_ids(&lm.seen.lock().unwrap()[1]),
        ["call_provider_1"]
    );
}

/// dspy `_forced_submit` steers the last ask with `tool_choice={"function":{"name":"submit"}}`: a
/// turn that makes no call ends the loop, and the forced ask pins the provider to `submit`.
#[tokio::test]
async fn a_forced_submit_pins_the_provider_to_submit() {
    let lm = NativeToolLM::new(vec![
        no_call(),
        native_call("call_submit", "submit", json!({ "answer": "forced" })),
    ]);
    let prediction = ReActV2::new("question -> answer".parse().unwrap(), vec![lookup()])
        .with_adapter(ChatAdapter::default().with_native_function_calling(true))
        .with_lm(lm.clone() as Arc<dyn DynChatModel>)
        .forward(Example::new([("question", json!("cats"))]))
        .await
        .expect("the loop runs");

    assert_eq!(
        prediction.get("answer").and_then(Value::as_str),
        Some("forced")
    );
    assert_eq!(
        prediction.get("termination_reason").and_then(Value::as_str),
        Some("forced_submit")
    );

    let seen = lm.seen.lock().unwrap();
    let forced = seen[1]
        .config
        .tool_choice
        .as_ref()
        .expect("the forced ask states a tool choice");
    assert_eq!(
        forced.allowed,
        Some(vec!["submit".to_owned()]),
        "pinned to submit"
    );
    // The normal turn before it was not steered.
    assert!(
        seen[0]
            .config
            .tool_choice
            .as_ref()
            .is_none_or(|choice| choice.allowed.is_none())
    );
}

/// dspy `test_react_v2_native_parallel_tool_calls_are_requested_and_replayed`: several calls come
/// back on one turn, keep their ids and order, and all replay as `tool` messages next turn.
#[tokio::test]
async fn native_parallel_calls_replay_in_order() {
    let parallel = api::LmResponse {
        outputs: vec![api::LmOutput {
            parts: vec![
                api::LmPart::ToolCall {
                    id: Some("call_provider_1".into()),
                    name: "lookup".into(),
                    args: json!({ "query": "cats" }).as_object().unwrap().clone(),
                    provider_data: Default::default(),
                    metadata: Default::default(),
                },
                api::LmPart::ToolCall {
                    id: Some("call_provider_2".into()),
                    name: "lookup".into(),
                    args: json!({ "query": "dogs" }).as_object().unwrap().clone(),
                    provider_data: Default::default(),
                    metadata: Default::default(),
                },
            ],
            finish_reason: Some("tool_calls".into()),
            ..Default::default()
        }],
        ..Default::default()
    };
    let lm = NativeToolLM::new(vec![
        parallel,
        native_call(
            "call_submit",
            "submit",
            json!({ "answer": "found cats and found dogs" }),
        ),
    ]);
    let prediction = ReActV2::new("question -> answer".parse().unwrap(), vec![lookup()])
        .with_adapter(
            ChatAdapter::default()
                .with_native_function_calling(true)
                .with_parallel_tool_calls(Some(true)),
        )
        .with_lm(lm.clone() as Arc<dyn DynChatModel>)
        .forward(Example::new([("question", json!("cats and dogs"))]))
        .await
        .expect("the loop runs");

    assert_eq!(
        prediction.get("answer").and_then(Value::as_str),
        Some("found cats and found dogs")
    );
    let history = history_of(&prediction);
    let first = calls_in(&history.messages[0]);
    let ids: Vec<&str> = first
        .tool_calls
        .iter()
        .filter_map(|call| call.id.as_deref())
        .collect();
    assert_eq!(ids, ["call_provider_1", "call_provider_2"]);
    let results = first.tool_call_results.expect("results");
    let result_ids: Vec<&str> = results
        .tool_call_results
        .iter()
        .filter_map(|result| result.call_id.as_deref())
        .collect();
    assert_eq!(result_ids, ["call_provider_1", "call_provider_2"]);
    assert_eq!(
        tool_message_ids(&lm.seen.lock().unwrap()[1]),
        ["call_provider_1", "call_provider_2"]
    );
}
