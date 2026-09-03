//! An async tool, driven by a real agent loop rather than called directly.
//!
//! dspy's `Tool.acall` awaits a tool whose callable is a coroutine, and its agents go through that
//! path. The unit tests in `tool_macro.rs` await `acall_value` themselves, which says nothing about
//! whether a loop does — so these run `ReAct` and `ReActV2` over a tool that genuinely suspends,
//! and check both that it ran and that the loop saw what it returned.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use dsrust::lm::DynChatModel;
use dsrust::{DummyLM, Example, Module, ReAct, ReActV2, tool};
use serde_json::{Value, json};

/// The calls the tool actually made, so a loop that never ran it is a failing assertion rather
/// than an empty answer that reads like the model declining. Per-instance, not a global: two
/// `#[tokio::test]`s in one binary run at once, and a shared log makes each the other's flake.
#[derive(Default)]
struct Remote {
    called: Mutex<Vec<String>>,
}

impl Remote {
    /// Look one term up on the remote.
    #[tool]
    pub async fn lookup(&self, query: String) -> anyhow::Result<String> {
        // A real suspension: the future is polled again after the timer, so a loop that resolved
        // it without awaiting could not observe the answer.
        tokio::time::sleep(Duration::from_millis(5)).await;
        self.called
            .lock()
            .expect("not poisoned")
            .push(query.clone());
        Ok(format!("found {query}"))
    }
}

fn turn(next_thought: &str, tool_calls: Value) -> Example {
    Example::new([
        ("next_thought".to_owned(), json!(next_thought)),
        ("tool_calls".to_owned(), tool_calls),
    ])
}

fn remote() -> Arc<Remote> {
    Arc::new(Remote::default())
}

#[tokio::test]
async fn react_v2_awaits_an_async_tool() {
    let remote = remote();
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
    let prediction = ReActV2::new(
        "question -> answer".parse().expect("a signature"),
        vec![remote.lookup_tool()],
    )
    .set_lm(lm as Arc<dyn DynChatModel>)
    .forward(Example::new([("question", json!("cats"))]))
    .await
    .expect("the loop runs");

    assert_eq!(
        prediction.get("answer").and_then(Value::as_str),
        Some("found cats")
    );
    assert_eq!(*remote.called.lock().expect("not poisoned"), ["cats"]);
}

#[tokio::test]
async fn react_awaits_an_async_tool_and_reads_its_answer() {
    let remote = remote();
    let lm = Arc::new(DummyLM::new([
        Example::new([
            ("next_thought", json!("Look it up.")),
            ("next_tool_name", json!("lookup")),
            ("next_tool_args", json!({ "query": "dogs" })),
        ]),
        Example::new([
            ("next_thought", json!("Done.")),
            ("next_tool_name", json!("finish")),
            ("next_tool_args", json!({})),
        ]),
        Example::new([
            ("reasoning", json!("The tool said so.")),
            ("answer", json!("found dogs")),
        ]),
    ]));
    let prediction = ReAct::new(
        "question -> answer".parse().expect("a signature"),
        vec![remote.lookup_tool()],
    )
    .set_lm(lm as Arc<dyn DynChatModel>)
    .forward(Example::new([("question", json!("dogs"))]))
    .await
    .expect("the loop runs");

    assert_eq!(*remote.called.lock().expect("not poisoned"), ["dogs"]);
    // The observation the awaited tool produced has to reach the trajectory, not just run.
    let trajectory = prediction
        .get("trajectory")
        .expect("a trajectory")
        .to_string();
    assert!(trajectory.contains("found dogs"), "{trajectory}");
}
