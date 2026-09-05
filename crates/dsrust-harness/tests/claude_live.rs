//! dsrust programs on the real Claude Code CLI.
//!
//! Ignored by default: needs `claude` installed and signed in, and costs tokens.
//!
//! ```text
//! cargo test -p dsrust-harness --test claude_live -- --ignored
//! ```

use std::sync::Arc;

use dsrust::lm::DynChatModel;
use dsrust::{Example, FnTool, Module, Predict, ReActV2, Tool};
use dsrust_harness::HarnessModel;
use harness::{Claude, ToolAccess};
use serde_json::{Value, json};

/// Not a word, and absent from the prompt: it reaches an answer only via the tool.
const PLANTED: &str = "quartzine-80417";

fn claude() -> Arc<dyn DynChatModel> {
    let workspace = std::env::temp_dir().join("dsrust-harness-live");
    std::fs::create_dir_all(&workspace).expect("workspace");
    Arc::new(
        HarnessModel::builder(Claude::new())
            .cwd(workspace)
            .max_turns(4)
            .build()
            .expect("claude withholds its tools"),
    )
}

/// Route B: an ordinary `Predict`, its fields read back out of the agent's reply.
#[tokio::test]
#[ignore = "live: needs the claude CLI installed and signed in; costs tokens"]
async fn a_predict_over_claude_code_fills_its_output_field() {
    let qa = Predict::parse("question -> answer")
        .expect("a signature")
        .set_lm(claude());
    let prediction = qa
        .forward(Example::new([(
            "question",
            json!("What is 2 + 2? Answer with just the number."),
        )]))
        .await
        .expect("the call completes");
    let answer = prediction
        .get("answer")
        .and_then(Value::as_str)
        .expect("an answer field")
        .to_owned();
    assert!(
        answer.contains('4'),
        "the answer field carries the answer: {answer:?}"
    );
}

/// Route C: `ReActV2`'s own loop over Claude Code as a text model. Three ways this
/// fails silently, each with its own signature in `termination_reason`, so the
/// control asserts the output is present **and** that neither signature fired:
///
/// - `empty_tool_calls` — the backend ran tools itself and returned prose, so
///   the loop had nothing to dispatch and broke on turn one;
/// - `parse_error` — the backend Markdown-formatted its reply and mangled the
///   field markers, so the turn could not be read at all.
///
/// The tool is the only route to the planted token, so a present answer carrying
/// it proves the loop dispatched the call in *this* process.
#[tokio::test]
#[ignore = "live: needs the claude CLI installed and signed in; costs tokens"]
async fn react_v2_over_claude_code_dispatches_its_own_tools_and_answers() {
    let lookup: Box<dyn Tool> = Box::new(FnTool::new(
        "lookup",
        "Look up the secret code for a project name.",
        json!({ "project": { "type": "string" } }),
        |args: &Value| {
            Ok(format!(
                "The code for {} is {PLANTED}.",
                args["project"].as_str().unwrap_or("?")
            ))
        },
    ));
    let agent = ReActV2::new(
        "question -> answer".parse().expect("a signature"),
        vec![lookup],
    )
    .set_lm(claude());
    let prediction = agent
        .forward(Example::new([(
            "question",
            json!("What is the secret code for project Aurora? Use the lookup tool."),
        )]))
        .await
        .expect("the loop ends rather than raising");

    let reason = prediction
        .get("termination_reason")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_owned();
    assert_ne!(
        reason, "empty_tool_calls",
        "the backend executed tools itself instead of returning calls"
    );
    assert_ne!(
        reason, "parse_error",
        "the backend Markdown-formatted its reply and broke the markers"
    );
    let answer = prediction
        .get("answer")
        .and_then(Value::as_str)
        .expect("the task's output field is present");
    assert!(
        answer.contains(PLANTED),
        "the answer carries what the tool returned: {answer:?} (reason {reason:?})"
    );
}

/// Route A: the agent as a module. Claude Code runs its own loop with a dsrust
/// tool mounted in this process, and an ordinary `Predict` reads the answer it
/// reached. The token is reachable only through the tool, and the flag is set
/// only by the closure below — so an answer carrying it proves the agent called
/// back into this test's process, on its own initiative, and dsrust never saw a
/// tool call at all.
#[tokio::test]
#[ignore = "live: needs the claude CLI installed and signed in; costs tokens"]
async fn a_predict_over_claude_as_an_agent_reaches_a_dsrust_tool_in_this_process() {
    use dsrust_harness::tool_server;
    use std::sync::atomic::{AtomicBool, Ordering};

    let called = Arc::new(AtomicBool::new(false));
    let seen = Arc::clone(&called);
    let lookup: Box<dyn Tool> = Box::new(FnTool::new(
        "lookup",
        "Look up the secret code for a project name.",
        json!({ "project": { "type": "string" } }),
        move |args: &Value| {
            seen.store(true, Ordering::SeqCst);
            Ok(format!(
                "The code for {} is {PLANTED}.",
                args["project"].as_str().unwrap_or("?")
            ))
        },
    ));
    let workspace = std::env::temp_dir().join("dsrust-harness-live-a");
    std::fs::create_dir_all(&workspace).expect("workspace");
    let model =
        HarnessModel::builder(Claude::new().with_tool_server(tool_server("shop", [lookup])))
            .tools(ToolAccess::Default)
            .cwd(workspace)
            .max_turns(6)
            .build()
            .expect("an agent with its tools");
    let qa = Predict::parse("question -> answer")
        .expect("a signature")
        .set_lm(Arc::new(model) as Arc<dyn DynChatModel>);
    let prediction = qa
        .forward(Example::new([(
            "question",
            json!("What is the secret code for project Aurora? The shop lookup tool knows."),
        )]))
        .await
        .expect("the call completes");

    assert!(
        called.load(Ordering::SeqCst),
        "the agent never called back into this process"
    );
    let answer = prediction
        .get("answer")
        .and_then(Value::as_str)
        .expect("the output field is present");
    assert!(
        answer.contains(PLANTED),
        "the answer carries what the tool returned: {answer:?}"
    );
}
