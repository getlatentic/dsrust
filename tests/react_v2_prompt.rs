//! ReActV2's per-turn signature renders the prompt dspy 3.3 renders, byte for byte.
//!
//! The goldens are the exact `content` of the messages `dspy.ChatAdapter().format(...)` produced
//! for `dspy.ReActV2("question -> answer", tools=[lookup]).react.signature` on dspy 3.3.0b1. The
//! system message states the fields, the `ToolCalls` type description and its JSON schema, and the
//! widened `UnionType[str, NoneType]` input annotation; the user message renders the tools as the
//! strings dspy prints for them.

use dsrust::adapter::{Adapter, ChatAdapter, Input};
use dsrust::{FnTool, ReActV2, Tool};
use serde_json::{Value, json};

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

fn agent() -> ReActV2 {
    ReActV2::new(
        "question -> answer".parse().expect("a signature"),
        vec![lookup()],
    )
}

/// The system message: the input and output field sections, the `ToolCalls` description and schema,
/// and the objective — all as dspy renders them.
#[test]
fn the_turn_signature_system_message_is_dspys_byte_for_byte() {
    let agent = agent();
    let inputs = [Input::new("question", json!("cats"))];
    let (system, _turns) = ChatAdapter::default()
        .format(agent.turn_signature(), &[], &inputs)
        .expect("formats");
    assert_eq!(system, include_str!("goldens/react_v2_turn_system.txt"));
}

/// The first user message: the question, the tools rendered as the strings dspy prints for each,
/// and the output-field requirements — an empty history contributes no turn of its own.
#[test]
fn the_turn_signature_user_message_is_dspys_byte_for_byte() {
    let agent = agent();
    let inputs = [
        Input::new("question", json!("cats")),
        Input::new("history", json!({ "messages": [] })),
        Input::new("tools", agent.turn_tools().clone()),
    ];
    let (_system, turns) = ChatAdapter::default()
        .format(agent.turn_signature(), &[], &inputs)
        .expect("formats");
    let user = turns
        .last()
        .expect("a user turn")
        .content
        .text()
        .expect("prose");
    assert_eq!(user, include_str!("goldens/react_v2_turn_user.txt"));
}
