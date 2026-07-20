//! An `Example` used as a demo travels all the way into the prompt.
//!
//! The unit tests check the split and the rendering separately; this checks the thing a
//! caller actually does — build labelled examples, hand them to a module, and have the model
//! see them as solved turns before the real request.

use dsrs::lm::Role;
use dsrs::signature::{FieldKind, InField, OutField, Signature};
use dsrs::{Adapter, ChatAdapter, Example};
use serde_json::json;

fn signature() -> Signature {
    Signature::single_input(
        "Pick a colour.",
        vec![OutField {
            name: "colour".into(),
            desc: "the chosen colour".into(),
            ..Default::default()
        }],
    )
}

fn demo(request: &str, colour: &str) -> Example {
    Example::new([("request", json!(request)), ("colour", json!(colour))]).with_inputs(["request"])
}

#[test]
fn demos_become_solved_turns_before_the_request() {
    let demos = [
        demo("something calm", "blue"),
        demo("something warm", "amber"),
    ];
    let inputs = [("request", json!("something bold"))];
    let (_, turns) = ChatAdapter::default()
        .format(&signature(), &demos, &inputs)
        .expect("renders");

    // Two demos, each a user/assistant pair, then the real ask.
    let roles: Vec<Role> = turns.iter().map(|turn| turn.role).collect();
    assert_eq!(
        roles,
        [
            Role::User,
            Role::Assistant,
            Role::User,
            Role::Assistant,
            Role::User
        ]
    );

    assert_eq!(
        turns[0].content.text().unwrap(),
        "[[ ## request ## ]]\nsomething calm"
    );
    assert_eq!(
        turns[1].content.text().unwrap(),
        "[[ ## colour ## ]]\nblue\n\n[[ ## completed ## ]]\n"
    );
    assert!(
        turns[4]
            .content
            .text()
            .unwrap()
            .starts_with("[[ ## request ## ]]\nsomething bold")
    );

    // Only the real ask carries the format reminder; a demo already shows the answer.
    assert!(
        !turns[0]
            .content
            .text()
            .unwrap()
            .contains("Respond with the corresponding output fields")
    );
    assert!(
        turns[4]
            .content
            .text()
            .unwrap()
            .contains("Respond with the corresponding output fields")
    );
}

/// The expected bytes are what `dspy.ChatAdapter().format` emits for this signature and demo
/// on dspy 3.2.1. Every field is supplied, which keeps the comparison to rendering alone —
/// the separate path upstream takes for a partial demo is covered in `adapter::demos`.
#[test]
fn a_demo_renders_every_value_shape_the_way_python_prints_it() {
    let inputs = ["obj", "arr", "flag", "text", "num"];
    let mut signature = Signature::single_input(
        "Read the fields.",
        vec![OutField {
            name: "out".into(),
            desc: "the answer".into(),
            ..Default::default()
        }],
    );
    signature.inputs = inputs
        .iter()
        .map(|name| InField {
            name: (*name).to_owned(),
            kind: FieldKind::opaque_json(),
            ..Default::default()
        })
        .collect();

    let demo = Example::new([
        ("obj", json!({ "a": 1, "b": { "c": [1, 2] } })),
        ("arr", json!([1, "two", true, null])),
        ("flag", json!(true)),
        ("text", json!("plain string")),
        ("num", json!(1.5)),
        ("out", json!("done")),
    ])
    .with_inputs(inputs);

    let (_, turns) = ChatAdapter::default()
        .format(&signature, &[demo], &[])
        .expect("renders");

    // A nested `null` keeps JSON's spelling because `json.dumps` writes it; only the bool,
    // which is a field value in its own right, reaches Python's `str`.
    assert_eq!(
        turns[0].content.text().unwrap(),
        concat!(
            "[[ ## obj ## ]]\n{\"a\": 1, \"b\": {\"c\": [1, 2]}}\n\n",
            "[[ ## arr ## ]]\n[1, \"two\", true, null]\n\n",
            "[[ ## flag ## ]]\nTrue\n\n",
            "[[ ## text ## ]]\nplain string\n\n",
            "[[ ## num ## ]]\n1.5",
        )
    );
    assert_eq!(
        turns[1].content.text().unwrap(),
        "[[ ## out ## ]]\ndone\n\n[[ ## completed ## ]]\n"
    );
}

/// Bootstrapped demos come from real runs, so an unanswered one reaches the adapter. Showing it
/// would teach the model that a request may go unanswered, so dspy leaves it out of the prompt.
#[test]
fn a_demo_with_no_answer_is_dropped_rather_than_half_shown() {
    let unanswered = Example::new([("request", json!("something calm"))]);
    let inputs = [("request", json!("something bold"))];
    let (_, turns) = ChatAdapter::default()
        .format(&signature(), &[unanswered], &inputs)
        .expect("renders");

    assert_eq!(turns.len(), 1, "only the real request survives");
    assert!(
        turns[0]
            .content
            .text()
            .unwrap()
            .starts_with("[[ ## request ## ]]\nsomething bold")
    );
}

/// A demo that answered with a null did produce a turn, so it reaches the prompt — flagged, and
/// ahead of the demos that have nothing to apologise for.
#[test]
fn a_partial_demo_is_flagged_and_leads_the_whole_ones() {
    let partial = Example::new([
        ("request", json!("something calm")),
        ("colour", json!(null)),
    ]);
    let inputs = [("request", json!("something bold"))];
    let demos = [demo("something warm", "amber"), partial];
    let (_, turns) = ChatAdapter::default()
        .format(&signature(), &demos, &inputs)
        .expect("renders");

    assert_eq!(
        turns[0].content.text().unwrap(),
        "This is an example of the task, though some input or output fields are not supplied.\
         \n\n[[ ## request ## ]]\nsomething calm"
    );
    assert_eq!(
        turns[1].content.text().unwrap(),
        "[[ ## colour ## ]]\nNone\n\n[[ ## completed ## ]]\n"
    );
    assert_eq!(
        turns[2].content.text().unwrap(),
        "[[ ## request ## ]]\nsomething warm"
    );
}

#[test]
fn the_input_label_split_is_what_an_evaluator_would_use() {
    let example = demo("something calm", "blue");
    assert_eq!(
        example.inputs().unwrap().rendered(),
        vec![("request".to_owned(), "something calm".to_owned())]
    );
    assert_eq!(
        example.labels().unwrap().rendered(),
        vec![("colour".to_owned(), "blue".to_owned())]
    );
}
