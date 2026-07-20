//! An `Example` used as a demo travels all the way into the prompt.
//!
//! The unit tests check the split and the rendering separately; this checks the thing a
//! caller actually does — build labelled examples, hand them to a module, and have the model
//! see them as solved turns before the real request.

use dsrs::lm::Role;
use dsrs::signature::{FieldKind, OutField, Signature};
use dsrs::{Adapter, ChatAdapter, Example};
use serde_json::json;

fn signature() -> Signature {
    Signature::single_input(
        "Pick a colour.",
        vec![OutField {
            name: "colour",
            desc: "the chosen colour".into(),
            kind: FieldKind::Str,
            values: None,
            schema: None,
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
    let inputs = [("request", "something bold".to_owned())];
    let (_, turns) = ChatAdapter::default().format(&signature(), &demos, &inputs);

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

    assert_eq!(turns[0].content, "[[ ## request ## ]]\nsomething calm");
    assert_eq!(
        turns[1].content,
        "[[ ## colour ## ]]\nblue\n\n[[ ## completed ## ]]\n"
    );
    assert!(
        turns[4]
            .content
            .starts_with("[[ ## request ## ]]\nsomething bold")
    );

    // Only the real ask carries the format reminder; a demo already shows the answer.
    assert!(
        !turns[0]
            .content
            .contains("Respond with the corresponding output fields")
    );
    assert!(
        turns[4]
            .content
            .contains("Respond with the corresponding output fields")
    );
}

#[test]
fn a_demo_missing_a_field_renders_the_fields_it_has() {
    // Bootstrapped demos come from real runs, so a partial one must degrade rather than panic.
    let partial = Example::new([("request", json!("something calm"))]);
    let inputs = [("request", "something bold".to_owned())];
    let (_, turns) = ChatAdapter::default().format(&signature(), &[partial], &inputs);

    assert_eq!(turns[0].content, "[[ ## request ## ]]\nsomething calm");
    assert_eq!(turns[1].content, "[[ ## completed ## ]]\n");
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
