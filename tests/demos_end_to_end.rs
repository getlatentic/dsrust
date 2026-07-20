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

/// The expected bytes are what `dspy.ChatAdapter().format` emits for this signature and demo
/// on dspy 3.2.1. Every field is supplied, which is what keeps the comparison to rendering:
/// upstream routes a demo with a missing or `None` field down a separate path that prefixes
/// the turn, and this crate does not draw that distinction yet.
#[test]
fn a_demo_renders_every_value_shape_the_way_python_prints_it() {
    let inputs = ["obj", "arr", "flag", "text", "num"];
    let mut signature = Signature::single_input(
        "Read the fields.",
        vec![OutField {
            name: "out".into(),
            desc: "the answer".into(),
            kind: FieldKind::Str,
            values: None,
            schema: None,
        }],
    );
    signature.inputs = inputs
        .iter()
        .map(|name| InField {
            name: (*name).to_owned(),
            desc: String::new(),
            kind: FieldKind::opaque_json(),
            values: None,
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

    let (_, turns) = ChatAdapter::default().format(&signature, &[demo], &[]);

    // A nested `null` keeps JSON's spelling because `json.dumps` writes it; only the bool,
    // which is a field value in its own right, reaches Python's `str`.
    assert_eq!(
        turns[0].content,
        concat!(
            "[[ ## obj ## ]]\n{\"a\": 1, \"b\": {\"c\": [1, 2]}}\n\n",
            "[[ ## arr ## ]]\n[1, \"two\", true, null]\n\n",
            "[[ ## flag ## ]]\nTrue\n\n",
            "[[ ## text ## ]]\nplain string\n\n",
            "[[ ## num ## ]]\n1.5",
        )
    );
    assert_eq!(
        turns[1].content,
        "[[ ## out ## ]]\ndone\n\n[[ ## completed ## ]]\n"
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
