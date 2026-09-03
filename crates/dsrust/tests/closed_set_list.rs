//! A closed set on a list of strings renders as dspy's `list[Literal[...]]`.
//!
//! Two spellings reach a prompt as a closed set and they are not the same rendering: a scalar
//! `Literal` prints its members on the field line and asks the model to match one exactly, while a
//! *list* of them prints `list[Literal[...]]` as the annotation and puts the members inside the
//! schema's `items`. dspy has no field-level closed set at all — `Literal` is part of the
//! annotation — so a list's set has to live in the same two places here.
//!
//! Found by porting DSPy's `gepa_facilitysupportanalyzer` tutorial, whose `categories` field is
//! exactly this shape. The derive rejected it outright: `values(...)` was allowed on `String` and
//! nothing else, so the tutorial could not be written at all.

use dsrust::Signature;
use dsrust::adapter::{Adapter, ChatAdapter, Input};
use dsrust::signature::SignatureSpec;

#[derive(Signature)]
/// Read the provided message and determine the set of categories applicable to the message.
struct Categories {
    #[input]
    message: String,
    #[output(values("emergency_repair_services", "routine_maintenance_requests"))]
    categories: Vec<String>,
}

#[derive(Signature)]
/// Read the provided message and determine the urgency.
struct Urgency {
    #[input]
    message: String,
    #[output(values("low", "medium", "high"))]
    urgency: String,
}

fn system_message(signature: &dsrust::signature::Signature) -> String {
    let inputs = [Input::new("message", "the boiler is leaking".into())];
    let messages = ChatAdapter::default()
        .format(signature, &[], &inputs)
        .expect("renders");
    messages[0].text().expect("a system prompt")
}

/// The whole rendering, against what dspy prints for `List[Literal[...]]`.
#[test]
fn a_constrained_list_renders_as_a_list_of_literal() {
    let rendered = system_message(&Categories::signature());
    assert!(
        rendered.contains(
            "1. `categories` (list[Literal['emergency_repair_services', \
             'routine_maintenance_requests']])"
        ),
        "got: {rendered}"
    );
    // The members are in the schema's `items`, and the keys are `move_type_to_front`'s.
    assert!(
        rendered.contains(
            r#"must adhere to the JSON schema: {"type": "array", "items": {"type": "string", "#
        ),
        "got: {rendered}"
    );
    assert!(
        rendered
            .contains(r#""enum": ["emergency_repair_services", "routine_maintenance_requests"]}}"#),
        "got: {rendered}"
    );
}

/// A scalar closed set keeps its own rendering, which is a different one: the members go on the
/// field's note and the annotation stays `Literal[...]` without a container.
#[test]
fn a_scalar_closed_set_is_still_a_bare_literal() {
    let rendered = system_message(&Urgency::signature());
    assert!(
        rendered.contains("1. `urgency` (Literal['low', 'medium', 'high'])"),
        "got: {rendered}"
    );
    assert!(
        rendered.contains("must exactly match (no extra characters) one of: low; medium; high"),
        "got: {rendered}"
    );
    // A scalar's set is not a schema, and the change that gave a list one must not have given
    // this one too.
    assert!(!rendered.contains("JSON schema"), "got: {rendered}");
}
