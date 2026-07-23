//! `dsrs::Reasoning` declared in a Rust signature, rendered against dspy 3.3's own bytes.
//!
//! The bridge proves the crate renders what upstream's tests assert; this proves the *API* a Rust
//! caller writes reaches the same place. A field declared `Reasoning` must print `(str)`, carry no
//! schema note, and still earn the output-requirement hint — dspy decides that last one by asking
//! whether the annotation is `str`, and `Reasoning` is not.
//!
//! The expected strings are upstream's, from
//! `test_chat_adapter_format_exact_messages_with_reasoning_and_code_outputs`.

use dsrs::signature::{FieldKind, SignatureSpec};
use dsrs::{Adapter, ChatAdapter, Reasoning, Signature};

// Declared for its signature, as the other derive tests are; nothing reads the fields back.
#[allow(dead_code)]
#[derive(Signature)]
/// Answer the question.
struct Explain {
    #[input]
    question: String,
    #[output]
    reasoning: Reasoning,
    #[output]
    answer: String,
}

#[test]
fn a_reasoning_field_declares_its_own_kind() {
    let signature = Explain::signature();
    assert_eq!(signature.outputs[0].kind, FieldKind::Reasoning);
    // dspy's `get_annotation_name` prints "str" for it, keeping ChainOfThought's old wording.
    assert_eq!(signature.outputs[0].annotation(), "str");
    // A plain `String` output beside it stays the plain str it is.
    assert_eq!(signature.outputs[1].kind, FieldKind::Str);
}

#[test]
fn it_reads_as_str_in_the_field_list_and_carries_no_schema_note() {
    let system = ChatAdapter::new()
        .system_message(&Explain::signature())
        .expect("renders");
    // dspy: "1. `reasoning` (str): " — the type prints as str and states nothing further.
    assert!(
        system.contains("1. `reasoning` (str): "),
        "reasoning should read as str, got:\n{system}"
    );
    // A str-like type formats as its content, so no JSON-schema note follows the slot.
    assert!(
        !system.contains("adhere to the JSON schema"),
        "a str-like type states no schema, got:\n{system}"
    );
}

#[test]
fn it_still_earns_the_output_requirement_hint() {
    let (_system, turns) = ChatAdapter::new()
        .format(
            &Explain::signature(),
            &[],
            &[dsrs::adapter::Input::new("question", serde_json::json!("Q"))],
        )
        .expect("renders");
    let last = turns.last().expect("a user turn").content.text().unwrap_or_default().to_owned();
    // dspy asks `annotation is not str`; Reasoning is not, so the hint stays — named "str".
    assert!(
        last.contains("`[[ ## reasoning ## ]]` (must be formatted as a valid Python str)"),
        "reasoning should keep the hint, got:\n{last}"
    );
    // The plain String output beside it is genuinely `str`, so it takes no hint.
    assert!(
        last.contains("then `[[ ## answer ## ]]`, and then ending"),
        "a plain str output takes no hint, got:\n{last}"
    );
}

#[test]
fn a_reasoning_value_round_trips_as_the_text_it_carries() {
    // dspy's validator accepts a bare string; its `format` yields the content back.
    let reasoning: Reasoning = "I checked the units.".into();
    assert_eq!(reasoning.format(), "I checked the units.");
    assert_eq!(reasoning, "I checked the units.");
    // It reaches a prompt as that text, not as an object.
    assert_eq!(
        serde_json::to_value(&reasoning).expect("serializes"),
        serde_json::json!("I checked the units.")
    );
}
