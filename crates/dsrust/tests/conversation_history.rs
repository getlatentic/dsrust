//! A `History` input against dspy's own rendering, and the two substitutions it is easy to confuse.
//!
//! Both strings were transcribed from `adapters/base.py` and neither was held by a golden, because
//! no fixture rendered a history at all. One of them was wrong: replaying an exchange passes *no*
//! `missing_field_message`, so a field the exchange never recorded substitutes Python's `None` and
//! renders as four letters. The sentence the crate used instead is reachable only from
//! `format_demos`'s complete branch, where nothing is missing.
//!
//! The other substitution — an incomplete demo's "Not supplied for this particular example. " — is
//! real, and sits here beside it so the pair cannot drift back together.

use dsrust::adapter::Input;
use dsrust::signature::Signature;
use dsrust::{Adapter, ChatAdapter, Example};
use serde_json::{Value, json};

fn fixture() -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/conformance/history/conversation_history.json");
    let text = std::fs::read_to_string(&path).expect("the history golden is committed");
    serde_json::from_str(&text).expect("the golden parses")
}

/// The turns an adapter produced, as `(role, content)` — the shape the golden records.
fn turns(signature: &Signature, demos: &[Example], values: &[Input<'_>]) -> Vec<(String, String)> {
    let rendered = ChatAdapter::default()
        .format(signature, demos, values)
        .expect("renders");
    // The system message leads; the golden records the turns after it.
    rendered[1..]
        .iter()
        .map(|turn| {
            (
                turn.role.as_str().to_owned(),
                turn.text().unwrap_or_default(),
            )
        })
        .collect()
}

fn agrees(ours: &[(String, String)], theirs: &Value, what: &str) {
    let recorded = theirs.as_array().expect("turns");
    assert_eq!(ours.len(), recorded.len(), "{what}: turn count");
    for (index, (ours, theirs)) in ours.iter().zip(recorded).enumerate() {
        assert_eq!(
            ours.0,
            theirs["role"].as_str().expect("role"),
            "{what}: role of turn {index}"
        );
        assert_eq!(
            ours.1,
            theirs["content"].as_str().expect("content"),
            "{what}: content of turn {index}"
        );
    }
}

/// A field an exchange never recorded renders as `None`, whatever it is annotated — the crate wrote
/// a sentence there, which is a different string from a different call site.
#[test]
fn a_history_exchange_substitutes_pythons_none_for_what_it_never_recorded() {
    let fixture = fixture();
    let signature: Signature =
        // `dspy.History`, not `History`: upstream evaluates an annotation with `dspy` in scope and
        // nothing else of its own, so the bare name is `ValueError: Unknown name: History`. This
        // read `History` until the two parsers were diffed over thirty-five strings.
        "question: str, history: dspy.History -> answer: str, tags: list[str], score: int"
            .parse()
            .expect("parses");
    let signature = signature.with_instructions("Answer the question.");

    let exchanges = fixture["history"]["exchanges"].clone();
    let values = [
        Input::new("question", json!("capital of France?")),
        Input::new("history", json!({ "messages": exchanges })),
    ];

    let ours = turns(&signature, &[], &values);
    agrees(&ours, &fixture["history"]["turns"], "history");

    // The golden has to *contain* a substitution, or it would pass on a renderer that drops the
    // field entirely — which is what the crate did when handed no message at all.
    assert!(
        ours.iter().any(|(_, content)| content.contains("\nNone")),
        "the golden no longer exercises a missing history field"
    );
}

/// The substitution that is real: an incomplete demo announces itself and names what it lacks.
#[test]
fn an_incomplete_demo_keeps_its_own_missing_field_message() {
    let fixture = fixture();
    let signature: Signature = "question: str -> answer: str, confidence: str"
        .parse()
        .expect("parses");
    let signature = signature.with_instructions("Answer the question.");

    let demos: Vec<Example> = fixture["demos"]["examples"]
        .as_array()
        .expect("examples")
        .iter()
        .map(|demo| {
            Example::new(
                demo.as_object()
                    .expect("a demo is an object")
                    .iter()
                    .map(|(name, value)| (name.as_str(), value.clone())),
            )
        })
        .collect();

    let ours = turns(
        &signature,
        &demos,
        &[Input::new("question", json!("capital of France?"))],
    );
    agrees(&ours, &fixture["demos"]["turns"], "demos");
    assert!(
        ours.iter()
            .any(|(_, content)| content.contains("Not supplied for this particular example.")),
        "the golden no longer exercises an incomplete demo"
    );
}
