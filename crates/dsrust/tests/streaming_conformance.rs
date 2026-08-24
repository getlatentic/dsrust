//! `FieldListener` against dspy's `StreamListener`, chunk boundary for chunk boundary.
//!
//! `tests/streaming/test_streaming.py` is excused from the bridge because most of it drives dspy's
//! async `streamify` plumbing. What that excuse hid is the half that is pure logic and is the whole
//! point of a stream listener: **where the chunks fall**. A caller renders what it is handed, so a
//! listener whose text concatenates correctly and whose boundaries do not still splits words down
//! the middle. This crate did exactly that — `["To", " ge", "t to", " the o", …]` where dspy yields
//! one chunk per token — and nothing said so, because nothing compared the boundaries.
//!
//! `scripts/generate_streaming_fixture.py` captures them by running the pinned dspy.

use std::path::Path;

use dsrust::adapter::stream::{FieldListener, JsonFieldListener};
use serde_json::Value;

fn golden() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/conformance/lm/streaming.json");
    serde_json::from_str(&std::fs::read_to_string(&path).expect("the golden is committed"))
        .expect("the golden parses")
}

/// Every recorded stream, compared as the list of `(text, is_last)` a caller receives.
#[test]
fn a_listeners_chunks_are_dspys_chunks() {
    let recorded = golden();
    let cases = recorded["cases"].as_array().expect("cases");
    assert!(!cases.is_empty(), "the golden carries no cases");

    for case in cases {
        let name = case["name"].as_str().expect("a name");
        let field = case["field"].as_str().expect("a field");
        let mut listener = FieldListener::new(field);

        let mut ours: Vec<(String, bool)> = case["deltas"]
            .as_array()
            .expect("deltas")
            .iter()
            .filter_map(|delta| listener.push(delta.as_str().expect("a delta")))
            .map(|chunk| (chunk.text, chunk.is_last))
            .collect();
        // dspy's `finalize`, which `streamify` calls once the model's stream is done.
        if let Some(tail) = listener.finish() {
            ours.push((tail.text, tail.is_last));
        }

        let theirs: Vec<(String, bool)> = case["chunks"]
            .as_array()
            .expect("chunks")
            .iter()
            .map(|chunk| {
                (
                    chunk["text"].as_str().expect("text").to_owned(),
                    chunk["is_last"].as_bool().expect("is_last"),
                )
            })
            .collect();

        assert_eq!(
            ours, theirs,
            "case `{name}` diverges from dspy\n  source: {}",
            recorded["_source"]
        );
    }
}

/// The same, over the XML wire — the same rule with different identifiers, which is why it is the
/// same listener rather than a third one.
///
/// The difference that matters: a tag closes *itself* (`</answer>`), where a marker section closes
/// when the next one opens. So an XML field left unclosed is never ended by whatever follows it.
#[test]
fn an_xml_listeners_chunks_are_dspys_chunks() {
    let recorded = golden();
    let cases = recorded["xml_cases"].as_array().expect("xml_cases");
    assert!(!cases.is_empty(), "the golden carries no XML cases");

    for case in cases {
        let mut listener = FieldListener::xml(case["field"].as_str().expect("a field"));
        let mut ours: Vec<(String, bool)> = case["deltas"]
            .as_array()
            .expect("deltas")
            .iter()
            .filter_map(|delta| listener.push(delta.as_str().expect("a delta")))
            .map(|chunk| (chunk.text, chunk.is_last))
            .collect();
        if let Some(tail) = listener.finish() {
            ours.push((tail.text, tail.is_last));
        }
        let theirs: Vec<(String, bool)> = case["chunks"]
            .as_array()
            .expect("chunks")
            .iter()
            .map(|chunk| {
                (
                    chunk["text"].as_str().expect("text").to_owned(),
                    chunk["is_last"].as_bool().expect("is_last"),
                )
            })
            .collect();
        assert_eq!(
            ours, theirs,
            "case `{}` diverges from dspy\n  source: {}",
            case["name"], recorded["_source"]
        );
    }
}

/// The same, over the JSON wire, where the field ends because the *object* moved on rather than
/// because a marker arrived.
///
/// The chunks carry their quotes — `"To` … `!"` — because dspy streams the field's raw JSON text.
/// That looks like a bug until it is compared, which is the argument for comparing rather than
/// deciding what it ought to be.
#[test]
fn a_json_listeners_chunks_are_dspys_chunks() {
    let recorded = golden();
    let cases = recorded["json_cases"].as_array().expect("json_cases");
    assert!(!cases.is_empty(), "the golden carries no JSON cases");

    for case in cases {
        let name = case["name"].as_str().expect("a name");
        let mut listener = JsonFieldListener::new(case["field"].as_str().expect("a field"));

        let mut ours: Vec<(String, bool)> = case["deltas"]
            .as_array()
            .expect("deltas")
            .iter()
            .filter_map(|delta| listener.push(delta.as_str().expect("a delta")))
            .map(|chunk| (chunk.text, chunk.is_last))
            .collect();
        if let Some(tail) = listener.finish() {
            ours.push((tail.text, tail.is_last));
        }
        let theirs: Vec<(String, bool)> = case["chunks"]
            .as_array()
            .expect("chunks")
            .iter()
            .map(|chunk| {
                (
                    chunk["text"].as_str().expect("text").to_owned(),
                    chunk["is_last"].as_bool().expect("is_last"),
                )
            })
            .collect();
        assert_eq!(
            ours, theirs,
            "case `{name}` diverges from dspy\n  source: {}",
            recorded["_source"]
        );
    }
}

/// Why the boundaries need their own assertion: the divergence this crate had did not change the
/// text by one byte.
///
/// The sequence below is what the previous listener produced for the recorded stream — it buffered
/// a fixed number of *characters* rather than whole deltas. It concatenates to exactly what dspy
/// streams, so any test written over the joined text passed the whole time, while a caller
/// rendering chunk by chunk saw `"To"`, `" ge"`, `"t to"`.
#[test]
fn the_old_boundaries_carried_the_same_text() {
    let was = [
        "To", " ge", "t to", " the o", "ther ", "sid", "e of", " the di", "nner p", "late!",
    ];
    let recorded = golden();
    let case = recorded["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .find(|case| case["name"] == "recorded_gpt_4o_mini")
        .expect("the recorded stream");
    let dspy: Vec<&str> = case["chunks"]
        .as_array()
        .expect("chunks")
        .iter()
        .map(|chunk| chunk["text"].as_str().expect("text"))
        .collect();

    assert_eq!(
        was.concat(),
        dspy.concat(),
        "the old listener streamed the same text"
    );
    assert_ne!(
        was.to_vec(),
        dspy,
        "and different chunks, which is the whole of what was wrong"
    );
}
