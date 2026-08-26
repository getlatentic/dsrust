//! Which annotations may keep a bare, non-JSON string, held to what pydantic actually accepts.
//!
//! An adapter reading a reply gets text, and for most annotations that text must be JSON. For some
//! the text *is* the value: a `datetime` is `2024-01-01`, a `dspy.Code` is the code. Casting the
//! second kind as JSON rejects a reply dspy accepts — which is what `dspy.Code` did here, taking
//! `test_json_adapter_with_code` and `test_baml_adapter_with_code` red in the upstream suite. They
//! were red at HEAD; the local gate does not run that suite, so nothing said so.
//!
//! The set is compared against a generated corpus rather than a list written twice, because a list
//! written twice agrees with itself. `scripts/generate_string_form_fixture.py`.

use dsrust::adapter::{Adapter, JsonAdapter};
use dsrust::signature::{FieldKind, InField, JsonType, OutField, Signature};
use serde_json::Value;

fn fixture() -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/conformance/adapter/string_form.json");
    serde_json::from_str(&std::fs::read_to_string(&path).expect("the corpus is committed"))
        .expect("it parses")
}

/// `question -> field`, where `field` carries the annotation under test.
fn signature_for(annotation: &str) -> Signature {
    Signature {
        instructions: "Answer.".into(),
        inputs: vec![InField {
            name: "question".into(),
            ..Default::default()
        }],
        outputs: vec![OutField {
            name: "field".into(),
            kind: FieldKind::Json(JsonType::plain(annotation)),
            ..Default::default()
        }],
    }
}

/// Whether a reply naming this field with a bare string parses, through the adapter a caller uses.
fn keeps_a_bare_string(annotation: &str, probe: &str) -> bool {
    let reply = serde_json::json!({ "field": probe }).to_string();
    JsonAdapter::default()
        .parse(&signature_for(annotation), &reply)
        .is_ok()
}

/// A type with *any* string form keeps a bare string; a type with none is asked for JSON.
///
/// The predicate is either corpus column, not the arbitrary one, and that is the coercion split
/// rather than a looseness: a `date` answered `print('hi')` is not rejected here, because what
/// refuses it is the caller's typing (or, across the bridge, dspy's own `parse_value`). The cast's
/// job is to tell a value that *is* text from one that should have been JSON, and a `date` is the
/// first. A type with no string form at all — an `Image`, a `ToolCalls` — has no such excuse.
///
/// Both directions, from one corpus. Only checking the first would pass a crate that kept every
/// bare string, which is a port with no casting at all.
#[test]
fn the_crate_keeps_exactly_what_pydantic_accepts() {
    let fixture = fixture();
    let probe = fixture["arbitrary_probe"].as_str().expect("a probe");
    let (mut kept, mut cast) = (0, 0);

    for row in fixture["rows"].as_array().expect("rows") {
        let name = row["annotation_name"].as_str().expect("a name");
        let accepts = row["accepts_arbitrary"].as_bool().expect("a verdict")
            || row["accepts_well_formed"].as_bool().expect("a verdict");
        // `str` is not a structured annotation and never reaches the JSON cast at all.
        if name == "str" {
            continue;
        }
        assert_eq!(
            keeps_a_bare_string(name, probe),
            accepts,
            "{name}: pydantic {} a string form and the crate {}",
            if accepts { "has" } else { "has no" },
            if accepts { "rejected one" } else { "kept one" },
        );
        if accepts { kept += 1 } else { cast += 1 }
    }
    assert_eq!(
        kept, 6,
        "Code, Code_java and the four temporal types have a string form"
    );
    assert!(cast >= 4, "only {cast} annotations require JSON");
}

/// A type that accepts only its own spelling still keeps the bare string, because what refuses a
/// bad one is the caller's typing rather than this cast.
///
/// The distinction the corpus's second column exists for: a `datetime` field answered `not a date`
/// is not rejected here, and a port that "fixed" that by requiring JSON would break every date.
#[test]
fn a_temporal_type_keeps_its_own_form() {
    let fixture = fixture();
    let mut checked = 0;
    for row in fixture["rows"].as_array().expect("rows") {
        let name = row["annotation_name"].as_str().expect("a name");
        if row["accepts_arbitrary"].as_bool().expect("a verdict")
            || !row["accepts_well_formed"].as_bool().expect("a verdict")
        {
            continue;
        }
        let probe = row["well_formed_probe"].as_str().expect("a probe");
        assert!(
            keeps_a_bare_string(name, probe),
            "{name} should keep {probe:?} for the caller's typing to read"
        );
        checked += 1;
    }
    assert_eq!(checked, 4, "the four temporal types should be here");
}

/// The reply that took two upstream tests red: a `Code` output answered as a bare string.
#[test]
fn a_code_field_reads_back_from_a_bare_string() {
    for annotation in ["Code", "Code_java"] {
        let reply = serde_json::json!({ "field": "print(\"Hello, world!\")" }).to_string();
        let parsed = JsonAdapter::default()
            .parse(&signature_for(annotation), &reply)
            .unwrap_or_else(|error| panic!("{annotation} takes a bare string: {error}"));
        assert_eq!(
            parsed["field"],
            serde_json::json!("print(\"Hello, world!\")")
        );
    }
}
