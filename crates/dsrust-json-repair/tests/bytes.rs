//! Input and output compared as bytes, so no encoder between here and Python can hide a difference.
//!
//! Every other fixture holds its cases as JSON strings, which means the comparison passes through
//! two JSON encoders and a Rust `&str`. That is fine until something normalises — a combining mark
//! composed, a byte-order mark eaten, `\r\n` folded — and then the strings match while the bytes do
//! not. These cases are recorded in hex on both sides.

use json_repair::Repair;
use serde_json::Value as Json;

fn fixture() -> Json {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/conformance/json_repair_bytes.json");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "{}: {error} — run scripts/generate_json_repair_bytes_fixture.py",
            path.display()
        )
    });
    serde_json::from_str(&text).expect("the fixture is JSON")
}

fn from_hex(hex: &str) -> Vec<u8> {
    (0..hex.len())
        .step_by(2)
        .map(|at| u8::from_str_radix(&hex[at..at + 2], 16).expect("a hex byte"))
        .collect()
}

fn to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[test]
fn the_bytes_out_are_the_bytes_python_wrote() {
    let fixture = fixture();
    let cases = fixture["cases"].as_array().expect("cases");
    let mut changed = 0;

    for case in cases {
        let name = case["name"].as_str().expect("a name");
        let why = case["why"].as_str().unwrap_or("");
        let input_hex = case["input_hex"].as_str().expect("input hex");
        let input = String::from_utf8(from_hex(input_hex)).expect("the input is UTF-8");

        let ours = Repair::new()
            .ensure_ascii(case["ensure_ascii"].as_bool().expect("ensure_ascii"))
            .repair_json(&input)
            .unwrap_or_else(|error| panic!("{name}: {why}\n  we refused: {error}"));

        let expected = case["output_hex"].as_str().expect("output hex");
        assert_eq!(to_hex(ours.as_bytes()), expected, "{name}: {why}");
        if input_hex != expected {
            changed += 1;
        }
    }

    assert_eq!(
        cases.len(),
        16,
        "the fixture is not the one that was generated"
    );
    assert!(
        changed >= 10,
        "only {changed} cases came back changed — hex would be proving little"
    );
    eprintln!(
        "  {} cases byte-for-byte against {}",
        cases.len(),
        fixture["source"]
    );
}

#[test]
fn a_combining_sequence_and_its_composed_form_stay_apart() {
    // The pair the fixture carries to make normalisation visible. Asserted here as well as through
    // the hex, because a comparison that folded them would still pass every *other* case.
    let decomposed = Repair::new()
        .ensure_ascii(false)
        .repair_json("{\"a\": \"e\u{301}\"}");
    let composed = Repair::new()
        .ensure_ascii(false)
        .repair_json("{\"a\": \"\u{e9}\"}");
    assert_ne!(
        decomposed.expect("repaired"),
        composed.expect("repaired"),
        "the two spellings of é came back the same, so something normalised"
    );
}
