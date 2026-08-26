//! The byte scanner and the char scanner are one grammar written twice, so they are held to one
//! answer here — over every committed input, valid and malformed alike, since a refusal is half of
//! what a scanner decides. A parallel implementation that can drift silently is worse than a slow
//! one.

use serde_json::Value as Json;

fn committed_inputs() -> Vec<String> {
    let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/conformance");
    let mut inputs = Vec::new();
    for name in [
        "json_repair.json",
        "json_repair_sweep.json",
        "json_repair_upstream.json",
        "json_repair_schema.json",
        "json_repair_bytes.json",
    ] {
        let text = std::fs::read_to_string(base.join(name)).expect("committed fixture");
        let fixture: Json = serde_json::from_str(&text).expect("fixture parses");
        for case in fixture["cases"].as_array().expect("cases") {
            if let Some(input) = case["input"].as_str() {
                inputs.push(input.to_owned());
            }
        }
    }
    inputs
}

#[test]
fn both_scanners_answer_every_committed_input_identically() {
    let inputs = committed_inputs();
    assert!(inputs.len() > 1500, "only {} inputs", inputs.len());
    let (mut accepted, mut refused) = (0, 0);
    for input in &inputs {
        let of_chars = json_repair::strict_scan_chars_for_tests(input);
        let of_bytes = json_repair::strict_scan_bytes_for_tests(input);
        match (of_chars, of_bytes) {
            (Some(left), Some(right)) => {
                assert_eq!(left, right, "the scanners parsed {input:?} differently");
                accepted += 1;
            }
            (None, None) => refused += 1,
            (left, right) => panic!(
                "one scanner refused {input:?}: chars {:?}, bytes {:?}",
                left.is_some(),
                right.is_some()
            ),
        }
    }
    // Both arms, or the comparison holds one grammar to nothing. The corpus is deliberately
    // malformed-heavy, so the accepted floor sits under the 184 measured, not at half.
    assert!(
        accepted > 150 && refused > 200,
        "{accepted} accepted, {refused} refused — the corpus stopped exercising both"
    );
    eprintln!("  {accepted} accepted and {refused} refused, identically");
}
