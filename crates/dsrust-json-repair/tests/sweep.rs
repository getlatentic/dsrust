//! Five hundred drawn inputs, and every repair `json_repair` logged answering them.
//!
//! `tests/conformance.rs` holds the cases someone named, which is what pins the branches a reader
//! went looking for. This holds the ones nobody did — the same grammar `scripts/fuzz_json_repair.py`
//! fuzzes with, at a fixed seed, recorded as a golden.
//!
//! It exists because the campaign corpus is not in git and should not be, which used to mean that
//! the crate's strongest oracle was absent from every mutation run: `tests/fuzz.rs` skips when the
//! corpus is missing, and cargo-mutants copies a tree that never has it. `lookahead.rs` lost 131 of
//! its 139 viable mutants that way — a file of pure lookahead helpers no hand-named corpus reaches.

use json_repair::Repair;
use serde_json::Value as Json;

fn fixture() -> Json {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/conformance/json_repair_sweep.json");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "{}: {error} — run scripts/generate_json_repair_sweep_fixture.py",
            path.display()
        )
    });
    serde_json::from_str(&text).expect("the fixture is JSON")
}

#[test]
fn every_drawn_input_repairs_and_logs_the_way_json_repair_does() {
    let fixture = fixture();
    let cases = fixture["cases"].as_array().expect("cases");
    let mut logged = 0;

    for (index, case) in cases.iter().enumerate() {
        let input = case["input"].as_str().expect("an input");
        let expected = case["dumps"].as_str().expect("a dumps");
        let (ours, log) = Repair::new()
            .loads_logged(input)
            .unwrap_or_else(|error| panic!("case {index}: {input:?}\n  we refused: {error}"));
        assert_eq!(ours.to_string(), expected, "case {index}: {input:?}");

        // The log is the half that says *which* rule got there. Compared entry by entry, context
        // window included, so a branch reached by a different route is a failure and not a
        // coincidence.
        let ours: Vec<Json> = log
            .into_iter()
            .map(|entry| serde_json::json!({ "text": entry.text, "context": entry.context }))
            .collect();
        assert_eq!(
            &ours,
            case["log"].as_array().expect("a log"),
            "case {index}: {input:?}\n  the repairs differ"
        );
        logged += ours.len();
    }

    // Floors at what the fixture holds: a sweep that shrank, or whose grammar stopped producing
    // malformed input, would otherwise still pass.
    assert!(cases.len() >= 620, "the sweep is {} cases", cases.len());
    assert!(
        logged >= 2842,
        "only {logged} logged repairs across the sweep"
    );
    eprintln!(
        "  {} drawn inputs, {logged} logged repairs, from {}",
        cases.len(),
        fixture["source"]
    );
}
