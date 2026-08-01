//! The upstream project's own test suite, replayed.
//!
//! Every call its fourteen test files make to `repair_json` — the input, the keyword arguments, and
//! what came back — recorded while the suite ran rather than copied out of its source. Those cases
//! were written by the people who wrote the heuristics and grown one bug report at a time, which
//! makes them better than any list assembled by reading the library once.
//!
//! What is *not* checked here is whether upstream's suite passes; it is their suite, and their
//! assertions are about their code. The calls are the point.

use json_repair::Repair;
use serde_json::Value as Json;

fn fixture() -> Json {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/conformance/json_repair_upstream.json");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "{}: {error} — run scripts/generate_json_repair_upstream_fixture.py",
            path.display()
        )
    });
    serde_json::from_str(&text).expect("the fixture is JSON")
}

/// The keyword arguments the call was made with. `return_objects` and `logging` decide which entry
/// point answers rather than how it behaves, so they are read at the call site instead.
fn options(case: &Json) -> Repair {
    let flag = |name: &str| case["kwargs"][name].as_bool().unwrap_or(false);
    let ascii = case["kwargs"]["ensure_ascii"].as_bool().unwrap_or(true);
    Repair::new()
        .strict(flag("strict"))
        .stream_stable(flag("stream_stable"))
        .skip_json_loads(flag("skip_json_loads"))
        .ensure_ascii(ascii)
}

#[test]
fn every_call_upstreams_tests_make_answers_the_same_way_here() {
    let fixture = fixture();
    let cases = fixture["cases"].as_array().expect("cases");
    let (mut checked, mut refused, mut logged) = (0, 0, 0);

    for (index, case) in cases.iter().enumerate() {
        let input = case["input"].as_str().expect("an input");
        let repair = options(case);
        let where_ = format!("call {index}, input {:.120?}", input);

        if !case["ok"].as_bool().expect("ok") {
            assert!(
                repair.loads(input).is_err(),
                "{where_}\n  json_repair raised {}, we returned a value",
                case["message"]
            );
            refused += 1;
            continue;
        }

        let result = &case["result"];
        if let Some(expected) = result["log"].as_array() {
            // `logging=True` answers with the value *and* every repair that produced it.
            let (value, log) = repair
                .loads_logged(input)
                .unwrap_or_else(|error| panic!("{where_}\n  we refused: {error}"));
            assert_eq!(
                value.to_string(),
                result["value"].as_str().expect("a value"),
                "{where_}"
            );
            let ours: Vec<Json> = log
                .into_iter()
                .map(|entry| serde_json::json!({ "text": entry.text, "context": entry.context }))
                .collect();
            assert_eq!(&ours, expected, "{where_}\n  the repairs differ");
            logged += ours.len();
        } else if let Some(expected) = result["value"].as_str() {
            // `return_objects=True`: the value, compared as the bytes `json.dumps` writes.
            let value = repair
                .loads(input)
                .unwrap_or_else(|error| panic!("{where_}\n  we refused: {error}"));
            assert_eq!(value.to_string(), expected, "{where_}");
        } else {
            // The default: the repaired text.
            let text = repair
                .repair_json(input)
                .unwrap_or_else(|error| panic!("{where_}\n  we refused: {error}"));
            assert_eq!(text, result["text"].as_str().expect("a text"), "{where_}");
        }
        checked += 1;
    }

    // Floors at what the recording holds, so a fixture that lost its cases fails rather than
    // passing quietly.
    assert!(checked >= 331, "only {checked} calls answered");
    assert!(
        refused >= 18,
        "only {refused} refusals — the raising half is not being checked"
    );
    assert!(logged >= 40, "only {logged} logged repairs");
    eprintln!(
        "  {checked} calls and {refused} refusals from {}",
        fixture["source"]
    );
}
