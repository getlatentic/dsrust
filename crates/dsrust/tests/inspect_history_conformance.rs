//! dspy `pretty_print_history`, against what upstream printed.
//!
//! Both colour settings, from the same recorded call: upstream's `file=` argument is its own
//! no-colour path, so the two are not a Rust invention.

use dsrust::lm::api::pretty_print_history;
use serde_json::Value;

#[test]
fn every_recorded_history_renders_the_same_text() {
    let golden: Value =
        serde_json::from_str(include_str!("conformance/history/inspect_history.json"))
            .expect("the inspect-history golden is valid JSON");
    let cases = golden["cases"].as_array().expect("cases");
    assert!(cases.len() >= 10, "the golden lost cases: {}", cases.len());
    for case in cases {
        let name = case["name"].as_str().expect("a name");
        let history: Vec<Value> = case["history"].as_array().expect("history").clone();
        let n = case["n"].as_u64().expect("n") as usize;
        for (colours, key) in [(false, "plain"), (true, "coloured")] {
            assert_eq!(
                pretty_print_history(&history, n, colours),
                case[key].as_str().expect("recorded text"),
                "{name}, {key}"
            );
        }
    }
}
