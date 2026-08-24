//! The predicate dspy's JSON stream listener decides on, against `jiter`'s own answers.
//!
//! `dsrust-json-repair` cannot stand in for this: it reproduces Python's `json_repair`, a different
//! library, and the two disagree exactly where the decision is made — `{"answer": "x", "judgement":`
//! is one key to jiter and two to the repairer, which would close a streamed field a delta early.
//! Four other accumulated shapes agree, which is what makes it worth pinning rather than eyeballing.
//!
//! Every *prefix* is compared, because a listener walks them all in turn as deltas arrive, and what
//! matters is that the predicate flips on the same one.

use std::path::Path;

use serde_json::Value;

fn golden() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/conformance/lm/partial_json.json");
    serde_json::from_str(&std::fs::read_to_string(&path).expect("the golden is committed"))
        .expect("the golden parses")
}

#[test]
fn the_keys_seen_in_a_half_written_object_are_jiters() {
    let recorded = golden();
    let cases = recorded["cases"].as_array().expect("cases");
    let mut compared = 0;
    for case in cases {
        for prefix in case["prefixes"].as_array().expect("prefixes") {
            let text = prefix["text"].as_str().expect("text");
            let theirs: Option<Vec<String>> = prefix["keys"].as_array().map(|keys| {
                keys.iter()
                    .map(|key| key.as_str().expect("a key").to_owned())
                    .collect()
            });
            let ours = dsrust::adapter::stream::keys_with_values(text);
            assert_eq!(
                ours, theirs,
                "case `{}` diverges at prefix {text:?}",
                case["name"]
            );
            compared += 1;
        }
    }
    assert!(compared > 300, "only {compared} prefixes compared");
}

#[test]
fn a_complete_object_is_recognised_as_complete() {
    for case in golden()["cases"].as_array().expect("cases") {
        for prefix in case["prefixes"].as_array().expect("prefixes") {
            let text = prefix["text"].as_str().expect("text");
            assert_eq!(
                dsrust::adapter::stream::is_complete(text),
                prefix["complete"].as_bool().expect("complete"),
                "case `{}` disagrees on completeness at {text:?}",
                case["name"]
            );
        }
    }
}
