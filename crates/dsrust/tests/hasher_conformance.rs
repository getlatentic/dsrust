//! dspy `Hasher.hash` against digests recorded from `sha256(pickle.dumps(...))` itself.
//!
//! The pickle bytes behind these digests are compared opcode by opcode in `hasher::pickle`'s own
//! tests. What this adds is the public entry point: a caller reaching `Hasher::hash` gets the
//! string dspy would have seeded its generator with.

use dsrust::{Example, Hasher};
use serde_json::Value;

#[test]
fn every_recorded_tuple_hashes_to_what_dspy_hashed_it_to() {
    let golden: Value = serde_json::from_str(include_str!("conformance/optimize/hasher.json"))
        .expect("the hasher golden is valid JSON");
    let cases = golden["cases"].as_array().expect("cases");
    assert!(cases.len() >= 13, "the golden lost cases: {}", cases.len());
    for case in cases {
        assert_eq!(
            Hasher::hash(&demos(case)),
            case["hash"].as_str().expect("hash"),
            "Hasher.hash for {}",
            case["name"].as_str().expect("name")
        );
    }
}

/// dspy `Hasher.hash_bytes`, which chains its chunks into one digest rather than hashing each.
#[test]
fn chunks_are_hashed_as_one_stream() {
    assert_eq!(
        Hasher::hash_bytes(&[b"abc", b"def"]),
        Hasher::hash_bytes(&[b"abcdef"]),
        "the chunk boundary is not part of the digest"
    );
    assert_eq!(
        Hasher::hash_bytes(&[b""]),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "the sha256 of nothing"
    );
}

fn demos(case: &Value) -> Vec<Example> {
    let keys: Vec<String> = case["input_keys"]
        .as_array()
        .expect("input_keys")
        .iter()
        .map(|key| key.as_str().expect("a key").to_owned())
        .collect();
    case["demos"]
        .as_array()
        .expect("demos")
        .iter()
        .map(|fields| {
            Example::new(
                fields
                    .as_object()
                    .expect("a demo is an object")
                    .iter()
                    .map(|(name, value)| (name.clone(), value.clone())),
            )
            .with_inputs(keys.clone())
        })
        .collect()
}
