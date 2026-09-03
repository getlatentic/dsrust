//! The `/embeddings` request `dspy.Embedder("openai/...")` makes, held to what litellm put on the
//! wire — `tests/conformance/lm_api/embedding_wire.json`, recorded at the HTTP layer by
//! `scripts/generate_embedding_wire_fixture.py`.

use dsrust::lm::openai::embeddings::{embeddings_of, request_body};
use serde_json::{Map, Value, json};

fn fixture() -> Vec<Value> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/conformance/lm_api/embedding_wire.json"
    );
    let fixture: Value =
        serde_json::from_str(&std::fs::read_to_string(path).expect("committed")).expect("parses");
    fixture["cases"].as_array().expect("cases").clone()
}

fn model_id(model: &str) -> &str {
    model.split_once('/').map_or(model, |(_, id)| id)
}

#[test]
fn the_body_is_litellms_byte_for_byte() {
    for case in fixture() {
        let inputs: Vec<String> = case["inputs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_owned())
            .collect();
        let kwargs: Map<String, Value> = case["kwargs"].as_object().cloned().unwrap_or_default();
        let ours = request_body(model_id(case["model"].as_str().unwrap()), &inputs, &kwargs);
        assert_eq!(
            serde_json::to_string(&ours).unwrap(),
            serde_json::to_string(&case["body"]).unwrap(),
            "{}: the body, keys in litellm's order",
            case["label"]
        );
        assert_eq!(
            case["url"],
            json!("https://api.openai.com/v1/embeddings"),
            "{}: the default endpoint",
            case["label"]
        );
        assert_eq!(
            case["bearer"],
            json!("Bearer sk-recorded"),
            "{}: bearer auth from the key",
            case["label"]
        );
    }
}

#[test]
fn the_reply_reads_back_as_one_vector_per_input() {
    for case in fixture() {
        let n = case["inputs"].as_array().unwrap().len();
        let reply = json!({
            "object": "list",
            "data": (0..n).map(|i| json!({ "object": "embedding", "index": i, "embedding": [0.1 * (i as f64 + 1.0), 0.2, 0.3] })).collect::<Vec<_>>(),
            "model": model_id(case["model"].as_str().unwrap()),
            "usage": { "prompt_tokens": 0, "total_tokens": 0 }
        });
        let vectors = embeddings_of(&reply).expect("reads");
        let theirs: Vec<Vec<f32>> = case["vectors"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| {
                row.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap() as f32)
                    .collect()
            })
            .collect();
        assert_eq!(vectors, theirs, "{}", case["label"]);
    }
    assert!(embeddings_of(&json!({ "error": "no" })).is_err());
}
