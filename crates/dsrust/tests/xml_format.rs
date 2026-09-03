//! What `XmlAdapter` writes for a signature with structured outputs, against dspy's own
//! `XMLAdapter.format` on the pin.
//!
//! dspy 3.3.1 writes a `list`, `dict`, TypedDict or model output as nested elements, sketches
//! those elements in the system prompt and in the request, and escapes `&` and `<` in a `str`
//! output. None of the chat fixtures can see any of it, so this golden renders each case's demos
//! and inputs through the XML adapter and compares every message on the wire.

use dsrust::adapter::Input;
use dsrust::lm::api::LmRequest;
use dsrust::signature::Signature;
use dsrust::{Adapter, Example, XmlAdapter};
use serde_json::Value;

fn fixture() -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/conformance/adapter/xml_format.json");
    let text = std::fs::read_to_string(&path).expect("the XML format golden is committed");
    serde_json::from_str(&text).expect("the golden parses")
}

#[test]
fn the_xml_adapter_writes_what_dspys_writes() {
    let fixture = fixture();
    let cases = fixture["cases"].as_array().expect("cases");
    assert!(!cases.is_empty(), "the golden records no cases");
    let mut nested = 0;
    for case in cases {
        let name = case["name"].as_str().expect("a name");
        let signature: Signature = case["signature"]
            .as_str()
            .expect("a signature")
            .parse()
            .unwrap_or_else(|error| panic!("case {name}: signature does not parse: {error}"));
        let demos: Vec<Example> = case["demos"]
            .as_array()
            .expect("demos")
            .iter()
            .map(|demo| {
                Example::new(
                    demo.as_object()
                        .expect("a demo object")
                        .iter()
                        .map(|(field, value)| (field.clone(), value.clone())),
                )
            })
            .collect();
        let inputs: Vec<Input<'_>> = case["inputs"]
            .as_object()
            .expect("inputs")
            .iter()
            .map(|(field, value)| Input::new(field.as_str(), value.clone()))
            .collect();
        let rendered = XmlAdapter::default()
            .format(&signature, &demos, &inputs)
            .unwrap_or_else(|error| panic!("case {name}: does not render: {error}"));
        let wire = LmRequest::new("", rendered).wire_messages();
        let expected = case["messages"].as_array().expect("messages");
        assert_eq!(wire.len(), expected.len(), "case {name}: message count");
        for (index, (ours, theirs)) in wire.iter().zip(expected).enumerate() {
            assert_eq!(
                ours["role"], theirs["role"],
                "case {name}: message {index} role"
            );
            assert_eq!(
                ours["content"].as_str().unwrap_or_default(),
                theirs["content"].as_str().expect("content"),
                "case {name}: message {index} ({}) differs",
                theirs["role"]
            );
        }
        if expected.iter().any(|message| {
            message["content"]
                .as_str()
                .is_some_and(|c| c.contains("<item>"))
        }) {
            nested += 1;
        }
    }
    assert!(
        nested > 0,
        "no case exercises nested XML, which is what this golden is for"
    );
}
