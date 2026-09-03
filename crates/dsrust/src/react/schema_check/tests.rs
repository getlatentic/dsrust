//! Replays `tests/conformance/react/schema_messages.json`, recorded from python-jsonschema by
//! `scripts/generate_schema_message_fixture.py`: for every case, the schema schemars emits for the
//! named Rust type is the schema recorded, and the message this module picks is the message
//! `validate` raised.

use std::collections::{BTreeSet, HashMap};

use serde_json::Value;

use super::message;
use crate::signature::json_argument_schema;

fn fixture() -> Value {
    serde_json::from_str(include_str!(
        "../../../tests/conformance/react/schema_messages.json"
    ))
    .expect("the fixture is JSON")
}

#[derive(schemars::JsonSchema)]
#[allow(dead_code)]
struct Inner {
    name: String,
    count: u8,
}

#[derive(schemars::JsonSchema)]
#[allow(dead_code)]
enum Colour {
    Red,
    Green,
}

#[derive(schemars::JsonSchema)]
#[allow(dead_code)]
enum Shape {
    Circle { radius: f64 },
    Square(u32),
}

fn schema_of(rust_type: &str) -> Option<Value> {
    Some(match rust_type {
        "u32" => json_argument_schema::<u32>(),
        "u8" => json_argument_schema::<u8>(),
        "f64" => json_argument_schema::<f64>(),
        "bool" => json_argument_schema::<bool>(),
        "String" => json_argument_schema::<String>(),
        "Option<u32>" => json_argument_schema::<Option<u32>>(),
        "Vec<String>" => json_argument_schema::<Vec<String>>(),
        "Vec<u32>" => json_argument_schema::<Vec<u32>>(),
        "Inner" => json_argument_schema::<Inner>(),
        "Vec<Inner>" => json_argument_schema::<Vec<Inner>>(),
        "Option<Inner>" => json_argument_schema::<Option<Inner>>(),
        "Colour" => json_argument_schema::<Colour>(),
        "Shape" => json_argument_schema::<Shape>(),
        "(u32, String)" => json_argument_schema::<(u32, String)>(),
        "HashMap<String, u32>" => json_argument_schema::<HashMap<String, u32>>(),
        _ => return None,
    })
}

#[test]
fn every_recorded_message_is_the_one_picked_here() {
    let fixture = fixture();
    let cases = fixture["cases"].as_array().expect("cases");
    assert!(cases.len() >= 70, "the fixture holds every case recorded");
    let mut mismatches = Vec::new();
    for case in cases {
        let label = case["label"].as_str().expect("label");
        let expected = case["message"].as_str().map(str::to_owned);
        let picked = message(&case["instance"], &case["schema"]);
        if picked != expected {
            mismatches.push(format!("{label}: expected {expected:?}, picked {picked:?}"));
        }
    }
    assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
}

#[test]
fn every_typed_case_carries_the_schema_schemars_emits() {
    let fixture = fixture();
    for case in fixture["cases"].as_array().expect("cases") {
        let Some(rust_type) = case["rust_type"].as_str() else {
            continue;
        };
        let emitted = schema_of(rust_type)
            .unwrap_or_else(|| panic!("{rust_type} is not a type this test knows"));
        assert_eq!(
            serde_json::to_string(&emitted).expect("serializes"),
            serde_json::to_string(&case["schema"]).expect("serializes"),
            "{rust_type}: the recorded schema is not what schemars emits",
        );
    }
}

/// Every keyword `apply` dispatches on appears in some recorded schema, so none is ported without
/// an oracle case behind it.
#[test]
fn every_ported_keyword_has_a_recorded_case() {
    let source = include_str!("keywords/mod.rs");
    let dispatch = source
        .split("match keyword {")
        .nth(1)
        .expect("the dispatch")
        .split("_ => Vec::new()")
        .next()
        .expect("its arms");
    let ported: BTreeSet<&str> = dispatch
        .lines()
        .filter_map(|line| {
            let (keyword, rest) = line.trim().strip_prefix('"')?.split_once('"')?;
            rest.starts_with(" =>").then_some(keyword)
        })
        .collect();
    let fixture = fixture();
    let mut exercised = BTreeSet::new();
    for case in fixture["cases"].as_array().expect("cases") {
        collect_keywords(&case["schema"], &mut exercised);
    }
    let unexercised: Vec<&&str> = ported
        .iter()
        .filter(|keyword| !exercised.contains(**keyword))
        .collect();
    assert!(ported.len() >= 30, "{} keywords ported", ported.len());
    assert!(
        unexercised.is_empty(),
        "ported without a recorded case: {unexercised:?}"
    );
}

fn collect_keywords(schema: &Value, into: &mut BTreeSet<String>) {
    match schema {
        Value::Object(map) => {
            for (key, value) in map {
                into.insert(key.clone());
                collect_keywords(value, into);
            }
        }
        Value::Array(items) => items.iter().for_each(|item| collect_keywords(item, into)),
        _ => {}
    }
}
