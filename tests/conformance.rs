//! Conformance against Python DSPy's own adapter tests.
//!
//! Each fixture in `tests/conformance/` is one case lifted from upstream's
//! `tests/adapters/test_chat_adapter.py`, with the `expected` strings copied verbatim from
//! its assertions. Passing means this crate renders the prompt Python DSPy renders, so the
//! question "are we faithful?" is answered by the test run instead of by reading both
//! codebases. A divergence here is a bug in this crate until upstream is shown to be wrong.

use dsrs::signature::{FieldKind, InField, OutField, Signature};
use dsrs::{Adapter, ChatAdapter};
use serde_json::Value;

struct Fixture {
    name: String,
    source: String,
    signature: Signature,
    values: Vec<(&'static str, String)>,
    expected_system: String,
    expected_user: String,
}

/// Field names are `&'static str` in the signature types, and fixtures are read at run time.
/// Leaking is the honest trade in a test binary: a handful of names for the process lifetime.
fn static_str(value: &str) -> &'static str {
    Box::leak(value.to_owned().into_boxed_str())
}

fn kind_from(name: &str) -> FieldKind {
    match name {
        "str" => FieldKind::Str,
        "int" => FieldKind::Int,
        "float" => FieldKind::Float,
        "bool" => FieldKind::Bool,
        other => panic!("fixture uses an unmapped field kind: {other}"),
    }
}

fn load(path: &std::path::Path) -> Fixture {
    let raw = std::fs::read_to_string(path).expect("fixture is readable");
    let json: Value = serde_json::from_str(&raw).expect("fixture is valid json");

    let inputs = json["inputs"]
        .as_array()
        .expect("inputs array")
        .iter()
        .map(|field| InField {
            name: static_str(field["name"].as_str().expect("input name")),
            desc: field["desc"].as_str().unwrap_or_default().to_owned(),
            kind: kind_from(field["kind"].as_str().expect("input kind")),
        })
        .collect();

    let outputs = json["outputs"]
        .as_array()
        .expect("outputs array")
        .iter()
        .map(|field| OutField {
            name: static_str(field["name"].as_str().expect("output name")),
            desc: field["desc"].as_str().unwrap_or_default().to_owned(),
            kind: kind_from(field["kind"].as_str().expect("output kind")),
            values: None,
            schema: None,
        })
        .collect();

    let values = json["values"]
        .as_object()
        .expect("values object")
        .iter()
        .map(|(name, value)| {
            let rendered = match value {
                Value::String(text) => text.clone(),
                other => other.to_string(),
            };
            (static_str(name), rendered)
        })
        .collect();

    Fixture {
        name: path.file_stem().unwrap().to_string_lossy().into_owned(),
        source: json["source"].as_str().unwrap_or_default().to_owned(),
        signature: Signature {
            instructions: json["instructions"].as_str().expect("instructions").to_owned(),
            inputs,
            outputs,
        },
        values,
        expected_system: json["expected_system"].as_str().expect("expected_system").to_owned(),
        expected_user: json["expected_user"].as_str().expect("expected_user").to_owned(),
    }
}

fn fixtures() -> Vec<Fixture> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/conformance");
    let mut found: Vec<Fixture> = std::fs::read_dir(dir)
        .expect("conformance directory exists")
        .filter_map(|entry| {
            let path = entry.expect("readable entry").path();
            (path.extension()? == "json").then(|| load(&path))
        })
        .collect();
    found.sort_by(|a, b| a.name.cmp(&b.name));
    assert!(!found.is_empty(), "no conformance fixtures found");
    found
}

/// Show the first difference rather than two walls of text, so a failure names the divergence.
fn assert_same(label: &str, fixture: &Fixture, expected: &str, actual: &str) {
    if expected == actual {
        return;
    }
    let at = expected
        .char_indices()
        .zip(actual.char_indices())
        .find(|((_, want), (_, got))| want != got)
        .map(|((index, _), _)| index)
        .unwrap_or_else(|| expected.len().min(actual.len()));
    panic!(
        "{} diverges from Python DSPy in fixture `{}`\n  source: {}\n  first difference at byte {}\n\n  expected: {:?}\n  actual:   {:?}\n",
        label,
        fixture.name,
        fixture.source,
        at,
        &expected[at.saturating_sub(40)..(at + 60).min(expected.len())],
        &actual[at.saturating_sub(40)..(at + 60).min(actual.len())],
    );
}

#[test]
fn chat_adapter_renders_what_python_dspy_renders() {
    for fixture in fixtures() {
        let (system, user) = ChatAdapter.format(&fixture.signature, &fixture.values);
        assert_same("system message", &fixture, &fixture.expected_system, &system);
        assert_same("user message", &fixture, &fixture.expected_user, &user);
    }
}
