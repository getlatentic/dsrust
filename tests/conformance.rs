//! Conformance against Python DSPy's own adapter tests.
//!
//! Each fixture in `tests/conformance/` is one case lifted from upstream's
//! `tests/adapters/test_chat_adapter.py`, with the `expected` strings copied verbatim from
//! its assertions. Passing means this crate renders the prompt Python DSPy renders, so the
//! question "are we faithful?" is answered by the test run instead of by reading both
//! codebases. A divergence here is a bug in this crate until upstream is shown to be wrong.

use dsrs::adapter::Input;
use dsrs::signature::{FieldKind, InField, JsonType, OutField, Signature};
use dsrs::{Adapter, ChatAdapter, Example};
use serde_json::Value;

struct Fixture {
    name: String,
    source: String,
    signature: Signature,
    demos: Vec<Example>,
    values: Vec<(String, Value)>,
    expected_system: String,
    expected_turns: Vec<(String, String)>,
}

fn kind_from(name: &str) -> FieldKind {
    match name {
        "str" => FieldKind::Str,
        "int" => FieldKind::Int,
        "float" => FieldKind::Float,
        "bool" => FieldKind::Bool,
        // A non-scalar arrives as `json:<annotation>`, carrying the Python type dspy prints on
        // the numbered line instead of collapsing every non-scalar to one word.
        other => match other.strip_prefix("json:") {
            Some(annotation) => FieldKind::Json(JsonType::plain(annotation)),
            None => panic!("fixture uses an unmapped field kind: {other}"),
        },
    }
}

fn load(path: &std::path::Path) -> Fixture {
    let raw = std::fs::read_to_string(path).expect("fixture is readable");
    let json: Value = serde_json::from_str(&raw).expect("fixture is valid json");

    let inputs: Vec<InField> = json["inputs"]
        .as_array()
        .expect("inputs array")
        .iter()
        .map(|field| InField {
            name: field["name"].as_str().expect("input name").to_owned(),
            desc: field["desc"].as_str().unwrap_or_default().to_owned(),
            kind: kind_from(field["kind"].as_str().expect("input kind")),
            ..Default::default()
        })
        .collect();

    let outputs = json["outputs"]
        .as_array()
        .expect("outputs array")
        .iter()
        .map(|field| OutField {
            name: field["name"].as_str().expect("output name").to_owned(),
            desc: field["desc"].as_str().unwrap_or_default().to_owned(),
            kind: kind_from(field["kind"].as_str().expect("output kind")),
            ..Default::default()
        })
        .collect();

    // Signature order, not the fixture map's: dspy renders the input turn field by field as the
    // signature declares them, and a JSON object hands them back sorted.
    let values = inputs
        .iter()
        .filter_map(|field| {
            let value = json["values"].get(&field.name)?;
            Some((field.name.clone(), value.clone()))
        })
        .collect();

    let demos = json["demos"]
        .as_array()
        .map(|entries| {
            entries
                .iter()
                .map(|entry| {
                    Example::new(
                        entry
                            .as_object()
                            .expect("demo object")
                            .iter()
                            .map(|(name, value)| (name.clone(), value.clone())),
                    )
                })
                .collect()
        })
        .unwrap_or_default();

    let expected_turns = json["expected_turns"]
        .as_array()
        .expect("expected_turns array")
        .iter()
        .map(|turn| {
            (
                turn["role"].as_str().expect("turn role").to_owned(),
                turn["content"].as_str().expect("turn content").to_owned(),
            )
        })
        .collect();

    Fixture {
        name: path.file_stem().unwrap().to_string_lossy().into_owned(),
        source: json["source"].as_str().unwrap_or_default().to_owned(),
        signature: Signature {
            instructions: json["instructions"]
                .as_str()
                .expect("instructions")
                .to_owned(),
            inputs,
            outputs,
        },
        demos,
        values,
        expected_system: json["expected_system"]
            .as_str()
            .expect("expected_system")
            .to_owned(),
        expected_turns,
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
        // A fixture's values are JSON on disk, which is loose by construction — a golden has no
        // struct to have come from.
        let values: Vec<Input<'_>> = fixture
            .values
            .iter()
            .map(|(name, value)| Input::new(name.as_str(), value.clone()))
            .collect();
        let (system, turns) = ChatAdapter::default()
            .format(&fixture.signature, &fixture.demos, &values)
            .expect("the fixture renders");
        assert_same(
            "system message",
            &fixture,
            &fixture.expected_system,
            &system,
        );
        assert_eq!(
            turns.len(),
            fixture.expected_turns.len(),
            "fixture `{}` expects {} turns, got {}",
            fixture.name,
            fixture.expected_turns.len(),
            turns.len()
        );
        for (index, (expected, actual)) in fixture.expected_turns.iter().zip(&turns).enumerate() {
            assert_eq!(expected.0, actual.role.as_str(), "turn {index} role");
            assert_same(
                &format!("turn {index}"),
                &fixture,
                &expected.1,
                actual.content.text().expect("a text-only fixture"),
            );
        }
    }
}
