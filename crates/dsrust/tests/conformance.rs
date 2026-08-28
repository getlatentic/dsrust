//! Conformance against Python DSPy's own adapter tests.
//!
//! Each fixture in `tests/conformance/` is one case lifted from upstream's
//! `tests/adapters/test_chat_adapter.py`, with the `expected` strings copied verbatim from
//! its assertions. Passing means this crate renders the prompt Python DSPy renders, so the
//! question "are we faithful?" is answered by the test run instead of by reading both
//! codebases. A divergence here is a bug in this crate until upstream is shown to be wrong.

use dsrust::adapter::Input;
use dsrust::lm::api::LmRequest;
use dsrust::signature::{FieldKind, InField, JsonType, OutField, Signature};
use dsrust::{Adapter, ChainOfThought, ChatAdapter, Example, ReAct};
use serde_json::Value;

struct Fixture {
    name: String,
    source: String,
    signature: Signature,
    demos: Vec<Example>,
    values: Vec<(String, Value)>,
    expected_system: String,
    /// The content dspy rendered, kept as the JSON it is: a string for a text-only turn, and an
    /// array of blocks once a multimodal field has split it.
    expected_turns: Vec<(String, Value)>,
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
        .unwrap_or_else(|| {
            panic!(
                "{} has no `inputs` array. Every file directly in tests/conformance/ is an \
                 adapter fixture; put a golden about anything else in a subdirectory.",
                path.display()
            )
        })
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
                turn["content"].clone(),
            )
        })
        .collect();

    let declared = Signature {
        instructions: json["instructions"]
            .as_str()
            .expect("instructions")
            .to_owned(),
        inputs,
        outputs,
    };

    Fixture {
        name: path.file_stem().unwrap().to_string_lossy().into_owned(),
        source: json["source"].as_str().unwrap_or_default().to_owned(),
        // A fixture records which module rendered it, because one of them changes the signature
        // first: `ChainOfThought` prepends `reasoning`, and comparing what it prepends against
        // what dspy prepends is the whole reason that case exists.
        signature: match json["module"].as_str() {
            Some("chain_of_thought") => {
                ChainOfThought::from_signature(declared).signature().clone()
            }
            Some("react") => ReAct::new(declared, Vec::new()).turn_signature().clone(),
            // Declared by the crate rather than by the fixture: what is under test is whether
            // our own OfferFeedback renders what dspy's does, so reading it back out of the
            // fixture would compare the fixture to itself.
            Some("offer_feedback") => dsrust::predict::refine::feedback::signature(),
            _ => declared,
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
            // Only the top level, and every file in it is an adapter fixture — a golden about
            // anything else belongs in one of the subdirectories, which this walk does not
            // descend into. Landing one here used to fail as `inputs array`.
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

/// A turn's content, whichever of the two shapes dspy gave it.
///
/// The shapes are compared against each other rather than coerced: a message dspy rendered as
/// blocks must not pass because ours happened to render as prose that reads the same.
/// A rendered message's content against dspy's. Prose gets the character-level diff, since that
/// is the failure a reader has to localise; anything structured is compared whole.
fn assert_content(fixture: &Fixture, label: &str, expected: &Value, actual: &Value) {
    match (expected, actual) {
        (Value::String(expected), Value::String(actual)) => {
            assert_same(label, fixture, expected, actual)
        }
        (Value::Array(expected), Value::Array(actual)) => {
            assert_eq!(
                expected.len(),
                actual.len(),
                "{label} of fixture `{}` has {} blocks, dspy rendered {}",
                fixture.name,
                actual.len(),
                expected.len()
            );
            for (at, (expected, actual)) in expected.iter().zip(actual).enumerate() {
                assert_eq!(
                    expected, actual,
                    "{label} block {at} of fixture `{}`\n  source: {}",
                    fixture.name, fixture.source
                );
            }
        }
        (expected, actual) => panic!(
            "{label} of fixture `{}` disagrees on shape\n  source: {}\n\n  dspy rendered: {expected}\n  we rendered:   {actual}\n",
            fixture.name, fixture.source,
        ),
    }
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
        let rendered = ChatAdapter::default()
            .format(&fixture.signature, &fixture.demos, &values)
            .expect("the fixture renders");
        // Compared as the messages that go on the wire, which is the shape dspy's `format`
        // answers in and the shape the fixture generator read its `expected` values out of.
        let wire = LmRequest::new("", rendered).wire_messages();
        let (system, turns) = wire.split_first().expect("a render is never empty");
        // The generator checks this on the Python side before it splits dspy's list; checking it
        // here too is what makes the split a comparison rather than a shared assumption.
        assert_eq!(
            system["role"], "system",
            "fixture `{}` leads with `{}`, not the system message",
            fixture.name, system["role"]
        );
        assert_content(
            &fixture,
            "system message",
            &Value::String(fixture.expected_system.clone()),
            &system["content"],
        );
        assert_eq!(
            turns.len(),
            fixture.expected_turns.len(),
            "fixture `{}` expects {} turns, got {}",
            fixture.name,
            fixture.expected_turns.len(),
            turns.len()
        );
        for (index, (expected, actual)) in fixture.expected_turns.iter().zip(turns).enumerate() {
            assert_eq!(expected.0, actual["role"], "turn {index} role");
            assert_content(
                &fixture,
                &format!("turn {index}"),
                &expected.1,
                &actual["content"],
            );
        }
    }
}
