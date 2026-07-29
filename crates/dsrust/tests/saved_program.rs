//! A compiled program saved here, read by dspy.
//!
//! The claim this file exists to hold is that the artifact is portable: someone can take the JSON
//! a Rust optimizer wrote and run it in Python. That is a claim about bytes, so it is checked
//! against dspy's own reader rather than against a description of it — `scripts/check_saved_program.py`
//! opens the file with `dspy.Module.load` and reports what came back.
//!
//! The Rust-only assertions below pin the shape; the Python one pins that the shape is *dspy's*.

use std::collections::BTreeMap;

use dsrust::example::Example;
use dsrust::module::{Module, ProgramState};
use dsrust::predict::{ChainOfThought, Predict};
use serde_json::{Value, json};

fn saved(program: &mut dyn Module, path: &std::path::Path) -> Value {
    program.save(path).expect("saves");
    serde_json::from_str(&std::fs::read_to_string(path).expect("readable")).expect("valid json")
}

fn scratch(name: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("dsrust-saved-{name}.json"));
    let _ = std::fs::remove_file(&path);
    path
}

/// dspy keys a `ChainOfThought`'s state by its inner attribute, `predict`, and puts the metadata
/// beside the predictors rather than under them.
#[test]
fn the_file_has_dspys_shape_key_for_key() {
    let mut program =
        ChainOfThought::from_signature("question -> answer".parse().expect("a signature"));
    let file = saved(&mut program, &scratch("shape"));

    let state = &file["predict"];
    assert!(
        state.is_object(),
        "keyed by the predictor dspy names: {file}"
    );
    assert_eq!(state["traces"], json!([]));
    assert_eq!(state["train"], json!([]));
    assert_eq!(state["demos"], json!([]));
    assert_eq!(state["lm"], Value::Null);
    assert_eq!(
        state["signature"]["fields"],
        json!([
            { "prefix": "Question:", "description": "${question}" },
            { "prefix": "Reasoning:", "description": "${reasoning}" },
            { "prefix": "Answer:", "description": "${answer}" },
        ]),
        "inputs then outputs, in declaration order"
    );
    assert_eq!(
        file["metadata"]["dependency_versions"]["dspy"],
        json!(dsrust::module::DSPY_VERSION)
    );
}

/// What each optimizer leaves differs while the shape does not: demos for a bootstrap, rewritten
/// instructions for GEPA or COPRO. Both are legible in the file.
#[test]
fn an_optimizers_work_is_visible_in_the_file() {
    let mut bootstrapped = Predict::parse("question -> answer").expect("a signature");
    bootstrapped.demos = vec![Example::new([
        ("question", json!("2+2?")),
        ("answer", json!("4")),
    ])];
    let file = saved(&mut bootstrapped, &scratch("demos"));
    assert_eq!(
        file["self"]["demos"],
        json!([{ "question": "2+2?", "answer": "4" }])
    );

    let mut reflected = Predict::parse("question -> answer").expect("a signature");
    reflected.signature.instructions = "Answer with GOOD precision.".into();
    let file = saved(&mut reflected, &scratch("instructions"));
    assert_eq!(
        file["self"]["signature"]["instructions"],
        json!("Answer with GOOD precision.")
    );
}

/// A field description an optimizer rewrote has to survive the round trip. It did not before the
/// state carried the signature: only instructions travelled, so a rewritten description was lost
/// on load while the file still looked complete.
#[test]
fn a_rewritten_field_description_survives_the_round_trip() {
    let path = scratch("fields");
    let mut compiled = Predict::parse("question -> answer").expect("a signature");
    compiled.signature.instructions = "Be precise.".into();
    compiled.signature.inputs[0].desc = "the question, restated".into();
    compiled.signature.outputs[0].desc = "a single number".into();
    compiled.save(&path).expect("saves");

    let mut fresh = Predict::parse("question -> answer").expect("a signature");
    fresh.load(&path).expect("loads");
    assert_eq!(fresh.signature.instructions, "Be precise.");
    assert_eq!(fresh.signature.inputs[0].desc, "the question, restated");
    assert_eq!(fresh.signature.outputs[0].desc, "a single number");
}

/// dspy's own reader, on a file this crate wrote. Ignored by default because it needs the dspy
/// virtualenv the conformance suite uses.
///
///     cargo test -p dsrust --test saved_program -- --ignored
#[test]
#[ignore = "needs .dspy-venv, as the conformance suite does"]
fn dspy_loads_and_runs_what_this_crate_saved() {
    let path = scratch("for-python");
    let mut compiled =
        ChainOfThought::from_signature("question -> answer".parse().expect("a signature"));
    for predictor in compiled.named_predictors() {
        predictor.signature.instructions = "Answer with GOOD precision.".into();
        *predictor.demos = vec![Example::new([
            ("question", json!("2+2?")),
            ("reasoning", json!("add them")),
            ("answer", json!("4")),
        ])];
    }
    compiled.save(&path).expect("saves");

    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let checked = std::process::Command::new(root.join(".dspy-venv/bin/python"))
        .arg(root.join("scripts/check_saved_program.py"))
        .arg(&path)
        .output()
        .expect("the dspy venv runs");
    let report = String::from_utf8_lossy(&checked.stdout);
    assert!(
        checked.status.success(),
        "dspy could not load what we saved:\n{report}{}",
        String::from_utf8_lossy(&checked.stderr)
    );
    // The script prints what dspy read back, so a silent success cannot pass for a real one.
    assert!(report.contains("Answer with GOOD precision."), "{report}");
    assert!(report.contains("demos=1"), "{report}");
}

/// The state parses back into its own types, so a file written by an older build still loads.
#[test]
fn a_saved_file_reads_back_into_the_state_it_was_written_from() {
    let mut program =
        ChainOfThought::from_signature("question -> answer".parse().expect("a signature"));
    let path = scratch("roundtrip");
    program.save(&path).expect("saves");
    let read: ProgramState =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("readable")).expect("parses");
    let expected: BTreeMap<String, _> = program.dump_state().predictors;
    assert_eq!(read.predictors, expected);
}
