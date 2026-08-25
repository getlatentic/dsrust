//! What `dspy.Flex` hands its sandbox, string for string.
//!
//! A `Flex`'s whole state is a *string of Python source*, and the guest that runs it is Python in
//! both ports — so this is a byte comparison, not an approximation. Two strings decide whether the
//! optimizer-authored code sees what upstream's does: the signature rendered back to
//! `"in: T -> out: T2"`, and the baseline module built around it. The shim is upstream's own file
//! and is held to it character for character.
//!
//! The golden (`tests/conformance/predict/flex.json`, see `scripts/generate_flex_fixture.py`) was
//! recorded by running the pinned dspy, and it settled two things a transcription would have got
//! wrong: every signature built from a string is called `StringSignature`, so the generated class
//! is `StringSignatureModule` whatever the fields are; and the branch upstream keeps for a
//! signature with no instructions is unreachable, because dspy writes default instructions for one.

use dsrust::predict::flex::{Flex, SANDBOX_SHIM, class_name_of};
use dsrust::{FnTool, Signature};
use serde_json::{Value, json};

fn golden() -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/conformance/predict/flex.json");
    let text = std::fs::read_to_string(&path).expect("the flex golden is committed");
    serde_json::from_str(&text).expect("the golden parses")
}

fn cases() -> Vec<Value> {
    golden()["cases"].as_array().expect("cases").clone()
}

fn flex_for(case: &Value) -> Flex {
    let signature: Signature = case["signature"]
        .as_str()
        .unwrap_or("question -> answer")
        .parse()
        .expect("parses");
    let signature = match case["instructions"].as_str() {
        Some(instructions) => Signature {
            instructions: instructions.to_owned(),
            ..signature
        },
        None => signature,
    };
    Flex::new(signature)
}

/// The sandbox shim is upstream's file, not a rewrite of it.
///
/// It defines the stand-in `dspy` module the optimizer-authored code imports, so any drift is drift
/// in what that code is allowed to say — and the guest is Python here exactly as it is upstream, so
/// there is nothing to translate and no excuse for a difference.
#[test]
fn the_vendored_shim_is_upstreams_own() {
    let recorded = golden();
    let upstream = recorded["shim"].as_str().expect("the shim");
    assert_eq!(
        SANDBOX_SHIM, upstream,
        "the vendored _sandbox_shim.py has drifted from the pinned dspy"
    );
}

/// The signature rendered back the way the baseline embeds it.
#[test]
fn it_renders_the_signature_dspy_renders() {
    for case in cases() {
        let Some(_) = case["signature"].as_str() else {
            continue;
        };
        let expected = case["rendered_signature"].as_str().expect("rendered");
        assert_eq!(
            flex_for(&case).signature_string(),
            expected,
            "render_signature_string for {}",
            case["label"]
        );
    }
}

/// The baseline module, character for character — the source the sandbox is handed.
#[test]
fn it_builds_the_baseline_source_dspy_builds() {
    for case in cases() {
        let Some(_) = case["signature"].as_str() else {
            continue;
        };
        let expected = case["baseline_src"].as_str().expect("baseline");
        assert_eq!(
            flex_for(&case).module_src(),
            expected,
            "_baseline_src for {}",
            case["label"]
        );
    }
}

/// A declared signature's own name travels into the generated class.
///
/// Every string-built case above renders `StringSignatureModule`, so none of them can show this.
#[test]
fn a_declared_signature_names_the_generated_class() {
    let case = cases()
        .into_iter()
        .find(|case| case["label"] == "declared subclass")
        .expect("the declared-subclass case");
    let signature: Signature = "question -> answer".parse().expect("parses");
    let flex = Flex::new(Signature {
        instructions: "Answer the question.".to_owned(),
        ..signature
    })
    .named("QA");
    assert_eq!(
        flex.class_name(),
        case["class_name"].as_str().expect("class")
    );
    assert_eq!(
        flex.module_src(),
        case["baseline_src"].as_str().expect("src")
    );
}

/// Tools make the baseline an `RLM` that names each of them.
#[test]
fn tools_make_the_baseline_an_rlm() {
    let case = cases()
        .into_iter()
        .find(|case| case["label"] == "with a tool")
        .expect("the tool case");
    let signature: Signature = "question -> answer".parse().expect("parses");
    let shout = FnTool::new("shout", "Shout it.", json!({}), |_: &Value| {
        Ok("SHOUTED".to_owned())
    });
    let flex = Flex::new(signature)
        .with_tools(vec![std::sync::Arc::new(shout)])
        .expect("a valid tool name");
    assert_eq!(
        flex.module_src(),
        case["baseline_src"].as_str().expect("src")
    );
}

/// The class the sandbox instantiates, read out of whatever source an optimizer bound.
#[test]
fn it_finds_the_class_the_sandbox_runs() {
    let with_forward = "class Helper:\n    pass\n\nclass Real(dspy.Module):\n    def forward(self, **inputs):\n        return None\n";
    assert_eq!(
        class_name_of(with_forward).expect("finds it"),
        "Real",
        "the class defining forward wins over the one declared first"
    );
    let no_forward = "class Only(dspy.Module):\n    pass\n";
    assert_eq!(class_name_of(no_forward).expect("falls back"), "Only");
    assert!(
        class_name_of("x = 1\n").is_err(),
        "source defining no class is refused"
    );
}

/// A `Flex` runs its baseline in a real sandbox and the predictor it builds reaches the host.
///
/// This is the whole point of the module and the only test that proves the bridge: the generated
/// Python builds a `dspy.Predict`, which crosses back out as `__dspy_construct__`, and calling it
/// crosses back as `__dspy_call__` and reaches a model *here*. A scripted model answers, so what is
/// under test is the crossing rather than any provider.
///
/// Ignored by default for the reason `deno_sandbox.rs` gives: it needs `deno` on the path and the
/// first run downloads Pyodide.
///
///     cargo test --test flex_conformance -- --ignored --nocapture
#[tokio::test]
#[ignore = "needs deno and, on the first run, a Pyodide download"]
async fn a_flex_runs_its_baseline_through_the_sandbox() {
    use dsrust::lm::global;
    use dsrust::{DummyLM, Module, example};
    use std::sync::Arc;

    let lm = Arc::new(DummyLM::new([example! { answer: "Paris" }]));
    global::configure_model(reqwest::Client::new(), lm);

    let signature: Signature = "question -> answer".parse().expect("parses");
    let flex = Flex::new(signature);
    let answered = flex
        .forward(example! { question: "What is the capital of France?" })
        .await
        .expect("the sandbox ran and the predictor reached the host");

    assert_eq!(
        answered.get("answer").and_then(Value::as_str),
        Some("Paris"),
        "the generated forward's prediction did not come back: {answered:?}"
    );
}
