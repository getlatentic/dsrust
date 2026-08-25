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
    let signature: Signature = "question -> answer".parse().expect("parses");
    let flex = Flex::new(signature);
    let answered = global::context_model(reqwest::Client::new(), lm)
        .run(flex.forward(example! { question: "What is the capital of France?" }))
        .await
        .expect("the sandbox ran and the predictor reached the host");

    assert_eq!(
        answered.get("answer").and_then(Value::as_str),
        Some("Paris"),
        "the generated forward's prediction did not come back: {answered:?}"
    );
}

/// The predictor-call budget stops optimizer-authored code from looping the model.
///
/// The source a `Flex` runs is written by a model and runs unattended, so a loop around a predictor
/// is one bad proposal away. Upstream caps it at 100 per forward; this binds source that calls one
/// five times against a budget of two and asks for the refusal by its wording.
///
///     cargo test --test flex_conformance -- --ignored --nocapture
#[tokio::test]
#[ignore = "needs deno and, on the first run, a Pyodide download"]
async fn a_budget_stops_the_sandbox_looping_the_model() {
    use dsrust::lm::global;
    use dsrust::{DummyLM, Module, example};
    use std::sync::Arc;

    let answers: Vec<_> = (0..6).map(|_| example! { answer: "Paris" }).collect();
    let lm = Arc::new(DummyLM::new(answers));

    let signature: Signature = "question -> answer".parse().expect("parses");
    let mut flex = Flex::new(signature).max_predictor_calls(Some(2));
    flex.bind(
        "class Looping(dspy.Module):\n\
         \x20   def __init__(self):\n\
         \x20       super().__init__()\n\
         \x20       self.predict = dspy.Predict('question: str -> answer: str')\n\
         \n\
         \x20   def forward(self, **inputs):\n\
         \x20       for _ in range(5):\n\
         \x20           result = self.predict(**inputs)\n\
         \x20       return dspy.Prediction(answer=result.answer)",
    )
    .expect("the source names a class with a forward");

    let refused = global::context_model(reqwest::Client::new(), lm)
        .run(flex.forward(example! { question: "What is the capital of France?" }))
        .await
        .expect_err("five calls against a budget of two must be refused");
    let said = format!("{refused:#}");
    assert!(
        said.contains("predictor-call budget (2)"),
        "the refusal did not name the budget: {said}"
    );
}

/// The generated code names a tool and the host resolves it to the one the `Flex` was given.
///
/// A callable cannot cross the JSON boundary, so `tools=[shout]` reaches the host as
/// `[{"__dspy_tool__": "shout"}]` and the name is looked up. Both halves are worth holding: a name
/// that resolves builds a predictor holding the real tool, and a name that does not is the
/// generated code inventing one, which is refused rather than quietly dropped.
///
///     cargo test --test flex_conformance -- --ignored --nocapture
#[tokio::test]
#[ignore = "needs deno and, on the first run, a Pyodide download"]
async fn a_tool_the_generated_code_names_is_the_one_the_flex_holds() {
    use dsrust::lm::global;
    use dsrust::{DummyLM, Module, example};
    use std::sync::Arc;

    let answers: Vec<_> = (0..4).map(|_| example! { answer: "Paris" }).collect();
    let lm = Arc::new(DummyLM::new(answers));

    let signature: Signature = "question -> answer".parse().expect("parses");
    let shout = FnTool::new("shout", "Shout it.", json!({}), |_: &Value| {
        Ok("SHOUTED".to_owned())
    });
    let mut flex = Flex::new(signature)
        .with_tools(vec![Arc::new(shout)])
        .expect("a valid tool name");

    // A name the Flex was never given: the refusal names it rather than building a toolless ReAct.
    flex.bind(
        "class Inventing(dspy.Module):\n\
         \x20   def __init__(self):\n\
         \x20       super().__init__()\n\
         \x20       self.agent = dspy.ReAct('question: str -> answer: str', tools=[whisper])\n\
         \n\
         \x20   def forward(self, **inputs):\n\
         \x20       return dspy.Prediction(answer='unused')",
    )
    .expect("the source names a class with a forward");
    let refused = global::context_model(reqwest::Client::new(), lm)
        .run(flex.forward(example! { question: "anything" }))
        .await
        .expect_err("a tool this Flex was not given must be refused");
    let said = format!("{refused:#}");
    assert!(
        said.contains("whisper"),
        "the refusal did not name the invented tool: {said}"
    );
}

/// A program's state map holds whatever each submodule saved, and the shapes differ.
///
/// Running dspy over a program holding one `Predict` and one `Flex` writes `{traces, train, demos,
/// signature, lm}` under one key and `{module_src, lm}` under the other. A map typed to predictor
/// states cannot read that file back, which is why `ProgramState` holds a `SubmoduleState` — and
/// the golden is dspy's own output rather than a shape agreed with itself.
#[test]
fn a_state_map_reads_both_shapes_dspy_writes() {
    use dsrust::module::{ProgramState, SubmoduleState};

    let recorded = golden();
    let written = recorded["mixed_state"].clone();
    let state: ProgramState =
        serde_json::from_value(written.clone()).expect("dspy's own state map parses");

    assert!(
        matches!(state.state("plain"), Some(SubmoduleState::Predictor(_))),
        "the predictor entry did not read as one: {:?}",
        state.state("plain")
    );
    let Some(SubmoduleState::Flex(flexed)) = state.state("flexed") else {
        panic!(
            "the Flex entry did not read as one: {:?}",
            state.state("flexed")
        );
    };
    assert_eq!(
        flexed.module_src.as_deref(),
        written["flexed"]["module_src"].as_str(),
        "the saved source did not survive the read"
    );
    // `get` answers only for predictors, so a Flex entry is not mistaken for one.
    assert!(state.get("flexed").is_none(), "a Flex read as a predictor");
}

/// What a `Flex` saves, and that binding it back restores the source an optimizer left.
#[test]
fn a_flex_saves_the_source_and_loads_it_again() {
    use dsrust::module::Module;

    let signature: Signature = "question -> answer".parse().expect("parses");
    let mut flex = Flex::new(signature);
    let baseline = flex.module_src().to_owned();

    let optimized = "class Rewritten(dspy.Module):\n\
                     \x20   def forward(self, **inputs):\n\
                     \x20       return dspy.Prediction(answer='x')";
    flex.bind(optimized).expect("valid source");
    let saved = flex.dump_state();

    let signature: Signature = "question -> answer".parse().expect("parses");
    let mut restored = Flex::new(signature);
    assert_eq!(
        restored.module_src(),
        baseline,
        "a fresh Flex holds its baseline"
    );
    restored.load_state(&saved).expect("the saved state loads");
    assert_eq!(
        restored.module_src(),
        optimized,
        "loading did not restore the source the optimizer left"
    );
}
/// A predictor entry that lost its signature is refused, not read as a Flex.
///
/// Both of `FlexState`'s fields are optional, so an untagged read matches *any* object unless the
/// unknown ones are denied — and this exact input read as `FlexState { module_src: None }` before
/// they were. A corrupt saved program loading quietly as a sourceless `Flex` is worse than one that
/// fails to load.
#[test]
fn a_malformed_predictor_is_not_read_as_a_flex() {
    use dsrust::module::ProgramState;

    let broken = json!({ "plain": { "traces": [], "demos": [], "lm": null } });
    assert!(
        serde_json::from_value::<ProgramState>(broken).is_err(),
        "a predictor entry with no signature was accepted"
    );
}

/// How a batch of failures reaches the code proposer's prompt, character for character.
///
/// The values go through Python's `repr`, not JSON: `{'question': 'Where?'}` with single quotes and
/// CPython's quote-switching for an apostrophe. Rendering them as JSON would put a different string
/// in front of the model, which is a defect this crate has had once before.
#[test]
fn it_renders_the_failures_dspy_renders() {
    use dsrust::predict::flex::proposal::format_failures;
    use serde_json::Map;

    for case in golden()["format_failures"].as_array().expect("cases") {
        let records: Vec<Map<String, Value>> = case["records"]
            .as_array()
            .expect("records")
            .iter()
            .map(|record| record.as_object().expect("an object").clone())
            .collect();
        assert_eq!(
            format_failures(&records),
            case["rendered"].as_str().expect("rendered"),
            "_format_failures for {}",
            case["label"]
        );
    }
}

/// The source read back out of a fenced reply.
///
/// Three edges the golden settles: the opening line goes whatever it says, a reply that is nothing
/// but a fence ends up empty rather than keeping its three characters, and tabs advance to the next
/// multiple of four *within their line* — not four spaces each, which is what a `replace` would do.
#[test]
fn it_strips_the_fences_dspy_strips() {
    use dsrust::predict::flex::proposal::strip_code_fences;

    for case in golden()["strip_code_fences"].as_array().expect("cases") {
        let raw = case["raw"].as_str().expect("raw");
        assert_eq!(
            strip_code_fences(raw),
            case["stripped"].as_str().expect("stripped"),
            "_strip_code_fences({raw:?})"
        );
    }
}
