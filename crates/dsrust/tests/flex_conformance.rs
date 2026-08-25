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

/// An optimizer reaches a `Flex` nested in a program, and writes a proposal back through it.
///
/// This is the walk `named_predictors` cannot do: a `Flex` has no predictors to hand out, so an
/// optimizer that only knew that walk would see a program with nothing to optimize. The composed
/// module below recurses exactly as it must for predictors, and the test would pass on a bare `Flex`
/// without proving anything — which is why the subject is a program with a `Flex` *inside* it.
#[test]
fn an_optimizer_reaches_a_flex_nested_in_a_program() {
    use dsrust::module::{Module, NamedFlex};
    use dsrust::predict::flex::proposal::{flex_components, rebind_flex_code};
    use dsrust::{Example, Prediction};
    use std::collections::BTreeMap;

    struct Program {
        flexed: Flex,
    }

    impl Module for Program {
        fn forward<'a>(
            &'a self,
            _inputs: Example,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = anyhow::Result<Prediction>> + Send + 'a>,
        > {
            Box::pin(async { unreachable!("this test never runs the program") })
        }

        fn named_flexes(&mut self) -> Vec<NamedFlex<'_>> {
            self.flexed
                .named_flexes()
                .into_iter()
                .map(|mut inner| {
                    inner.name = "flexed".to_owned();
                    inner
                })
                .collect()
        }
    }

    let signature: Signature = "question -> answer".parse().expect("parses");
    let mut program = Program {
        flexed: Flex::new(signature),
    };

    let components = flex_components(&mut program);
    assert_eq!(
        components.keys().collect::<Vec<_>>(),
        ["flexed"],
        "the walk did not reach the Flex under its parent's name"
    );
    assert!(
        components["flexed"].contains("class StringSignatureModule"),
        "the component is not the module's source: {}",
        components["flexed"]
    );

    let proposed = "class Rewritten(dspy.Module):\n\
                    \x20   def forward(self, **inputs):\n\
                    \x20       return dspy.Prediction(answer='better')";
    let candidate = BTreeMap::from([
        ("flexed".to_owned(), proposed.to_owned()),
        // A component the program has no Flex for — an optimizer's candidate carries every
        // component and only some of them are code.
        ("some_predictor".to_owned(), "not source".to_owned()),
    ]);
    rebind_flex_code(&mut program, &candidate).expect("the proposal binds");
    assert_eq!(program.flexed.module_src(), proposed);

    // Source naming no class is refused here rather than at the next forward.
    let broken = BTreeMap::from([("flexed".to_owned(), "x = 1".to_owned())]);
    assert!(rebind_flex_code(&mut program, &broken).is_err());
}

/// What the code proposer is shown about the module it is rewriting.
///
/// Two renderings with edges the golden settles: a description is skipped when it is dspy's own
/// `${field}` placeholder — which a signature parsed here spells as an empty string instead, so both
/// forms of "nobody said" have to be skipped — and a tool's blurb is the *first line* of its
/// description, stripped, so a tool with nothing to say keeps its trailing space.
#[test]
fn it_shows_the_proposer_what_dspy_shows() {
    use std::sync::Arc;

    fn flex_for(label: &str) -> Flex {
        let signature: Signature = "question -> answer".parse().expect("parses");
        match label {
            "described" => {
                let mut described: Signature = "question: str, context: str -> answer: str"
                    .parse()
                    .expect("parses");
                described.instructions = "Answer carefully.".to_owned();
                described.inputs[0].desc = "The question asked.".to_owned();
                described.outputs[0].desc = "A short answer.".to_owned();
                Flex::new(described).named("Described")
            }
            "one tool" => Flex::new(signature)
                .with_tools(vec![Arc::new(FnTool::new(
                    "shout",
                    "Shout it.\nSecond line ignored.",
                    json!({}),
                    |_: &Value| Ok(String::new()),
                ))])
                .expect("a valid tool name"),
            "tool with no docstring" => Flex::new(signature)
                .with_tools(vec![Arc::new(FnTool::new(
                    "quiet",
                    "",
                    json!({}),
                    |_: &Value| Ok(String::new()),
                ))])
                .expect("a valid tool name"),
            "two tools" => Flex::new(signature)
                .with_tools(vec![
                    Arc::new(FnTool::new("shout", "Shout it.", json!({}), |_: &Value| {
                        Ok(String::new())
                    })),
                    Arc::new(FnTool::new(
                        "whisper",
                        "  Whisper it.  ",
                        json!({}),
                        |_: &Value| Ok(String::new()),
                    )),
                ])
                .expect("a valid tool name"),
            _ => Flex::new(signature),
        }
    }

    for case in golden()["task_context"].as_array().expect("cases") {
        let label = case["label"].as_str().expect("label");
        let flex = flex_for(label);
        assert_eq!(
            flex.signature_spec(),
            case["signature_spec"].as_str().expect("spec"),
            "render_signature_spec for {label}"
        );
        assert_eq!(
            flex.context_blurb(true),
            case["context_blurb"].as_str().expect("blurb"),
            "render_context_blurb(sandboxed=True) for {label}"
        );
        assert_eq!(
            flex.context_blurb(false),
            case["context_blurb_unsandboxed"].as_str().expect("blurb"),
            "render_context_blurb() for {label}"
        );
    }
}

/// A metric scoring a program that holds a `Flex` reads the run from `program_trace`, not `trace`.
///
/// Upstream is explicit that `trace` stays `None` here, "preserving the eval-mode semantics of
/// non-Flex GEPA scoring" — so a metric written against `trace` behaves the same whether or not a
/// Flex appears, and one that wants the run asks for `program_trace`. Two claims, and the second is
/// the one a port drops: it is easy to fill both and pass any test that only checks the run arrived.
#[test]
fn a_flex_program_scores_through_program_trace_and_leaves_trace_empty() {
    use dsrust::module::{Module, NamedFlex};
    use dsrust::optimize::{Feedback, MetricContext};
    use dsrust::{Example, Prediction};
    use std::sync::{Arc, Mutex};

    #[derive(Default)]
    struct Seen {
        trace: Vec<bool>,
        program_trace: Vec<bool>,
    }

    struct Holder {
        flexed: Option<Flex>,
        inner: dsrust::predict::Predict,
    }

    impl Module for Holder {
        fn forward<'a>(
            &'a self,
            inputs: Example,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = anyhow::Result<Prediction>> + Send + 'a>,
        > {
            self.inner.forward(inputs)
        }

        fn named_predictors(&mut self) -> Vec<dsrust::NamedPredictor<'_>> {
            self.inner.named_predictors()
        }

        fn named_flexes(&mut self) -> Vec<NamedFlex<'_>> {
            self.flexed
                .iter_mut()
                .flat_map(Module::named_flexes)
                .collect()
        }
    }

    fn scored_with(holding_a_flex: bool) -> Seen {
        let seen = Arc::new(Mutex::new(Seen::default()));
        let recorded = seen.clone();
        let signature: Signature = "question -> answer".parse().expect("parses");
        let mut program = Holder {
            flexed: holding_a_flex.then(|| Flex::new(signature.clone())),
            inner: dsrust::predict::Predict::from_signature(signature.clone()),
        };
        let metric = move |_: &Example, _: &Prediction, context: &MetricContext<'_>| {
            let mut at = recorded.lock().expect("not poisoned");
            at.trace.push(context.trace.is_some());
            at.program_trace.push(context.program_trace.is_some());
            Feedback::score_only(1.0)
        };
        let lm = Arc::new(dsrust::DummyLM::new([
            dsrust::example! { answer: "Paris" },
            dsrust::example! { answer: "Paris" },
        ]));
        dsrust::lm::global::configure_model(reqwest::Client::new(), lm);
        let trainset = vec![dsrust::example! { question: "Where?" }.with_inputs(["question"])];
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("a runtime")
            .block_on(async {
                let _ = dsrust::GEPA::new(metric, Arc::new(dsrust::DummyLM::new([])) as Arc<_>)
                    .max_metric_calls(2)
                    .compile(&mut program, &trainset, &trainset)
                    .await;
            });
        Arc::try_unwrap(seen)
            .map(|held| held.into_inner().expect("not poisoned"))
            .ok()
            .expect("one holder left")
    }

    let with_flex = scored_with(true);
    assert!(
        with_flex.program_trace.iter().any(|filled| *filled),
        "a Flex program scored with no program_trace"
    );
    assert!(
        with_flex.trace.iter().all(|filled| !filled),
        "trace was filled for a Flex program, which upstream keeps empty so eval-mode semantics hold"
    );

    let without = scored_with(false);
    assert!(
        without.program_trace.iter().all(|filled| !filled),
        "program_trace was filled for a program with no Flex"
    );
}

/// GEPA optimizes a `Flex` by rewriting its *source*, which is the thing a Flex exists for.
///
/// Every other flex test here exercises one piece. This is the loop: GEPA finds the Flex through
/// `named_flexes`, builds its reflective records from whole-program I/O, sends them to the code
/// proposer with the primitives catalog, and binds the module the model answers with. The reflection
/// model is scripted, so what is under test is the wiring rather than any model's judgement.
///
/// Before this, `propose_new_texts` looked every component up in `named_predictors` and a Flex
/// silently fell out of the map — the search ran and optimized nothing.
#[tokio::test]
async fn gepa_rewrites_a_flexs_source() {
    use dsrust::lm::{ChatModel, DynChatModel, api};
    use dsrust::module::{Module, NamedFlex};
    use dsrust::optimize::{Feedback, MetricContext};
    use dsrust::{Example, Prediction};
    use std::sync::{Arc, Mutex};

    const REWRITTEN: &str = "class Better(dspy.Module):\n\
                             \x20   def forward(self, **inputs):\n\
                             \x20       return dspy.Prediction(answer='better')";

    /// A reflection model that answers every proposal with the same fenced source, and records the
    /// prompts it was sent so the test can assert the proposer was reached at all.
    struct Coder(Arc<Mutex<Vec<String>>>);

    impl ChatModel for Coder {
        async fn forward(&self, request: &api::LmRequest) -> anyhow::Result<api::LmResponse> {
            self.0
                .lock()
                .expect("not poisoned")
                .push(request.system().to_owned());
            Ok(api::LmResponse::text(format!(
                "[[ ## revised_source ## ]]\n```python\n{REWRITTEN}\n```\n\n[[ ## completed ## ]]"
            )))
        }
    }

    struct Program {
        flexed: Flex,
    }

    impl Module for Program {
        fn forward<'a>(
            &'a self,
            _inputs: Example,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = anyhow::Result<Prediction>> + Send + 'a>,
        > {
            // The program answers from its *own current source*, so a proposal that is never bound
            // scores exactly as the seed does and GEPA rightly rejects it. Without this the test
            // passes on a search that accepted nothing.
            let answer = match self.flexed.module_src().contains("Better") {
                true => "better",
                false => "worse",
            };
            Box::pin(async move {
                Ok(Prediction::new(
                    dsrust::example! { answer: answer },
                    String::new(),
                ))
            })
        }

        fn named_flexes(&mut self) -> Vec<NamedFlex<'_>> {
            self.flexed
                .named_flexes()
                .into_iter()
                .map(|mut inner| {
                    inner.name = "flexed".to_owned();
                    inner
                })
                .collect()
        }
    }

    let prompts = Arc::new(Mutex::new(Vec::new()));
    let signature: Signature = "question -> answer".parse().expect("parses");
    let mut program = Program {
        flexed: Flex::new(signature),
    };
    let trainset = vec![dsrust::example! { question: "Where?" }.with_inputs(["question"])];

    dsrust::GEPA::new(
        |_: &Example, prediction: &Prediction, _: &MetricContext<'_>| match prediction
            .get("answer")
            .and_then(Value::as_str)
        {
            Some("better") => Feedback::new(1.0, "right"),
            _ => Feedback::new(0.0, "wrong answer"),
        },
        Arc::new(Coder(prompts.clone())) as Arc<dyn DynChatModel>,
    )
    .max_metric_calls(6)
    .reflection_minibatch_size(1)
    .compile(&mut program, &trainset, &trainset)
    .await
    .expect("compiles");

    let sent = prompts.lock().expect("not poisoned");
    assert!(
        sent.iter()
            .any(|prompt| prompt.contains("Revise the full source code")),
        "the code proposer was never reached: {sent:?}"
    );
    assert_eq!(
        program.flexed.module_src(),
        REWRITTEN,
        "GEPA accepted a source and did not bind it"
    );
}

/// A tool name is refused exactly where Python refuses one, over the cases that were measured.
///
/// Upstream rejects a tool whose name is not a Python identifier, because the generated code
/// references it by name. Python's rule is `XID_Start`/`XID_Continue` — derived properties with no
/// table in this crate — so the implementation is an approximation and this is what says how good
/// it is. `x` plus a combining acute is the case that found the first gap: an identifier to Python,
/// refused here, so a tool dspy accepts could not be registered.
#[test]
fn it_refuses_the_tool_names_python_refuses() {
    use serde_json::json;
    use std::sync::Arc;

    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/conformance/predict/identifiers.json");
    let recorded: Value =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("committed")).expect("parses");

    for (name, python) in recorded["python"].as_object().expect("cases") {
        let signature: Signature = "question -> answer".parse().expect("parses");
        let tool = FnTool::new(name.clone(), "a tool", json!({}), |_: &Value| {
            Ok(String::new())
        });
        let accepted = Flex::new(signature)
            .with_tools(vec![Arc::new(tool)])
            .is_ok();
        // A name in the sandbox's own `_dspy` namespace is refused for a second reason, which these
        // cases do not reach — none of them starts with an underscore followed by `dspy`.
        assert_eq!(
            accepted,
            python.as_bool().expect("a verdict"),
            "tool name {name:?}: dspy would {}",
            match python.as_bool() == Some(true) {
                true => "accept it",
                false => "refuse it",
            }
        );
    }
}

/// The primitives catalog is upstream's file, not a rewrite of it.
///
/// It is what the code proposer is told it may write, and the code it describes is Python running in
/// the sandbox — so it is vendored for the shim's reason: there is nothing to translate, and drift
/// from the pin would mean the model is told a different set of rules than dspy tells it.
///
/// This test is here because a doc comment claimed it before it existed.
#[test]
fn the_vendored_catalog_is_upstreams_own() {
    use dsrust::predict::flex::proposal::PRIMITIVES_CATALOG;

    let recorded = golden();
    let upstream = recorded["primitives_catalog"]
        .as_str()
        .expect("the catalog");
    assert_eq!(
        PRIMITIVES_CATALOG, upstream,
        "the vendored primitives_doc.py has drifted from the pinned dspy"
    );
}
