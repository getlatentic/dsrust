//! dspy's `BaseCallback`, held to the handler sequences dspy itself fired.
//!
//! `tests/observe.rs` says the six points exist as spans. This says they exist as *callbacks*, and
//! that they fire where upstream's do — which is the half a trait with defaulted methods cannot get
//! wrong by accident and can very easily get wrong by counting. The sequences come from
//! `tests/conformance/observe/callbacks.json`, recorded by running pinned dspy under a recording
//! `BaseCallback`; `chain_of_thought_n3` is the case upstream asserts by hand in
//! `tests/callback/test_callback.py`.
//!
//! Depth is computed here the way the generator computes it: a start handler's `parent` is the call
//! it happened inside, so a child is one deeper than its parent. That is upstream's `ACTIVE_CALL_ID`
//! read at start-handler time, which is what `test_active_id` asserts.

use std::sync::{Arc, Mutex};

use anyhow::Error;
use dsrust::callback::Rendered;
use dsrust::evaluate::{Evaluation, Pass};
use dsrust::lm::{api, global};
use dsrust::signature::Signature;
use dsrust::{
    CallId, Callback, ChainOfThought, DummyLM, Evaluate, Example, Forward, Module, Prediction,
    configure_callbacks, example,
};
use serde_json::Value;

/// Both the configured model and the callback list are process-wide, as dspy's are, so these tests
/// take turns.
static SERIAL: Mutex<()> = Mutex::new(());

fn install(
    lm: Arc<dyn dsrust::lm::DynChatModel>,
    recording: Arc<Recording>,
) -> std::sync::MutexGuard<'static, ()> {
    let guard = SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    global::configure_model(reqwest::Client::new(), lm);
    configure_callbacks([recording as Arc<dyn Callback>]);
    guard
}

/// A model whose call fails, which is how the abandoned evaluation loses its one row.
///
/// Not an exhausted [`DummyLM`]: upstream's answers `No more responses` rather than refusing, so a
/// script that runs dry is a parse failure and a retry, not the single clean model error this case
/// is about.
struct FailingLM;

impl dsrust::lm::ChatModel for FailingLM {
    fn forward<'a>(
        &'a self,
        _request: &'a api::LmRequest,
    ) -> impl std::future::Future<Output = anyhow::Result<api::LmResponse>> + Send + 'a {
        std::future::ready(Err(anyhow::anyhow!("no answer")))
    }
}

/// Every handler, recording its name and how deep the call it belongs to was.
#[derive(Default)]
struct Recording {
    calls: Mutex<Vec<(String, usize)>>,
    depth: Mutex<std::collections::HashMap<u64, usize>>,
    /// The roles of what `on_adapter_format_end` was handed. The handler recorded only that it
    /// fired, so the payload it exists to deliver was never read — and a change to `Rendered`'s
    /// shape passed every assertion here.
    rendered: Mutex<Vec<Vec<String>>>,
}

impl Recording {
    fn started(&self, handler: &str, call: &CallId) {
        let mut depths = self.depth.lock().expect("not poisoned");
        let depth = call
            .parent()
            .and_then(|parent| depths.get(&parent).copied())
            .map_or(0, |parent| parent + 1);
        depths.insert(call.id(), depth);
        drop(depths);
        self.calls
            .lock()
            .expect("not poisoned")
            .push((handler.to_owned(), depth));
    }

    fn ended(&self, handler: &str, call: &CallId) {
        let depth = self.depth.lock().expect("not poisoned")[&call.id()];
        self.calls
            .lock()
            .expect("not poisoned")
            .push((handler.to_owned(), depth));
    }

    /// The recording, rendered the way the fixture writes it: one indented line per handler, so a
    /// failure prints two readable trees rather than two lists of tuples.
    fn tree(&self) -> String {
        self.calls
            .lock()
            .expect("not poisoned")
            .iter()
            .map(|(handler, depth)| format!("{}{handler}", "  ".repeat(*depth)))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

impl Callback for Recording {
    fn on_module_start(&self, call: &CallId, _module: &str, _inputs: &Example) {
        self.started("on_module_start", call);
    }

    fn on_module_end(&self, call: &CallId, _answered: Result<&Prediction, &Error>) {
        self.ended("on_module_end", call);
    }

    fn on_lm_start(&self, call: &CallId, _request: &api::LmRequest) {
        self.started("on_lm_start", call);
    }

    fn on_lm_end(&self, call: &CallId, _answered: Result<&api::LmResponse, &Error>) {
        self.ended("on_lm_end", call);
    }

    fn on_adapter_format_start(
        &self,
        call: &CallId,
        _adapter: &str,
        _signature: &Signature,
        _demos: &[Example],
        _inputs: &[dsrust::adapter::Input<'_>],
    ) {
        self.started("on_adapter_format_start", call);
    }

    fn on_adapter_format_end(&self, call: &CallId, answered: Result<&Rendered<'_>, &Error>) {
        if let Ok(rendered) = answered {
            self.rendered.lock().expect("not poisoned").push(
                rendered
                    .messages
                    .iter()
                    .map(|message| message.role.clone())
                    .collect(),
            );
        }
        self.ended("on_adapter_format_end", call);
    }

    fn on_adapter_parse_start(&self, call: &CallId, _adapter: &str, _raw: &str) {
        self.started("on_adapter_parse_start", call);
    }

    fn on_adapter_parse_end(&self, call: &CallId, _answered: Result<&Value, &Error>) {
        self.ended("on_adapter_parse_end", call);
    }

    fn on_tool_start(&self, call: &CallId, _tool: &str, _args: &Value) {
        self.started("on_tool_start", call);
    }

    fn on_tool_end(&self, call: &CallId, _answered: Result<&Value, &Error>) {
        self.ended("on_tool_end", call);
    }

    fn on_evaluate_start(
        &self,
        call: &CallId,
        _devset: &[Example],
        _threads: usize,
        _pass: Option<Pass>,
    ) {
        self.started("on_evaluate_start", call);
    }

    fn on_evaluate_end(&self, call: &CallId, _evaluated: Result<&Evaluation, &Error>) {
        self.ended("on_evaluate_end", call);
    }
    fn on_interpreter_execute_start(&self, call: &CallId, _interpreter: &str, _code: &str) {
        self.started("on_interpreter_execute_start", call);
    }
    fn on_interpreter_execute_end(
        &self,
        call: &CallId,
        _answered: Result<&dsrust::interpreter::Executed, &Error>,
    ) {
        self.ended("on_interpreter_execute_end", call);
    }
    fn on_interpreter_tool_call_start(&self, call: &CallId, _tool: &str, _args: &Value) {
        self.started("on_interpreter_tool_call_start", call);
    }
    fn on_interpreter_tool_call_end(&self, call: &CallId, _answered: Result<&Value, &Error>) {
        self.ended("on_interpreter_tool_call_end", call);
    }
    fn on_interpreter_startup_start(&self, call: &CallId, _interpreter: &str) {
        self.started("on_interpreter_startup_start", call);
    }
    fn on_interpreter_startup_end(&self, call: &CallId, _answered: Result<(), &Error>) {
        self.ended("on_interpreter_startup_end", call);
    }
    fn on_interpreter_shutdown_start(&self, call: &CallId, _interpreter: &str) {
        self.started("on_interpreter_shutdown_start", call);
    }
    fn on_interpreter_shutdown_end(&self, call: &CallId, _answered: Result<(), &Error>) {
        self.ended("on_interpreter_shutdown_end", call);
    }
    fn on_compile_start(
        &self,
        call: &CallId,
        _optimizer: &str,
        _trainset: &[Example],
        _valset: Option<&[Example]>,
    ) {
        self.started("on_compile_start", call);
    }
    fn on_compile_end(&self, call: &CallId, _compiled: Result<(), &Error>) {
        self.ended("on_compile_end", call);
    }
}

/// dspy 3.3.1's compile point: an optimizer's `compile` opens a call that every module run inside
/// it is a child of, and closes it whatever happened — upstream's `test_optimizer_callback`.
#[allow(clippy::await_holding_lock)] // the installer's own note: `SERIAL` is a test token, taken by nothing under test
#[tokio::test]
async fn a_compile_encloses_the_runs_it_makes() {
    let recording = Arc::new(Recording::default());
    let lm: Arc<dyn dsrust::lm::DynChatModel> = Arc::new(DummyLM::new(std::iter::repeat_n(
        example! { answer: "fine" },
        8,
    )));
    let _serial = install(lm, recording.clone());
    let mut student = dsrust::Predict::from_signature(signature());
    let trainset =
        vec![example! { question: "How are you?", answer: "fine" }.with_inputs(["question"])];
    let metric = dsrust::evaluate::exact_match;
    dsrust::optimize::BootstrapFewShot::new(&metric)
        .compile(&mut student, &trainset)
        .await
        .expect("compiles");
    let tree = recording.tree();
    let lines: Vec<&str> = tree.lines().collect();
    assert_eq!(lines.first().copied(), Some("on_compile_start"), "{tree}");
    assert_eq!(lines.last().copied(), Some("on_compile_end"), "{tree}");
    assert!(
        lines.contains(&"  on_module_start"),
        "the teacher's runs are children of the compile:\n{tree}"
    );
    assert_eq!(
        lines
            .iter()
            .filter(|line| **line == "on_compile_start")
            .count(),
        1,
        "the trait's compile delegating to the inherent one is one compile, not two:\n{tree}"
    );
    // dspy's `BootstrapFewShot` compiles a `LabeledFewShot` for its teacher, and that compile is
    // its own point, one level down.
    assert!(
        lines.contains(&"  on_compile_start"),
        "the teacher's LabeledFewShot compile is a child compile:\n{tree}"
    );
}

/// One case out of the fixture, as the same indented tree [`Recording::tree`] renders.
fn expected(name: &str) -> String {
    let fixture: Value = serde_json::from_str(include_str!("conformance/observe/callbacks.json"))
        .expect("the fixture parses");
    let case = fixture["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .find(|case| case["name"] == name)
        .unwrap_or_else(|| panic!("no case named {name} in callbacks.json"));
    case["handlers"]
        .as_array()
        .expect("handlers")
        .iter()
        .map(|handler| {
            format!(
                "{}{}",
                "  ".repeat(handler["depth"].as_u64().expect("depth") as usize),
                handler["handler"].as_str().expect("handler"),
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn signature() -> Signature {
    "question -> answer".parse().expect("parses")
}

fn asked() -> Example {
    example! { question: "How are you?" }.with_inputs(["question"])
}

/// One predictor: the module, its render, the model call, one parse.
#[allow(clippy::await_holding_lock)] // the installer's own note: `SERIAL` is a test token, taken by nothing under test
#[tokio::test]
async fn a_predict_fires_the_sequence_dspy_fires() {
    let recording = Arc::new(Recording::default());
    let lm = Arc::new(DummyLM::new([example! { answer: "test output" }]));
    let _serial = install(lm, recording.clone());

    dsrust::predict::Predict::from_signature(signature())
        .forward(asked())
        .await
        .expect("the scripted answer parses");
    configure_callbacks([]);

    assert_eq!(recording.tree(), expected("predict"));
    // What the handler was handed, not merely that it fired: `Rendered` carries the message list
    // `format` produced, the system prompt leading it as dspy's does.
    assert_eq!(
        recording.rendered.lock().expect("not poisoned").as_slice(),
        [vec!["system".to_owned(), "user".to_owned()]],
    );
}

/// dspy's third registration path: handlers attached to *one* predictor.
///
/// `dspy.Module.__init__` stores a `callbacks` list every subclass inherits, so
/// `dspy.Predict("q -> a", callbacks=[cb])` watches that predictor and no other. A Rust `Module` is
/// a trait with no shared constructor, so the list is a defaulted trait method a module overrides
/// — and what this asserts is the part that could not be had otherwise: two predictors of the *same
/// signature*, one watched, and only its points recorded. Filtering by the module's kind or by
/// `CallId::parent` cannot tell those two apart.
#[allow(clippy::await_holding_lock)] // the installer's own note: `SERIAL` is a test token, taken by nothing under test
#[tokio::test]
async fn only_the_predictor_a_caller_watched_is_recorded() {
    let watched = Arc::new(Recording::default());
    let lm = Arc::new(DummyLM::new([
        example! { answer: "first" },
        example! { answer: "second" },
    ]));
    // Nothing process-wide: the only handler in play is the one attached to a single predictor.
    let _serial = install(lm, Arc::new(Recording::default()));
    configure_callbacks([]);

    let seen = dsrust::predict::Predict::from_signature(signature())
        .callbacks([watched.clone() as Arc<dyn dsrust::Callback>]);
    let unseen = dsrust::predict::Predict::from_signature(signature());

    seen.forward(asked())
        .await
        .expect("the scripted answer parses");
    unseen
        .forward(asked())
        .await
        .expect("the scripted answer parses");

    let recorded = watched.tree();
    let modules = recorded
        .lines()
        .filter(|line| line.trim() == "on_module_start")
        .count();
    assert_eq!(
        modules, 1,
        "two predictors ran and one carried the handler:\n{recorded}"
    );
}

/// A composed module and the predictor inside it, which is what makes the nesting observable.
///
/// Upstream asserts this sequence by hand for `n=3`, where parsing runs once per output. The crate's
/// `Predict::forward` reads one output whatever `n` is — see the `predict-completions` story — so
/// this case is run at the default `n`, where the two agree, and the fixture's `n=3` case is checked
/// against it below.
#[allow(clippy::await_holding_lock)] // the installer's own note: `SERIAL` is a test token, taken by nothing under test
#[tokio::test]
async fn a_chain_of_thought_nests_its_predictor() {
    let recording = Arc::new(Recording::default());
    let lm = Arc::new(DummyLM::new([
        example! { reasoning: "No more responses", answer: "test output" },
    ]));
    let _serial = install(lm, recording.clone());

    ChainOfThought::from_signature(signature())
        .forward(asked())
        .await
        .expect("the scripted answer parses");
    configure_callbacks([]);

    // dspy's own `n=3` recording, with the two extra parses removed: those are the candidates
    // `Predict::forward` does not read. Everything else — the two module points, the render, the
    // model call and the nesting — has to be identical, and this is what says so.
    let upstream = expected("chain_of_thought_n3");
    let mut parses = 0;
    let once = upstream
        .lines()
        .filter(|line| {
            let parsing = line.trim().starts_with("on_adapter_parse_");
            parses += usize::from(parsing);
            !parsing || parses <= 2
        })
        .collect::<Vec<_>>()
        .join("\n");
    assert_eq!(recording.tree(), once, "\nupstream at n=3 was:\n{upstream}");
}

/// Two predictors under one module: both children name the same parent, and neither names the
/// other's — the property upstream's `test_active_id` asserts.
#[allow(clippy::await_holding_lock)] // the installer's own note: `SERIAL` is a test token, taken by nothing under test
#[tokio::test]
async fn two_predictors_under_one_module_share_a_parent() {
    #[derive(Module)]
    struct Nested {
        first: dsrust::predict::Predict,
        second: dsrust::predict::Predict,
    }

    impl Forward for Nested {
        async fn forward(&self, inputs: Example) -> anyhow::Result<Prediction> {
            let first = self.first.forward(inputs).await?;
            let answer = first.get("answer").and_then(Value::as_str).unwrap_or("");
            self.second
                .forward(example! { question: answer }.with_inputs(["question"]))
                .await
        }
    }

    let recording = Arc::new(Recording::default());
    let lm = Arc::new(DummyLM::keyed([
        ("How are you?", example! { answer: "second question" }),
        ("second question", example! { answer: "test output" }),
    ]));
    let _serial = install(lm, recording.clone());

    Module::forward(
        &Nested {
            first: dsrust::predict::Predict::from_signature(signature()),
            second: dsrust::predict::Predict::from_signature(signature()),
        },
        asked(),
    )
    .await
    .expect("both scripted answers parse");
    configure_callbacks([]);

    assert_eq!(recording.tree(), expected("nested_modules"));
}

/// A devset run, with the module calls it made inside it — the outermost point, and the one whose
/// nesting a `span.enter()` held across an await would get wrong.
#[allow(clippy::await_holding_lock)] // the installer's own note: `SERIAL` is a test token, taken by nothing under test
#[tokio::test]
async fn an_evaluation_encloses_the_calls_it_made() {
    let recording = Arc::new(Recording::default());
    let lm = Arc::new(DummyLM::new([example! { answer: "test output" }]));
    let _serial = install(lm, recording.clone());

    let devset = vec![
        example! { question: "How are you?", answer: "test output" }.with_inputs(["question"]),
    ];
    let predict = dsrust::predict::Predict::from_signature(signature());
    Evaluate::new(
        devset,
        |inputs: Example| predict.forward(inputs),
        |_example: &Example, _prediction: &Prediction| 1.0,
    )
    .run()
    .await
    .expect("no row fails, so the evaluation runs to completion");
    configure_callbacks([]);

    assert_eq!(recording.tree(), expected("evaluate"));
}

/// Past `max_errors` the run abandons the devset — and the point still closes.
///
/// The sequence is the fixture's, because upstream's decorator fires the end handler on the way out
/// with the exception rather than leaving the start unanswered. The crate returned from inside the
/// open point instead, so a handler that had been told an evaluation began was never told it ended.
#[allow(clippy::await_holding_lock)] // the installer's own note: `SERIAL` is a test token, taken by nothing under test
#[tokio::test]
async fn an_abandoned_evaluation_still_closes_its_point() {
    let recording = Arc::new(Recording::default());
    // The model call itself errors, which is the one failure both ports produce the same way. A
    // parse failure would take each side down its JSON-fallback path instead.
    let lm = Arc::new(FailingLM);
    let _serial = install(lm, recording.clone());

    let devset = vec![
        example! { question: "How are you?", answer: "test output" }.with_inputs(["question"]),
    ];
    let predict = dsrust::predict::Predict::from_signature(signature());
    let gave_up = Evaluate::new(
        devset,
        |inputs: Example| predict.forward(inputs),
        |_example: &Example, _prediction: &Prediction| 1.0,
    )
    .max_errors(1)
    .run()
    .await;
    configure_callbacks([]);

    assert_eq!(
        gave_up.unwrap_err().to_string(),
        "Execution cancelled due to errors or interruption."
    );
    assert_eq!(recording.tree(), expected("evaluate_abandoned"));
}

/// A tool call is watched, which no fixture case reaches: dspy's own tool point fires from
/// `dspy.Tool.__call__`, and the crate's agents go through `observe::tool_call` instead.
#[tokio::test]
async fn a_tool_call_is_watched() {
    let recording = Arc::new(Recording::default());
    let lm = Arc::new(DummyLM::new([example! { answer: "unused" }]));
    let _serial = install(lm, recording.clone());

    let tool = dsrust::FnTool::new(
        "weather",
        "the weather",
        serde_json::json!({ "type": "object", "properties": { "city": { "type": "string" } } }),
        |args: &Value| {
            Ok(format!(
                "sunny in {}",
                args["city"].as_str().unwrap_or_default()
            ))
        },
    );
    let answered = dsrust::observe::tool_call(&tool, &serde_json::json!({ "city": "Paris" }))
        .expect("the tool answers");
    configure_callbacks([]);

    assert_eq!(answered, serde_json::json!("sunny in Paris"));
    assert_eq!(recording.tree(), "on_tool_start\non_tool_end");
}

/// The awaited twin fires the same two handlers, which is what says a tool that suspends is still
/// watched: the point has to open before the future is polled and close after it resolves, and a
/// span guard held across an await would report the wrong parent or none.
#[allow(clippy::await_holding_lock)] // the installer's own note: `SERIAL` is a test token, taken by nothing under test
#[tokio::test]
async fn an_awaited_tool_call_is_watched() {
    let recording = Arc::new(Recording::default());
    let lm = Arc::new(DummyLM::new([example! { answer: "unused" }]));
    let _serial = install(lm, recording.clone());

    let tool = dsrust::AsyncFnTool::new(
        "weather",
        "the weather",
        serde_json::json!({ "type": "object", "properties": { "city": { "type": "string" } } }),
        |args: Value| async move {
            tokio::task::yield_now().await;
            Ok(format!(
                "sunny in {}",
                args["city"].as_str().unwrap_or_default()
            ))
        },
    );
    let answered = dsrust::observe::tool_acall(&tool, &serde_json::json!({ "city": "Paris" }))
        .await
        .expect("the tool answers");
    configure_callbacks([]);

    assert_eq!(answered, serde_json::json!("sunny in Paris"));
    assert_eq!(recording.tree(), "on_tool_start\non_tool_end");
}

/// `lm.call([…])` is dspy's `BaseLM.__call__`, which is the method upstream decorates — so asking a
/// model directly fires the lm point exactly as asking it through a module does.
///
/// It did not: `call` reached `ChatModel::forward` rather than the blanket `forward_dyn`, so the one
/// entry named after upstream's decorated method was the one entry with no point on it.
#[allow(clippy::await_holding_lock)] // the installer's own note: `SERIAL` is a test token, taken by nothing under test
#[tokio::test]
async fn asking_a_model_directly_fires_the_lm_point() {
    use dsrust::lm::ChatModel;

    let recording = Arc::new(Recording::default());
    let lm = Arc::new(DummyLM::new([example! { answer: "test output" }]));
    let _serial = install(lm.clone(), recording.clone());

    lm.call(["What is the capital of France?"])
        .await
        .expect("the scripted answer comes back");
    configure_callbacks([]);

    assert_eq!(recording.tree(), "on_lm_start\non_lm_end");
}

/// dspy 3.3.1's interpreter points, driven the way a module outside this crate would: `observe`'s
/// helpers are public so a caller's own interpreter lands on the same callback tree as this one's,
/// and nothing but this says a caller can reach them.
///
/// The nesting is upstream's — a tool the sandboxed code called happens *inside* the execution,
/// and the sandbox coming up is its own point — so the tree is what the handlers are given.
#[allow(clippy::await_holding_lock)] // the installer's own note: `SERIAL` is a test token, taken by nothing under test
#[tokio::test]
async fn a_callers_own_interpreter_reaches_the_interpreter_points() {
    use dsrust::observe::{
        InterpreterLifecycle, executing, interpreter_lifecycle, interpreter_tool_call,
    };

    let recording = Arc::new(Recording::default());
    let lm: Arc<dyn dsrust::lm::DynChatModel> = Arc::new(DummyLM::new([example! { answer: "x" }]));
    let _serial = install(lm, recording.clone());

    interpreter_lifecycle(InterpreterLifecycle::Startup, "OwnInterpreter", || Ok(()))
        .expect("starts");
    executing("OwnInterpreter", "lookup('cats')", || {
        let observed =
            interpreter_tool_call("lookup", &serde_json::json!({ "q": "cats" }), || {
                Ok(Value::String("found".to_owned()))
            })?;
        assert_eq!(observed, Value::String("found".to_owned()));
        Ok(dsrust::interpreter::Executed::Printed(Value::String(
            "done".to_owned(),
        )))
    })
    .expect("runs");
    interpreter_lifecycle(InterpreterLifecycle::Shutdown, "OwnInterpreter", || Ok(()))
        .expect("stops");

    assert_eq!(
        recording.tree(),
        "on_interpreter_startup_start\n\
         on_interpreter_startup_end\n\
         on_interpreter_execute_start\n\
         \u{20}\u{20}on_interpreter_tool_call_start\n\
         \u{20}\u{20}on_interpreter_tool_call_end\n\
         on_interpreter_execute_end\n\
         on_interpreter_shutdown_start\n\
         on_interpreter_shutdown_end"
    );
}

/// The compile point for an optimizer that neither awaits nor fails — dspy wraps every
/// `Teleprompter.compile` the same way, and `compiling_sync` is the half for the ones that answer
/// without a `Result`.
#[allow(clippy::await_holding_lock)] // as above
#[tokio::test]
async fn a_synchronous_compile_fires_the_same_pair() {
    let recording = Arc::new(Recording::default());
    let lm: Arc<dyn dsrust::lm::DynChatModel> = Arc::new(DummyLM::new([example! { answer: "x" }]));
    let _serial = install(lm, recording.clone());

    let trainset =
        vec![example! { question: "How are you?", answer: "fine" }.with_inputs(["question"])];
    let mut student = dsrust::Predict::from_signature(signature());
    dsrust::optimize::LabeledFewShot::new(1).compile(&mut student, &trainset);

    assert_eq!(recording.tree(), "on_compile_start\non_compile_end");
}
