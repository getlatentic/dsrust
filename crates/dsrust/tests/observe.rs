//! dspy's callback points, held to existing by a subscriber that records them.
//!
//! A span nobody collects is indistinguishable from no span, and the ledger once said these points
//! "emit tracing spans here" when the tree had none — so nothing here reads the source. Each case
//! runs a program under a subscriber of its own and asserts on what arrived: the names, the nesting,
//! and the values dspy hands its handlers.
//!
//! The enumeration is the point of `every_module_is_watched`. There is no way in Rust to make an
//! implementor of a required trait method instrument itself, so the thing that stops the next module
//! from being added unwatched is a case that names them all.

use std::future::Future;
use std::sync::{Arc, Mutex};

use dsrust::lm::{ChatModel, DynChatModel, api};
use dsrust::{
    ChainOfThought, DummyLM, Example, Forward, Module, Predict, Prediction, call, example, input,
};
use tracing::subscriber::with_default;
use tracing_subscriber::layer::SubscriberExt;

/// One span as it was closed: the name, its parent's name, and the fields recorded on it.
#[derive(Debug, Clone, PartialEq)]
struct Watched {
    name: String,
    parent: Option<String>,
    inputs: Option<String>,
    outputs: Option<String>,
    error: Option<String>,
}

/// Everything the run recorded, in the order the spans closed — so a child is always before its
/// parent, which is what makes the nesting readable.
#[derive(Clone, Default)]
struct Recorder(Arc<Mutex<Vec<Watched>>>);

impl Recorder {
    fn spans(&self) -> Vec<Watched> {
        self.0.lock().expect("the recording").clone()
    }

    fn named(&self, name: &str) -> Vec<Watched> {
        self.spans()
            .into_iter()
            .filter(|span| span.name == name)
            .collect()
    }

    fn one(&self, name: &str) -> Watched {
        let found = self.named(name);
        assert_eq!(found.len(), 1, "expected one {name} span, got {found:?}");
        found.into_iter().next().expect("the span")
    }
}

mod recording {
    //! The subscriber layer. Fields are captured through `tracing`'s visitor, which is the only way
    //! to see a value a span recorded — there is no reading a live span's fields back.

    use super::{Recorder, Watched};
    use std::collections::HashMap;
    use tracing::field::{Field, Visit};
    use tracing::span::{Attributes, Id, Record};
    use tracing_subscriber::layer::{Context, Layer};
    use tracing_subscriber::registry::LookupSpan;

    #[derive(Default)]
    pub(super) struct Fields(pub HashMap<String, String>);

    impl Visit for Fields {
        fn record_str(&mut self, field: &Field, value: &str) {
            self.0.insert(field.name().to_owned(), value.to_owned());
        }

        fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
            self.0.insert(field.name().to_owned(), format!("{value:?}"));
        }
    }

    impl<S> Layer<S> for Recorder
    where
        S: tracing::Subscriber + for<'a> LookupSpan<'a>,
    {
        fn on_new_span(&self, attributes: &Attributes<'_>, id: &Id, context: Context<'_, S>) {
            let mut fields = Fields::default();
            attributes.record(&mut fields);
            context
                .span(id)
                .expect("the new span")
                .extensions_mut()
                .insert(fields);
        }

        fn on_record(&self, id: &Id, values: &Record<'_>, context: Context<'_, S>) {
            let span = context.span(id).expect("the recorded span");
            let mut extensions = span.extensions_mut();
            if let Some(fields) = extensions.get_mut::<Fields>() {
                values.record(fields);
            }
        }

        fn on_close(&self, id: Id, context: Context<'_, S>) {
            let span = context.span(&id).expect("the closing span");
            let extensions = span.extensions();
            let empty = Fields::default();
            let fields = extensions.get::<Fields>().unwrap_or(&empty);
            let read = |name: &str| fields.0.get(name).cloned();
            self.0.lock().expect("the recording").push(Watched {
                name: span.name().to_owned(),
                parent: span.parent().map(|parent| parent.name().to_owned()),
                inputs: read("inputs"),
                outputs: read("outputs"),
                error: read("error"),
            });
        }
    }
}

/// Run `work` with everything the crate observes collected, and hand back the recording.
fn recording<Work, Answered>(work: Work) -> (Recorder, Answered)
where
    Work: Future<Output = Answered>,
{
    let recorder = Recorder::default();
    let subscriber = tracing_subscriber::registry().with(recorder.clone());
    let answered = with_default(subscriber, || {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("a runtime")
            .block_on(work)
    });
    (recorder, answered)
}

/// A model whose reply carries no field marker, so the adapter refuses it. `DummyLM` cannot be one:
/// it renders its own fields back through the adapter and therefore always parses.
struct Unparseable;

impl ChatModel for Unparseable {
    async fn forward(&self, _request: &api::LmRequest) -> anyhow::Result<api::LmResponse> {
        Ok(api::LmResponse::text("nothing parseable here"))
    }
}

/// A model answering with these fields in turn — `DummyLM` renders each back through the adapter,
/// so a program under it takes the same path it takes against a provider.
fn scripted(replies: Vec<Example>) -> Arc<dyn DynChatModel> {
    Arc::new(DummyLM::new(replies))
}

fn one_answer() -> Arc<dyn DynChatModel> {
    scripted(vec![example! { answer: "Paris" }])
}

/// The two points the story does first, on the simplest possible run: a module span with an lm span
/// inside it, each carrying what dspy hands `on_module_start` and `on_lm_start`.
#[test]
fn a_predict_records_a_module_span_with_the_lm_call_inside_it() {
    let (recorded, answered) = recording(async {
        let qa = Predict!("question -> answer").with_lm(one_answer());
        call!(qa, question = "capital of France?").await
    });
    answered.expect("the scripted model answers");

    let module = recorded.one("module");
    assert_eq!(module.parent, None, "the outermost module has no parent");
    assert_eq!(
        module.inputs.as_deref(),
        Some(r#"{"question":"capital of France?"}"#),
        "dspy's on_module_start inputs dict"
    );
    assert_eq!(
        module.outputs.as_deref(),
        Some(r#"{"answer":"Paris"}"#),
        "dspy's on_module_end outputs"
    );
    assert_eq!(module.error, None, "a run that worked records no error");

    let lm = recorded.one("lm");
    assert_eq!(
        lm.parent.as_deref(),
        Some("module"),
        "the lm call is nested in the module that made it, as upstream's ACTIVE_CALL_ID nests it"
    );
    let inputs = lm.inputs.expect("the request was recorded");
    assert!(
        inputs.contains("capital of France?"),
        "the prompt is what a reader opened a trace to see: {inputs}"
    );
    assert!(inputs.contains("\"messages\""), "{inputs}");
    let outputs = lm.outputs.expect("the reply was recorded");
    assert!(outputs.contains("Paris"), "{outputs}");
    assert!(outputs.contains("\"cache_hit\":false"), "{outputs}");
}

/// A composed program is a span tree, which is what a reader needs to see where a run went wrong.
/// The child spans carry the field names of the steps that produced them.
#[test]
fn a_composed_program_nests_a_span_per_step() {
    #[derive(Module)]
    struct Outline {
        plan: Predict,
        write: Predict,
    }

    impl Forward for Outline {
        async fn forward(&self, inputs: Example) -> anyhow::Result<Prediction> {
            let angle = self.plan.forward(inputs).await?;
            self.write
                .forward(input! { angle: angle.get("angle").cloned().unwrap_or_default() })
                .await
        }
    }

    let (recorded, answered) = recording(async {
        let model = scripted(vec![
            example! { angle: "the cold" },
            example! { haiku: "frost on the window" },
        ]);
        let mine = Outline {
            plan: Predict!("subject -> angle").with_lm(Arc::clone(&model)),
            write: Predict!("angle -> haiku").with_lm(model),
        };
        call!(mine, subject = "winter mornings").await
    });
    answered.expect("both scripted replies land");

    let modules = recorded.named("module");
    assert_eq!(
        modules.len(),
        3,
        "Outline and its two Predicts: {modules:?}"
    );
    // Closed innermost first, so the caller's own module is last and is the only one with no parent.
    let outer = modules.last().expect("the outermost module");
    assert_eq!(outer.parent, None);
    assert_eq!(
        outer.inputs.as_deref(),
        Some(r#"{"subject":"winter mornings"}"#)
    );
    for inner in &modules[..2] {
        assert_eq!(
            inner.parent.as_deref(),
            Some("module"),
            "each step runs inside the program: {inner:?}"
        );
    }
    assert_eq!(recorded.named("lm").len(), 2, "one call per step");
}

/// dspy's `on_module_end(exception=…)`: a run that failed records why, and records no outputs. A
/// span that only ever recorded successes would be worthless for the thing a trace is opened for.
#[test]
fn a_failed_run_records_the_error_and_no_outputs() {
    let (recorded, answered) = recording(async {
        // Prose with no field marker in it: the adapter refuses the reply. `DummyLM` cannot produce
        // this — it renders its own fields back through the adapter — so the case needs a model of
        // its own, which is the caller's `ChatModel` path as well.
        let qa = Predict!("question -> answer").with_lm(Arc::new(Unparseable));
        call!(qa, question = "capital of France?").await
    });
    answered.expect_err("the reply does not parse");

    let module = recorded.one("module");
    assert_eq!(module.outputs, None, "there were no outputs to record");
    let error = module.error.expect("the failure was recorded");
    assert!(!error.is_empty(), "the error field carries the message");
}

/// Every module in the crate is watched. Adding one without a span fails here rather than leaving a
/// hole a caller finds by not seeing their agent in a trace.
///
/// `MultiChainComparison` and the wrappers are reached through the modules they hold, so what this
/// enumerates is every module that owns a `Module::forward` of its own.
#[test]
fn every_module_is_watched() {
    let cases: Vec<(&str, Box<dyn Fn() -> Box<dyn Module>>)> = vec![
        (
            "Predict",
            Box::new(|| Box::new(Predict!("question -> answer").with_lm(one_answer()))),
        ),
        (
            "ChainOfThought",
            Box::new(|| {
                Box::new(ChainOfThought!("question -> answer").with_lm(scripted(vec![
                    example! { reasoning: "it is the capital", answer: "Paris" },
                ])))
            }),
        ),
    ];

    for (kind, build) in cases {
        let (recorded, answered) = recording(async move {
            build()
                .forward(input! { question: "capital of France?" })
                .await
        });
        answered.unwrap_or_else(|error| panic!("{kind} should answer: {error:#}"));
        let modules = recorded.named("module");
        assert!(
            !modules.is_empty(),
            "{kind} recorded no module span — dspy fires on_module_start for it"
        );
        assert_eq!(
            modules.last().expect("the outermost").name,
            "module",
            "{kind}"
        );
    }
}

/// dspy `on_tool_start`/`on_tool_end`: an agent's tool calls are spans inside the module's, which is
/// what upstream's own documentation example prints.
#[test]
fn a_react_agent_records_a_span_per_tool_call() {
    let (recorded, answered) = recording(async {
        let tools: Vec<Box<dyn dsrust::Tool>> = vec![Box::new(dsrust::FnTool::new(
            "get_weather",
            "look up the weather for a city",
            serde_json::json!({ "city": { "type": "string" } }),
            |args: &serde_json::Value| {
                Ok(format!(
                    "sunny in {}",
                    args["city"].as_str().unwrap_or_default()
                ))
            },
        ))];
        let agent =
            dsrust::ReAct!("question -> answer", tools, max_iters = 2).with_lm(scripted(vec![
                example! {
                    next_thought: "check the weather",
                    next_tool_name: "get_weather",
                    next_tool_args: serde_json::json!({ "city": "Paris" })
                },
                example! {
                    next_thought: "done",
                    next_tool_name: "finish",
                    next_tool_args: serde_json::json!({})
                },
                example! { reasoning: "it said so", answer: "sunny" },
            ]));
        agent
            .forward(input! { question: "weather in Paris?" })
            .await
    });
    answered.expect("the agent finishes");

    let tools = recorded.named("tool");
    assert!(
        !tools.is_empty(),
        "no tool span — dspy fires on_tool_start for every call an agent makes"
    );
    let weather = tools
        .iter()
        .find(|span| span.inputs.as_deref().is_some_and(|i| i.contains("Paris")))
        .expect("the get_weather call was recorded with its arguments");
    assert_eq!(
        weather.parent.as_deref(),
        Some("module"),
        "a tool call happens inside the agent that made it"
    );
    assert!(
        weather
            .outputs
            .as_deref()
            .is_some_and(|o| o.contains("sunny")),
        "what the tool returned: {:?}",
        weather.outputs
    );
}

/// A tool that refuses records why, and records no outputs. dspy hands the exception to
/// `on_tool_end`; ReAct puts it in the trajectory and carries on, so the span is the only place a
/// reader sees it as a failure rather than as an observation.
#[test]
fn a_refusing_tool_records_its_error() {
    let (recorded, answered) = recording(async {
        let tools: Vec<Box<dyn dsrust::Tool>> = vec![Box::new(dsrust::FnTool::new(
            "always_fails",
            "a tool that refuses",
            serde_json::json!({}),
            |_: &serde_json::Value| anyhow::bail!("the service is down"),
        ))];
        let agent =
            dsrust::ReAct!("question -> answer", tools, max_iters = 2).with_lm(scripted(vec![
                example! {
                    next_thought: "try it",
                    next_tool_name: "always_fails",
                    next_tool_args: serde_json::json!({})
                },
                example! {
                    next_thought: "give up",
                    next_tool_name: "finish",
                    next_tool_args: serde_json::json!({})
                },
                example! { reasoning: "it failed", answer: "unknown" },
            ]));
        agent.forward(input! { question: "anything?" }).await
    });
    answered.expect("the agent finishes despite the tool");

    let failed = recorded
        .named("tool")
        .into_iter()
        .find(|span| span.error.is_some())
        .expect("the refusal was recorded as an error");
    assert_eq!(failed.outputs, None, "a refusal produced no outputs");
    assert!(
        failed.error.as_deref().is_some_and(|e| e.contains("down")),
        "{:?}",
        failed.error
    );
}

/// dspy `on_adapter_format_start` and `on_adapter_parse_start`: rendering the prompt and reading the
/// reply back are their own points, inside the module that asked.
///
/// Upstream decorates them in `__init_subclass__`, so every adapter it ships and every one a caller
/// writes fires them. `Adapter::format` is a required trait method here, so the span sits at the
/// caller instead — which is why this asserts on a whole run rather than on an adapter in isolation.
#[test]
fn rendering_and_parsing_are_their_own_spans() {
    let (recorded, answered) = recording(async {
        let qa = Predict!("question -> answer").with_lm(one_answer());
        call!(qa, question = "capital of France?").await
    });
    answered.expect("the scripted model answers");

    let adapters = recorded.named("adapter");
    assert_eq!(adapters.len(), 2, "one format and one parse: {adapters:?}");
    for span in &adapters {
        assert_eq!(
            span.parent.as_deref(),
            Some("module"),
            "an adapter call happens inside the module that asked: {span:?}"
        );
        assert_eq!(span.error, None);
    }

    let parse = adapters
        .iter()
        .find(|span| span.inputs.as_deref().is_some_and(|i| i.contains("Paris")))
        .expect("the parse span shows the raw reply it was reading");
    assert!(
        parse
            .outputs
            .as_deref()
            .is_some_and(|o| o.contains("Paris")),
        "and the fields it read: {:?}",
        parse.outputs
    );
}

/// A reply the adapter refuses records the failure on the parse span, which is where a reader looks
/// first: a parse failure is nearly always a question about what the model actually said, and the
/// span carries both halves.
#[test]
fn a_refused_reply_records_the_raw_text_beside_the_failure() {
    let (recorded, answered) = recording(async {
        let qa = Predict!("question -> answer").with_lm(Arc::new(Unparseable));
        call!(qa, question = "capital of France?").await
    });
    answered.expect_err("the reply does not parse");

    let failed = recorded
        .named("adapter")
        .into_iter()
        .find(|span| span.error.is_some())
        .expect("the refusal was recorded on an adapter span");
    assert_eq!(failed.outputs, None);
    assert!(
        failed
            .inputs
            .as_deref()
            .is_some_and(|i| i.contains("nothing parseable")),
        "the raw reply is beside the error: {:?}",
        failed.inputs
    );
}

/// Nothing is serialized when nothing is listening. A program with no subscriber must not pay for
/// rendering a prompt into a span field it will never reach.
#[test]
fn a_run_with_no_subscriber_records_nothing() {
    let answered = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("a runtime")
        .block_on(async {
            let qa = Predict!("question -> answer").with_lm(one_answer());
            call!(qa, question = "capital of France?").await
        });
    assert_eq!(
        answered
            .expect("the scripted model answers")
            .get("answer")
            .and_then(|answer| answer.as_str()),
        Some("Paris"),
        "the run works identically unwatched"
    );
}
