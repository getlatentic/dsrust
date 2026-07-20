//! A tool's argument schema reaching the model.
//!
//! A name and a description tell the model that a tool exists; only the schema tells it what
//! to send. dspy renders `Tool.args` into the ReAct instructions for exactly that reason, so
//! these tests follow the schema all the way to the system prompt a provider would receive,
//! rather than stopping at the signature that produced it.

use std::sync::{Arc, Mutex};

use dsrs::lm::global;
use dsrs::react::tool_args;
use dsrs::signature::{FieldKind, OutField, Signature};
use dsrs::{DummyLM, FnTool, Module, ReAct, Tool, example, react::arg_str};
use schemars::JsonSchema;
use serde_json::{Value, json};

/// The arguments `get_weather` parses, so the schema shown to the model and the code reading
/// it come from one declaration.
#[derive(JsonSchema)]
#[allow(dead_code)]
struct WeatherArgs {
    city: String,
}

fn weather() -> Box<dyn Tool> {
    Box::new(FnTool::new(
        "get_weather",
        "look up the weather for a city",
        tool_args::<WeatherArgs>(),
        |args: &Value| {
            Ok(format!(
                "The weather in {} is sunny.",
                arg_str(args, "city")?
            ))
        },
    ))
}

/// A tool that takes nothing at all, which dspy renders as an empty argument object rather
/// than omitting the clause.
fn clock() -> Box<dyn Tool> {
    Box::new(FnTool::new(
        "current_time",
        "read the current time",
        json!({}),
        |_: &Value| Ok("12:00".to_owned()),
    ))
}

fn task() -> Signature {
    Signature::single_input(
        "Answer the question.",
        vec![OutField {
            name: "answer",
            desc: "the answer".into(),
            kind: FieldKind::Str,
            values: None,
            schema: None,
        }],
    )
}

/// The per-turn instructions, read the way an optimizer would rather than through a private
/// field.
fn turn_instructions(react: &mut ReAct) -> String {
    react
        .named_predictors()
        .into_iter()
        .find(|predictor| predictor.name == "react")
        .expect("the per-turn predictor is named")
        .signature
        .instructions
        .clone()
}

/// The configured model is process-wide, so these tests take turns.
static GLOBAL_LM: Mutex<()> = Mutex::new(());

fn install(lm: Arc<DummyLM>) -> std::sync::MutexGuard<'static, ()> {
    let guard = GLOBAL_LM
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    global::configure_model(reqwest::Client::new(), lm);
    guard
}

#[test]
fn a_derived_schema_reaches_the_turn_instructions() {
    let mut react = ReAct::new(task(), vec![weather()]);
    assert!(
        turn_instructions(&mut react).contains(
            "(1) get_weather, whose description is <desc>look up the weather for a city</desc>. \
             It takes arguments {'city': {'type': 'string'}}."
        ),
        "the model is told the argument name and its type"
    );
}

#[test]
fn a_tool_with_no_arguments_still_declares_an_empty_object() {
    let mut react = ReAct::new(task(), vec![clock()]);
    assert!(turn_instructions(&mut react).contains(
        "(1) current_time, whose description is <desc>read the current time</desc>. \
         It takes arguments {}."
    ));
}

#[test]
fn every_tool_is_numbered_in_the_order_supplied_with_finish_last() {
    // dspy numbers a dict built from the caller's list, so the catalogue follows the order the
    // tools were handed over. Sorting them would renumber the prompt the model reads.
    let mut react = ReAct::new(task(), vec![weather(), clock()]);
    let instructions = turn_instructions(&mut react);
    let catalogue: Vec<&str> = instructions
        .lines()
        .filter(|line| line.starts_with('('))
        .collect();
    assert_eq!(
        catalogue,
        [
            "(1) get_weather, whose description is <desc>look up the weather for a city</desc>. \
             It takes arguments {'city': {'type': 'string'}}.",
            "(2) current_time, whose description is <desc>read the current time</desc>. \
             It takes arguments {}.",
            "(3) finish, whose description is <desc>Marks the task as complete. That is, \
             signals that all information for producing the outputs, i.e. `answer`, are now \
             available to be extracted.</desc>. It takes arguments {}.",
        ]
    );
}

#[tokio::test]
async fn the_schema_reaches_the_prompt_the_model_is_actually_sent() {
    let lm = Arc::new(DummyLM::new([
        example! { next_thought: "no lookup needed", next_tool_name: "finish", next_tool_args: json!({}) },
        example! { reasoning: "no lookup was needed", answer: "It is sunny." },
    ]));
    let _guard = install(lm.clone());

    let react = ReAct::new(task(), vec![weather()]);
    react
        .forward(example! { request: "weather in Tokyo?" }.with_inputs(["request"]))
        .await
        .expect("the episode completes");

    assert!(
        lm.asked()[0]
            .system
            .contains("It takes arguments {'city': {'type': 'string'}}."),
        "the schema is in the system prompt, not just the signature"
    );
}

#[tokio::test]
async fn a_call_with_a_bad_argument_becomes_an_observation_rather_than_aborting() {
    // dspy reports the failure into the trajectory so the model can correct itself against
    // the schema it was shown; aborting would discard the reasoning that got this far.
    let lm = Arc::new(DummyLM::new([
        example! { next_thought: "guessing the argument", next_tool_name: "get_weather", next_tool_args: json!({ "town": "Tokyo" }) },
        example! { next_thought: "the error names the real argument", next_tool_name: "get_weather", next_tool_args: json!({ "city": "Tokyo" }) },
        example! { next_thought: "now I can answer", next_tool_name: "finish", next_tool_args: json!({}) },
        example! { reasoning: "the tool said sunny", answer: "It is sunny in Tokyo." },
    ]));
    let _guard = install(lm.clone());

    let react = ReAct::new(task(), vec![weather()]);
    let prediction = react
        .forward(example! { request: "weather in Tokyo?" }.with_inputs(["request"]))
        .await
        .expect("a bad argument does not end the episode");

    assert_eq!(
        prediction.get("answer").and_then(Value::as_str),
        Some("It is sunny in Tokyo.")
    );
    let asked = lm.asked();
    assert!(
        asked[1]
            .last_message()
            .contains("Execution error in get_weather"),
        "the model reads its own mistake"
    );
    assert!(
        asked[1]
            .last_message()
            .contains("missing string argument `city`"),
        "the error names the argument the schema declared"
    );
    assert!(
        asked[2]
            .last_message()
            .contains("The weather in Tokyo is sunny."),
        "the corrected call runs"
    );
}
