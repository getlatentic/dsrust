//! ReAct against Python DSPy 3.2.1, byte for byte.
//!
//! Every expected string here was produced by running `dspy.ReAct` over the same task and
//! copying what it built. The unit tests in `src/react/` check the pieces; these pin the
//! whole artefact — the per-turn instructions, both inner signatures, and the prompt a
//! provider actually receives — because a paraphrase of dspy's wording is a different prompt
//! and gets different behaviour out of a model.

use std::sync::{Arc, Mutex};

use dsrust::lm::global;
use dsrust::signature::{FieldKind, InField, OutField, Signature};
use dsrust::{DummyLM, FnTool, Module, ReAct, Tool, example};
use serde_json::{Value, json};

fn weather() -> Box<dyn Tool> {
    Box::new(FnTool::new(
        "get_weather",
        "look up the weather for a city",
        json!({ "city": { "type": "string" } }),
        |_: &Value| Ok("The weather in Tokyo is sunny.".to_owned()),
    ))
}

fn out(name: &str, kind: FieldKind) -> OutField {
    OutField {
        name: name.into(),
        kind,
        ..Default::default()
    }
}

/// `dspy.Signature("request: str -> answer: str").with_instructions("Answer the question.")`.
/// The field descriptions are empty because dspy's inline signatures carry none, and a
/// description this crate invented would show up in the prompt dspy never wrote.
fn task() -> Signature {
    Signature {
        instructions: "Answer the question.".to_owned(),
        inputs: vec![InField {
            name: "request".into(),
            ..Default::default()
        }],
        outputs: vec![out("answer", FieldKind::Str)],
    }
}

/// `dspy.Signature("question: str, context: str -> answer: str, confidence: float")`.
fn wide_task() -> Signature {
    Signature {
        instructions: "Do it.".to_owned(),
        inputs: vec![
            InField {
                name: "question".into(),
                ..Default::default()
            },
            InField {
                name: "context".into(),
                ..Default::default()
            },
        ],
        outputs: vec![
            out("answer", FieldKind::Str),
            out("confidence", FieldKind::Float),
        ],
    }
}

fn turn_signature(react: &mut ReAct) -> &mut Signature {
    react
        .named_predictors()
        .into_iter()
        .find(|predictor| predictor.name == "react")
        .expect("the per-turn predictor is named")
        .signature
}

fn extract_signature(react: &mut ReAct) -> &mut Signature {
    react
        .named_predictors()
        .into_iter()
        .find(|predictor| predictor.name == "extract")
        .expect("the extract predictor is named")
        .signature
}

fn field_names<'a>(fields: impl Iterator<Item = &'a str>) -> Vec<&'a str> {
    fields.collect()
}

/// Copied from `dspy.ReAct(sig, tools=[get_weather]).react.signature.instructions`.
const DSPY_TURN_INSTRUCTIONS: &str = "Answer the question.\n\
     \n\
     You are an Agent. In each episode, you will be given the fields `request` as input. And you can see your past trajectory so far.\n\
     Your goal is to use one or more of the supplied tools to collect any necessary information for producing `answer`.\n\
     \n\
     To do this, you will interleave next_thought, next_tool_name, and next_tool_args in each turn, and also when finishing the task.\n\
     After each tool call, you receive a resulting observation, which gets appended to your trajectory.\n\
     \n\
     When writing next_thought, you may reason about the current situation and plan for future steps.\n\
     When selecting the next_tool_name and its next_tool_args, the tool must be one of:\n\
     \n\
     (1) get_weather, whose description is <desc>look up the weather for a city</desc>. It takes arguments {'city': {'type': 'string'}}.\n\
     (2) finish, whose description is <desc>Marks the task as complete. That is, signals that all information for producing the outputs, i.e. `answer`, are now available to be extracted.</desc>. It takes arguments {}.\n\
     When providing `next_tool_args`, the value inside the field must be in JSON format";

#[test]
fn the_turn_instructions_are_dspys_word_for_word() {
    let mut react = ReAct::new(task(), vec![weather()]);
    assert_eq!(
        turn_signature(&mut react).instructions,
        DSPY_TURN_INSTRUCTIONS
    );
}

#[test]
fn every_field_name_in_the_instructions_is_backticked() {
    // dspy interpolates ``", ".join(f"`{k}`" ...)``, so a bare name is a divergence the model
    // can see. Several inputs and outputs put the joining to work as well.
    let mut react = ReAct::new(wide_task(), vec![weather()]);
    let instructions = turn_signature(&mut react).instructions.clone();
    assert!(
        instructions.contains(
            "You are an Agent. In each episode, you will be given the fields `question`, \
             `context` as input. And you can see your past trajectory so far.\n\
             Your goal is to use one or more of the supplied tools to collect any necessary \
             information for producing `answer`, `confidence`."
        ),
        "got: {instructions}"
    );
    assert!(
        instructions
            .contains("i.e. `answer`, `confidence`, are now available to be extracted.</desc>."),
        "got: {instructions}"
    );
}

#[test]
fn a_task_without_instructions_drops_the_leading_block() {
    // dspy builds `[f"{instructions}\n"] if instructions else []`, so an empty task opens on
    // "You are an Agent." rather than on a blank line.
    let mut bare = task();
    bare.instructions = String::new();
    let mut react = ReAct::new(bare, vec![weather()]);
    assert!(
        turn_signature(&mut react)
            .instructions
            .starts_with("You are an Agent. In each episode,"),
        "got: {}",
        turn_signature(&mut react).instructions
    );
}

#[test]
fn the_turn_signature_carries_dspys_fields_in_dspys_order() {
    let mut react = ReAct::new(task(), vec![weather()]);
    let signature = turn_signature(&mut react);

    assert_eq!(
        field_names(signature.inputs.iter().map(|field| field.name.as_str())),
        ["request", "trajectory"]
    );
    assert_eq!(
        field_names(signature.outputs.iter().map(|field| field.name.as_str())),
        ["next_thought", "next_tool_name", "next_tool_args"]
    );
}

#[test]
fn the_fields_react_adds_carry_no_description_of_their_own() {
    // dspy appends bare `dspy.InputField()`/`dspy.OutputField()`s, whose placeholder desc
    // renders as nothing. A description invented here would print in the system prompt.
    let mut react = ReAct::new(task(), vec![weather()]);
    let signature = turn_signature(&mut react);

    let trajectory = signature
        .inputs
        .iter()
        .find(|field| field.name == "trajectory")
        .expect("trajectory is an input");
    assert_eq!(trajectory.desc, "");
    for field in &signature.outputs {
        assert_eq!(field.desc, "", "{} carries a description", field.name);
    }
}

#[test]
fn the_tool_name_field_is_closed_over_the_tools_that_exist() {
    // dspy types it `Literal[tuple(tools.keys())]`, which is what puts the "must exactly
    // match" note in the prompt and rules out a hallucinated tool.
    let mut react = ReAct::new(task(), vec![weather()]);
    let signature = turn_signature(&mut react);
    let tool_name = signature
        .outputs
        .iter()
        .find(|field| field.name == "next_tool_name")
        .expect("next_tool_name is an output");
    assert_eq!(
        tool_name.values,
        Some(vec!["get_weather".into(), "finish".into()])
    );
}

#[test]
fn the_extract_signature_is_a_chain_of_thought_over_the_untouched_task() {
    // dspy builds `ChainOfThought(Signature({**inputs, **outputs}, instructions).append(
    // "trajectory", ...))`: the task's instructions carry through with nothing appended, and
    // `reasoning` leads the outputs.
    let mut react = ReAct::new(wide_task(), vec![weather()]);
    let signature = extract_signature(&mut react);

    assert_eq!(signature.instructions, "Do it.");
    assert_eq!(
        field_names(signature.inputs.iter().map(|field| field.name.as_str())),
        ["question", "context", "trajectory"]
    );
    assert_eq!(
        field_names(signature.outputs.iter().map(|field| field.name.as_str())),
        ["reasoning", "answer", "confidence"]
    );
}

#[test]
fn a_multi_step_trajectory_renders_exactly_as_dspy_renders_it() {
    // Copied from `react._format_trajectory(trajectory)` over the same two steps.
    let expected = "[[ ## thought_0 ## ]]\n\
         I should look it up\n\
         \n\
         [[ ## tool_name_0 ## ]]\n\
         get_weather\n\
         \n\
         [[ ## tool_args_0 ## ]]\n\
         {\"city\": \"Tokyo\"}\n\
         \n\
         [[ ## observation_0 ## ]]\n\
         The weather in Tokyo is sunny.\n\
         \n\
         [[ ## thought_1 ## ]]\n\
         Now I can answer\n\
         \n\
         [[ ## tool_name_1 ## ]]\n\
         finish\n\
         \n\
         [[ ## tool_args_1 ## ]]\n\
         {}\n\
         \n\
         [[ ## observation_1 ## ]]\n\
         Completed.";

    let lm = Arc::new(DummyLM::new([
        example! {
            next_thought: "I should look it up",
            next_tool_name: "get_weather",
            next_tool_args: json!({ "city": "Tokyo" })
        },
        example! {
            next_thought: "Now I can answer",
            next_tool_name: "finish",
            next_tool_args: json!({})
        },
        example! { reasoning: "It said sunny.", answer: "It is sunny in Tokyo." },
    ]));
    let _guard = install(lm.clone());

    let react = ReAct::new(task(), vec![weather()]);
    let prediction =
        block_on(react.forward(
            example! { request: "What is the weather in Tokyo?" }.with_inputs(["request"]),
        ));

    let asked = lm.asked();
    assert_eq!(asked.len(), 3, "two turns and the extract pass");
    assert_eq!(
        asked[2].last_message(),
        format!(
            "[[ ## request ## ]]\nWhat is the weather in Tokyo?\n\n[[ ## trajectory ## ]]\n{expected}\n\nRespond with the corresponding output fields, starting with the field `[[ ## reasoning ## ]]`, then `[[ ## answer ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`."
        )
    );
    assert_eq!(
        prediction.get("answer").and_then(Value::as_str),
        Some("It is sunny in Tokyo.")
    );
}

#[test]
fn the_first_turn_sees_an_empty_trajectory_field() {
    // dspy strips the formatted trajectory, so turn one carries an empty field rather than
    // blank lines the model has to interpret.
    let lm = Arc::new(DummyLM::new([
        example! { next_thought: "done", next_tool_name: "finish", next_tool_args: json!({}) },
        example! { reasoning: "nothing to do", answer: "ok" },
    ]));
    let _guard = install(lm.clone());

    let react = ReAct::new(task(), vec![weather()]);
    block_on(
        react.forward(
            example! { request: "What is the weather in Tokyo?" }.with_inputs(["request"]),
        ),
    );

    assert!(
        lm.asked()[0]
            .last_message()
            .starts_with("[[ ## request ## ]]\nWhat is the weather in Tokyo?\n\n[[ ## trajectory ## ]]\n\n\nRespond with"),
        "got: {}",
        lm.asked()[0].last_message()
    );
}

#[test]
fn the_turn_prompt_tells_the_model_which_tool_names_are_legal() {
    // dspy's Literal becomes this note in the system prompt; without the closed set the line
    // is missing entirely and nothing constrains the tool name.
    let lm = Arc::new(DummyLM::new([
        example! { next_thought: "done", next_tool_name: "finish", next_tool_args: json!({}) },
        example! { reasoning: "nothing to do", answer: "ok" },
    ]));
    let _guard = install(lm.clone());

    let react = ReAct::new(task(), vec![weather()]);
    block_on(react.forward(example! { request: "x" }.with_inputs(["request"])));

    assert!(
        lm.asked()[0].system().contains(
            "[[ ## next_tool_name ## ]]\n{next_tool_name}        \
             # note: the value you produce must exactly match (no extra characters) one of: \
             get_weather; finish"
        ),
        "got: {}",
        lm.asked()[0].system()
    );
}

/// Copied from `dspy.ChatAdapter().format(react.react.signature, ...)` over the same task.
/// dspy types the two fields it adds as `Literal[tuple(tools)]` and `dict[str, Any]`, and
/// prints those Python types beside the field names — the closed set is the annotation, not a
/// note appended after the description.
#[test]
fn the_turn_prompt_numbers_the_fields_with_dspys_python_annotations() {
    let lm = Arc::new(DummyLM::new([
        example! { next_thought: "done", next_tool_name: "finish", next_tool_args: json!({}) },
        example! { reasoning: "nothing to do", answer: "ok" },
    ]));
    let _guard = install(lm.clone());

    let react = ReAct::new(task(), vec![weather()]);
    block_on(react.forward(example! { request: "x" }.with_inputs(["request"])));

    assert!(
        lm.asked()[0].system().starts_with(
            "Your input fields are:\n\
             1. `request` (str): \n\
             2. `trajectory` (str):\n\
             Your output fields are:\n\
             1. `next_thought` (str): \n\
             2. `next_tool_name` (Literal['get_weather', 'finish']): \n\
             3. `next_tool_args` (dict[str, Any]):\n"
        ),
        "got: {}",
        lm.asked()[0].system()
    );
}

/// Copied from the same `dspy.ChatAdapter().format` call. pydantic turns `dict[str, Any]` into
/// an open object schema, and the slot states it — the only note among the three fields, since
/// `next_tool_name`'s closed set already speaks through its `Literal[...]` annotation.
#[test]
fn the_turn_prompts_argument_slot_carries_dspys_open_object_schema() {
    let lm = Arc::new(DummyLM::new([
        example! { next_thought: "done", next_tool_name: "finish", next_tool_args: json!({}) },
        example! { reasoning: "nothing to do", answer: "ok" },
    ]));
    let _guard = install(lm.clone());

    let react = ReAct::new(task(), vec![weather()]);
    block_on(react.forward(example! { request: "x" }.with_inputs(["request"])));

    assert!(
        lm.asked()[0].system().contains(
            "[[ ## next_tool_args ## ]]\n\
             {next_tool_args}        # note: the value you produce must adhere to the JSON \
             schema: {\"type\": \"object\", \"additionalProperties\": true}"
        ),
        "got: {}",
        lm.asked()[0].system()
    );
}

/// dspy's closing reminder repeats the Python type of every output that is not plain `str`,
/// which both fields ReAct adds are.
#[test]
fn the_turn_prompt_repeats_those_annotations_in_the_closing_reminder() {
    let lm = Arc::new(DummyLM::new([
        example! { next_thought: "done", next_tool_name: "finish", next_tool_args: json!({}) },
        example! { reasoning: "nothing to do", answer: "ok" },
    ]));
    let _guard = install(lm.clone());

    let react = ReAct::new(task(), vec![weather()]);
    block_on(react.forward(example! { request: "x" }.with_inputs(["request"])));

    assert!(
        lm.asked()[0].last_message().ends_with(
            "Respond with the corresponding output fields, starting with the field \
             `[[ ## next_thought ## ]]`, then `[[ ## next_tool_name ## ]]` (must be formatted \
             as a valid Python Literal['get_weather', 'finish']), then \
             `[[ ## next_tool_args ## ]]` (must be formatted as a valid Python dict[str, Any]), \
             and then ending with the marker for `[[ ## completed ## ]]`."
        ),
        "got: {}",
        lm.asked()[0].last_message()
    );
}

#[test]
fn the_episode_hands_back_the_trajectory_beside_the_answer() {
    // dspy returns `Prediction(trajectory=trajectory, **extract)`, so the caller can see the
    // tool calls, not only the conclusion.
    let lm = Arc::new(DummyLM::new([
        example! {
            next_thought: "look it up",
            next_tool_name: "get_weather",
            next_tool_args: json!({ "city": "Tokyo" })
        },
        example! { next_thought: "done", next_tool_name: "finish", next_tool_args: json!({}) },
        example! { reasoning: "sunny", answer: "It is sunny." },
    ]));
    let _guard = install(lm.clone());

    let react = ReAct::new(task(), vec![weather()]);
    let prediction = block_on(react.forward(example! { request: "x" }.with_inputs(["request"])));

    assert_eq!(
        prediction.get("trajectory"),
        Some(&json!({
            "thought_0": "look it up",
            "tool_name_0": "get_weather",
            "tool_args_0": { "city": "Tokyo" },
            "observation_0": "The weather in Tokyo is sunny.",
            "thought_1": "done",
            "tool_name_1": "finish",
            "tool_args_1": {},
            "observation_1": "Completed.",
        }))
    );
    // dspy's ChainOfThought reasoning is part of the returned prediction, not stripped.
    assert_eq!(
        prediction.get("reasoning").and_then(Value::as_str),
        Some("sunny")
    );
}

#[test]
fn the_budget_ends_the_loop_without_a_final_warning_turn() {
    // dspy falls straight out of `range(max_iters)` into extract: the last turn is an
    // ordinary one, and nothing tells the model its budget ran out.
    let lm = Arc::new(DummyLM::new([
        example! {
            next_thought: "t",
            next_tool_name: "get_weather",
            next_tool_args: json!({ "city": "A" })
        },
        example! {
            next_thought: "t",
            next_tool_name: "get_weather",
            next_tool_args: json!({ "city": "B" })
        },
        example! { reasoning: "r", answer: "a" },
    ]));
    let _guard = install(lm.clone());

    let react = ReAct::new(task(), vec![weather()]).max_iters(2);
    block_on(react.forward(example! { request: "x" }.with_inputs(["request"])));

    let asked = lm.asked();
    assert_eq!(asked.len(), 3, "two capped turns, then extract");
    assert_eq!(
        asked[1].system(),
        asked[0].system(),
        "the final turn is asked with the same instructions as the first"
    );
    assert!(
        asked[2].last_message().ends_with(
            "[[ ## observation_1 ## ]]\nThe weather in Tokyo is sunny.\n\nRespond with the \
             corresponding output fields, starting with the field `[[ ## reasoning ## ]]`, \
             then `[[ ## answer ## ]]`, and then ending with the marker for \
             `[[ ## completed ## ]]`."
        ),
        "got: {}",
        asked[2].last_message()
    );
}

/// The configured model is process-wide, so these tests take turns rather than racing to
/// overwrite each other's script.
static GLOBAL_LM: Mutex<()> = Mutex::new(());

fn install(lm: Arc<DummyLM>) -> std::sync::MutexGuard<'static, ()> {
    let guard = GLOBAL_LM
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    global::configure_model(reqwest::Client::new(), lm);
    guard
}

/// These cases assert on prompts, not on concurrency, so each drives its episode to
/// completion on a current-thread runtime while it holds the global model.
fn block_on<T>(future: impl Future<Output = anyhow::Result<T>>) -> T {
    tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("runtime builds")
        .block_on(future)
        .expect("the episode completes")
}
