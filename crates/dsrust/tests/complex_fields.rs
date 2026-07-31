//! Complex field types end to end: a struct input, `Vec<String>` and `Vec<Struct>` inputs,
//! and a `Vec<Struct>` output declared on one derived signature, driven through a scripted
//! model — prompt rendering, JSON coercion, both retry layers, and the call macros.

use dsrust::Adapter;
use dsrust::adapter::Input;
use std::collections::VecDeque;
use std::sync::Mutex;

use anyhow::{Result, anyhow};
use dsrust::JsonAdapter;
use dsrust::lm::api::{self, Content, content_of};
use dsrust::lm::{self, ChatModel, ChatTurn, LM, Role};
use dsrust::signature::{ChainOfThought, Predict, Signature, SignatureSpec, json_field_schema};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Clone, Serialize)]
struct Recipient {
    name: String,
    age: u32,
    hobbies: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
struct GiftIdea {
    title: String,
    why: String,
}

/// Suggest gift ideas.
// The derive is declaration data; the struct itself is never built.
#[allow(dead_code)]
#[derive(Signature)]
struct IdeasTask {
    #[input(desc = "who the gift is for")]
    recipient: Recipient,
    #[input(desc = "keywords to build on")]
    themes: Vec<String>,
    #[input(desc = "gifts already given")]
    past: Vec<GiftIdea>,
    #[output(desc = "three concrete ideas")]
    ideas: Vec<GiftIdea>,
    #[output(desc = "one closing tip")]
    tip: String,
}

fn inputs() -> IdeasTaskInputs {
    IdeasTaskInputs {
        recipient: Recipient {
            name: "Dad".into(),
            age: 61,
            hobbies: vec!["fishing".into(), "grilling".into()],
        },
        themes: vec!["surprise".into()],
        past: vec![GiftIdea {
            title: "Socks".into(),
            why: "Warm".into(),
        }],
    }
}

const GOOD_IDEAS: &str = r#"[{"title":"Fly rod","why":"He fishes at dawn"},{"title":"Grill set","why":"Sunday grilling"},{"title":"Boat day","why":"Time together"}]"#;

fn marker_reply(ideas: &str) -> String {
    format!("[[ ## ideas ## ]]\n{ideas}\n\n[[ ## tip ## ]]\nWrap it well.\n\n[[ ## completed ## ]]")
}

/// Scripted stand-in for a provider: pops one canned reply per call and records what each
/// call asked, so tests can assert on the retry conversation.
struct Scripted {
    replies: Mutex<VecDeque<String>>,
    calls: Mutex<Vec<Call>>,
}

#[derive(Clone)]
struct Call {
    system: String,
    turns: Vec<ChatTurn>,
    json_mode: bool,
}

impl Scripted {
    fn new(replies: &[&str]) -> Self {
        Self {
            replies: Mutex::new(replies.iter().map(|reply| (*reply).to_owned()).collect()),
            calls: Mutex::new(Vec::new()),
        }
    }

    fn calls(&self) -> Vec<Call> {
        self.calls.lock().expect("not poisoned").clone()
    }
}

impl ChatModel for Scripted {
    async fn forward(&self, request: &api::LmRequest) -> Result<api::LmResponse> {
        self.calls.lock().expect("not poisoned").push(Call {
            system: request.system().to_owned(),
            turns: recorded_turns(request),
            json_mode: request.output_schema().is_some(),
        });
        self.replies
            .lock()
            .expect("not poisoned")
            .pop_front()
            .map(api::LmResponse::text)
            .ok_or_else(|| anyhow!("script exhausted"))
    }
}

/// The non-system messages as the turns this test asserts on, each part collapsed to its prose.
fn recorded_turns(request: &api::LmRequest) -> Vec<ChatTurn> {
    request
        .messages
        .iter()
        .filter(|message| message.role != "system")
        .map(|message| ChatTurn {
            role: match message.role.as_str() {
                "assistant" => Role::Assistant,
                _ => Role::User,
            },
            content: content_of(&message.parts).unwrap_or_else(|_| Content::Text(String::new())),
        })
        .collect()
}

#[test]
fn derive_spells_complex_field_types_the_way_dspy_prints_them() {
    let signature = IdeasTask::signature();
    let annotation = |kind: &dsrust::signature::FieldKind| match kind {
        dsrust::signature::FieldKind::Json(json) => json.annotation.clone(),
        other => format!("{other:?}"),
    };
    let inputs: Vec<String> = signature
        .inputs
        .iter()
        .map(|field| annotation(&field.kind))
        .collect();
    assert_eq!(
        inputs,
        ["Recipient", "list[str]", "list[GiftIdea]"],
        "a declared type keeps its name and a Vec becomes a list, as dspy prints them"
    );
    assert_eq!(annotation(&signature.outputs[0].kind), "list[GiftIdea]");
    assert_eq!(signature.outputs[1].kind, dsrust::signature::FieldKind::Str);

    let ideas_schema = json_field_schema::<Vec<GiftIdea>>();
    assert_eq!(signature.outputs[0].schema.as_ref(), Some(&ideas_schema));
    assert_eq!(ideas_schema["type"], json!("array"));
    assert_eq!(ideas_schema["items"]["required"], json!(["title", "why"]));

    let schema = signature.schema();
    assert_eq!(schema["properties"]["ideas"], ideas_schema);
    assert_eq!(schema["properties"]["tip"], json!({ "type": "string" }));
    let rendered = schema.to_string();
    assert!(!rendered.contains("$ref"), "got: {rendered}");
    assert!(!rendered.contains("$schema"), "got: {rendered}");
}

#[test]
fn input_pairs_hand_complex_inputs_over_with_their_structure_intact() {
    // The adapter renders, so a field arrives as the value it is rather than as text. A
    // structured field could not otherwise expand into the turns a `History` needs.
    let pairs = IdeasTask::input_pairs(&inputs());
    assert_eq!(
        pairs[0],
        Input::record(
            "recipient",
            json!({ "name": "Dad", "age": 61, "hobbies": ["fishing", "grilling"] })
        )
    );
    assert_eq!(pairs[1], Input::new("themes", json!(["surprise"])));
    assert_eq!(
        pairs[2],
        Input::new("past", json!([{ "title": "Socks", "why": "Warm" }]))
    );
}

/// dspy renders a value differently depending on whether it *is* a model instance, so the derive
/// has to say which fields are. A struct is one; a `Vec` of them is not, and neither is a `Vec`
/// of strings — the same distinction `isinstance(value, BaseModel)` draws upstream.
#[test]
fn the_derive_marks_a_struct_field_as_a_record_and_a_collection_as_not() {
    let pairs = IdeasTask::input_pairs(&inputs());
    let marked: Vec<(&str, bool)> = pairs.iter().map(|i| (i.name, i.record)).collect();
    assert_eq!(
        marked,
        [("recipient", true), ("themes", false), ("past", false)]
    );
}

#[tokio::test]
async fn prompts_annotate_json_fields_and_a_marker_reply_deserializes() {
    let lm = Scripted::new(&[&marker_reply(GOOD_IDEAS)]);
    let outputs = IdeasTask::predict()
        .call_inputs_with(&lm, &inputs())
        .await
        .expect("valid reply");
    assert_eq!(outputs.ideas.len(), 3);
    assert_eq!(
        outputs.ideas[0],
        GiftIdea {
            title: "Fly rod".into(),
            why: "He fishes at dawn".into()
        }
    );
    assert_eq!(outputs.tip, "Wrap it well.");

    let calls = lm.calls();
    assert_eq!(calls.len(), 1);
    let system = &calls[0].system;
    assert!(system.contains("1. `recipient` (Recipient): who the gift is for\n"));
    assert!(system.contains("2. `themes` (list[str]): keywords to build on\n"));
    assert!(
        system.contains("1. `ideas` (list[GiftIdea]): three concrete ideas\n"),
        "got: {system}"
    );
    // The schema reaches the model through the slot note alone, spaced as `json.dumps` writes
    // it. Its shape is this crate's own — inlined where upstream would emit `$defs`/`$ref`.
    assert!(
        system.contains(
            "{ideas}        # note: the value you produce must adhere to the JSON schema: \
             {\"type\": \"array\", \"items\": {\"type\": \"object\", \"properties\": \
             {\"title\": {\"type\": \"string\"}, \"why\": {\"type\": \"string\"}}, \
             \"required\": [\"title\", \"why\"]}}"
        ),
        "got: {system}"
    );

    // `json.dumps` spacing, because the adapter renders the value rather than receiving text
    // some other serializer already wrote.
    let opening = calls[0].turns[0].content.text().unwrap();
    assert!(
        opening.contains("[[ ## recipient ## ]]\n{\"name\": \"Dad\", \"age\": 61"),
        "got: {opening}"
    );
    assert!(opening.contains("[[ ## past ## ]]\n[{\"title\": \"Socks\", \"why\": \"Warm\"}]"));
}

#[tokio::test]
async fn a_fenced_json_marker_section_still_parses() {
    let fenced = format!("```json\n{GOOD_IDEAS}\n```");
    let lm = Scripted::new(&[&marker_reply(&fenced)]);
    let outputs = IdeasTask::predict()
        .call_inputs_with(&lm, &inputs())
        .await
        .expect("valid reply");
    assert_eq!(outputs.ideas.len(), 3);
    assert_eq!(lm.calls().len(), 1);
}

#[tokio::test]
async fn invalid_json_rides_the_feedback_retry() {
    let bad = marker_reply("three lovely ideas, honest");
    let lm = Scripted::new(&[&bad, &marker_reply(GOOD_IDEAS)]);
    let outputs = IdeasTask::predict()
        .feedback_retry()
        .call_inputs_with(&lm, &inputs())
        .await
        .expect("second reply is valid");
    assert_eq!(outputs.ideas.len(), 3);

    let calls = lm.calls();
    assert_eq!(calls.len(), 2);
    let retry = &calls[1].turns;
    assert_eq!(retry[1].role, Role::Assistant);
    assert_eq!(retry[1].content.text().unwrap(), bad);
    assert!(
        retry[2]
            .content
            .text()
            .unwrap()
            .contains("ideas must be valid JSON")
    );
}

#[tokio::test]
async fn the_json_adapter_passes_native_arrays_through() {
    // The adapter is the caller's choice, so ask for JSON explicitly rather than arriving
    // there by accident after a failed parse.
    let native = format!(r#"{{ "ideas": {GOOD_IDEAS}, "tip": "Wrap it well." }}"#);
    let lm = Scripted::new(&[&native]);
    let outputs = IdeasTask::predict()
        .adapter(JsonAdapter::default())
        .call_inputs_with(&lm, &inputs())
        .await
        .expect("native json reply");
    assert_eq!(outputs.ideas.len(), 3);

    let calls = lm.calls();
    assert_eq!(calls.len(), 1);
    assert!(
        calls[0].json_mode,
        "the json adapter engages native structured output"
    );
}

#[tokio::test]
async fn a_shape_mismatch_gets_one_deep_retry_carrying_the_serde_error() {
    let shallow = marker_reply(r#"[{"title":"Fly rod"}]"#);
    let lm = Scripted::new(&[&shallow, &marker_reply(GOOD_IDEAS)]);
    let outputs = IdeasTask::predict()
        .call_inputs_with(&lm, &inputs())
        .await
        .expect("corrected reply deserializes");
    assert_eq!(outputs.ideas.len(), 3);

    let calls = lm.calls();
    assert_eq!(calls.len(), 2);
    let retry = &calls[1].turns;
    assert_eq!(retry.len(), 3);
    assert_eq!(retry[1].role, Role::Assistant);
    assert_eq!(retry[1].content.text().unwrap(), shallow);
    assert!(
        retry[2]
            .content
            .text()
            .unwrap()
            .contains("missing field `why`"),
        "got: {:?}",
        retry[2].content
    );
}

#[tokio::test]
async fn a_second_shape_failure_is_final_with_no_third_ask() {
    let shallow = marker_reply(r#"[{"title":"Fly rod"}]"#);
    let lm = Scripted::new(&[&shallow, &shallow]);
    let error = IdeasTask::predict()
        .call_inputs_with(&lm, &inputs())
        .await
        .expect_err("second bad shape is final");
    assert!(
        error
            .to_string()
            .contains("validated reply did not fit the requested type")
    );
    assert_eq!(lm.calls().len(), 2);
}

#[tokio::test]
async fn typed_calls_stay_bounded_at_three_provider_calls() {
    // Without an adapter fallback the ceiling is the ask plus one feedback retry per stage:
    // a validation failure, then a shape failure, and no more.
    let script = [
        format!(r#"{{ "ideas": {GOOD_IDEAS} }}"#),
        r#"{ "ideas": [{"title":"Fly rod"}], "tip": "Wrap it well." }"#.to_owned(),
        format!(r#"{{ "ideas": {GOOD_IDEAS}, "tip": "Wrap it well." }}"#),
    ];
    let script: Vec<&str> = script.iter().map(String::as_str).collect();
    let lm = Scripted::new(&script);
    let outputs = IdeasTask::predict()
        .feedback_retry()
        .adapter(JsonAdapter::default())
        .call_inputs_with(&lm, &inputs())
        .await
        .expect("third reply lands");
    assert_eq!(outputs.ideas.len(), 3);

    let calls = lm.calls();
    assert_eq!(calls.len(), 3);
    let modes: Vec<bool> = calls.iter().map(|call| call.json_mode).collect();
    assert_eq!(
        modes,
        [true, true, true],
        "the chosen adapter is used throughout"
    );
    assert!(
        calls[1]
            .turns
            .last()
            .expect("turns")
            .content
            .text()
            .unwrap()
            .contains("the tip field is missing")
    );
    assert!(
        calls[2]
            .turns
            .last()
            .expect("turns")
            .content
            .text()
            .unwrap()
            .contains("missing field `why`")
    );
}

#[tokio::test]
async fn chain_of_thought_deep_retry_keeps_the_full_previous_reply() {
    let reasoned_bad = format!(
        "[[ ## reasoning ## ]]\nthinking hard\n\n{}",
        marker_reply(r#"[{"title":"Fly rod"}]"#)
    );
    let reasoned_good = format!(
        "[[ ## reasoning ## ]]\nthinking again\n\n{}",
        marker_reply(GOOD_IDEAS)
    );
    let lm = Scripted::new(&[&reasoned_bad, &reasoned_good]);
    let outputs = IdeasTask::chain_of_thought()
        .call_inputs_with(&lm, &inputs())
        .await
        .expect("corrected reply deserializes");
    assert_eq!(outputs.ideas.len(), 3);
    assert_eq!(outputs.tip, "Wrap it well.");

    let calls = lm.calls();
    assert_eq!(calls.len(), 2);
    let retry = &calls[1].turns;
    assert_eq!(retry[1].content.text().unwrap(), reasoned_bad);
    assert!(
        retry[2]
            .content
            .text()
            .unwrap()
            .contains("missing field `why`")
    );
}

/// Pins down what a call macro evaluates to: the module call's future, yielding the task's
/// outputs. Constructing an async-fn future runs nothing, so the expansions typecheck and
/// drop here without a configured global.
fn expands_to_an_ideas_future<F>(_: F)
where
    F: std::future::Future<Output = Result<IdeasTaskOutputs>>,
{
}

#[test]
fn call_macros_take_struct_literals_and_vecs() {
    let recipient = Recipient {
        name: "Dad".into(),
        age: 61,
        hobbies: vec!["fishing".into()],
    };
    expands_to_an_ideas_future(Predict!(IdeasTask {
        recipient: Recipient {
            name: "Dad".into(),
            age: 61,
            hobbies: vec![],
        },
        themes: vec![],
        past: vec![GiftIdea {
            title: "Socks".into(),
            why: "Warm".into(),
        }],
    }));
    // An empty vec! literal infers its element type through the identity conversion; a
    // non-empty one must already hold the field's element type (String, not &str).
    expands_to_an_ideas_future(ChainOfThought!(IdeasTask {
        recipient: recipient,
        themes: vec!["surprise".to_owned()],
        past: vec![],
    }));
}

/// Live check, informative rather than a gate: does a real model fill a `Vec<Struct>`
/// output on the first try? Run from dspy/ with an OpenRouter key:
/// `OPENROUTER_API_KEY=... cargo test --test complex_fields -- --ignored --nocapture`
#[tokio::test]
#[ignore = "talks to a live provider; needs OPENROUTER_API_KEY"]
async fn live_complex_output() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("dsrust=warn")
        .try_init()
        .ok();
    let model =
        std::env::var("LIVE_LM").unwrap_or_else(|_| "openrouter/openai/gpt-oss-120b".into());
    lm::configure(LM::new(&model)?);
    let outputs = Predict!(IdeasTask {
        recipient: Recipient {
            name: "Dad".into(),
            age: 61,
            hobbies: vec!["fishing".into(), "grilling".into()],
        },
        themes: vec!["60th birthday".to_owned()],
        past: vec![GiftIdea {
            title: "Wool socks".into(),
            why: "His feet get cold".into(),
        }],
    })
    .await?;
    println!("tip: {}\nideas: {:#?}", outputs.tip, outputs.ideas);
    assert!(!outputs.ideas.is_empty());
    assert!(
        outputs
            .ideas
            .iter()
            .all(|idea| !idea.title.is_empty() && !idea.why.is_empty())
    );
    Ok(())
}

/// What the whole reflection tree exists for: BAML states a type rather than a schema of it, and
/// a Rust-declared type reached it as the bare word `json` until the derive started carrying its
/// shape.
///
/// The expectation is dspy 3.2.1's own output for the equivalent pydantic signature, taken by
/// running it rather than reasoned about. Nothing else in the suite pins this — dropping the
/// reflection from the derive leaves every other test passing.
#[test]
fn a_rust_type_reaches_baml_as_its_structure_rather_than_as_the_word_json() {
    let system = dsrust::BamlAdapter
        .system_message(&IdeasTask::signature())
        .expect("renders");
    let at = system.find("Output field").expect("an output type block");
    let end = system[at..]
        .find("[[ ## completed")
        .map_or(system.len(), |offset| at + offset);

    assert_eq!(
        system[at..end].trim_end(),
        "Output field `ideas` should be of type: [\n\
         \x20 {\n\
         \x20   title: string,\n\
         \x20   why: string,\n\
         \x20 }\n\
         ]\n\n\
         [[ ## tip ## ]]\n\
         Output field `tip` should be of type: string"
    );
}

/// A field that says nothing about itself contributes nothing to its line.
///
/// dspy stores the sentinel `${name}` for an undescribed field and drops it again when rendering
/// (`adapters/utils.py::get_field_description_string`), so a field's own name never reaches a
/// prompt. The derive used to substitute the name here, which put it on the end of every
/// undescribed field line — invisible to every fixture, because a fixture builds its `Signature`
/// from JSON rather than through the derive.
#[test]
fn an_undescribed_field_line_ends_at_the_colon() {
    #[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
    struct Note {
        body: String,
    }

    #[allow(dead_code)]
    #[derive(Signature)]
    /// Suggest a gift.
    struct Bare {
        #[input]
        recipient: Note,
        #[output]
        idea: String,
    }

    let system = dsrust::BamlAdapter
        .system_message(&Bare::signature())
        .expect("renders");
    assert!(
        system.contains("1. `recipient` (Note):\n"),
        "the name must not follow the colon; got: {system}"
    );
    assert!(!system.contains("(Note): recipient"));
}
