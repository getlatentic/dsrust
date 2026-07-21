//! Complex field types end to end: a struct input, `Vec<String>` and `Vec<Struct>` inputs,
//! and a `Vec<Struct>` output declared on one derived signature, driven through a scripted
//! model — prompt rendering, JSON coercion, both retry layers, and the call macros.

use std::collections::VecDeque;
use std::sync::Mutex;

use anyhow::{Result, anyhow};
use dsrs::JsonAdapter;
use dsrs::lm::{self, ChatModel, ChatTurn, LM, LmRequest, LmResponse, OutputMode, Role};
use dsrs::signature::{Signature, SignatureSpec, chain_of_thought, json_field_schema, predict};
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
    async fn chat(&self, _http: &reqwest::Client, request: &LmRequest<'_>) -> Result<LmResponse> {
        self.calls.lock().expect("not poisoned").push(Call {
            system: request.system.to_owned(),
            turns: request.turns.to_vec(),
            json_mode: matches!(request.mode, OutputMode::Json { .. }),
        });
        self.replies
            .lock()
            .expect("not poisoned")
            .pop_front()
            .map(LmResponse::text)
            .ok_or_else(|| anyhow!("script exhausted"))
    }
}

#[test]
fn derive_maps_complex_field_types_to_json() {
    let signature = IdeasTask::signature();
    // The derive knows the Rust type but not the Python name dspy would print for it.
    let json = dsrs::signature::FieldKind::opaque_json();
    assert!(signature.inputs.iter().all(|field| field.kind == json));
    assert_eq!(signature.outputs[0].kind, json);
    assert_eq!(signature.outputs[1].kind, dsrs::signature::FieldKind::Str);

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
        (
            "recipient",
            json!({ "name": "Dad", "age": 61, "hobbies": ["fishing", "grilling"] })
        )
    );
    assert_eq!(pairs[1], ("themes", json!(["surprise"])));
    assert_eq!(
        pairs[2],
        ("past", json!([{ "title": "Socks", "why": "Warm" }]))
    );
}

#[tokio::test]
async fn prompts_annotate_json_fields_and_a_marker_reply_deserializes() {
    let lm = Scripted::new(&[&marker_reply(GOOD_IDEAS)]);
    let outputs = IdeasTask::predict()
        .call_inputs_with(&reqwest::Client::new(), &lm, &inputs())
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
    assert!(system.contains("1. `recipient` (json): who the gift is for\n"));
    assert!(system.contains("2. `themes` (json): keywords to build on\n"));
    assert!(
        system.contains("1. `ideas` (json): three concrete ideas\n"),
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
        .call_inputs_with(&reqwest::Client::new(), &lm, &inputs())
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
        .call_inputs_with(&reqwest::Client::new(), &lm, &inputs())
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
        .with_adapter(JsonAdapter)
        .call_inputs_with(&reqwest::Client::new(), &lm, &inputs())
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
        .call_inputs_with(&reqwest::Client::new(), &lm, &inputs())
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
        .call_inputs_with(&reqwest::Client::new(), &lm, &inputs())
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
        .with_adapter(JsonAdapter)
        .call_inputs_with(&reqwest::Client::new(), &lm, &inputs())
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
        .call_inputs_with(&reqwest::Client::new(), &lm, &inputs())
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
    expands_to_an_ideas_future(predict!(IdeasTask {
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
    expands_to_an_ideas_future(chain_of_thought!(IdeasTask {
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
        .with_env_filter("dsrs=warn")
        .try_init()
        .ok();
    let model =
        std::env::var("LIVE_LM").unwrap_or_else(|_| "openrouter/openai/gpt-oss-120b".into());
    lm::configure(LM::new(&model)?);
    let outputs = predict!(IdeasTask {
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
