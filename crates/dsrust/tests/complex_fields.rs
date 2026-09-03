//! Complex field types end to end: a struct input, `Vec<String>` and `Vec<Struct>` inputs,
//! and a `Vec<Struct>` output declared on one derived signature, driven through a scripted
//! model — prompt rendering, JSON coercion, both retry layers, and the call macros.

use dsrust::adapter::Input;
use std::collections::VecDeque;
use std::sync::Mutex;

use anyhow::{Result, anyhow};
use dsrust::JsonAdapter;
use dsrust::lm::ChatModel;
use dsrust::lm::api::{self, LmMessage};
use dsrust::signature::{Signature, SignatureSpec, json_field_schema};
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
/// One call as the model received it — the messages, which is what the request carries.
struct Call {
    messages: Vec<LmMessage>,
    json_mode: bool,
}

impl Call {
    fn system(&self) -> &str {
        api::system_of(&self.messages)
    }

    /// The conversation without the system prompt, which leads a render.
    fn turns(&self) -> &[LmMessage] {
        api::after_system(&self.messages)
    }
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
            messages: request.messages.clone(),
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
    // Hoisted, as pydantic hoists: the item is a reference and the model is a definition.
    assert_eq!(ideas_schema["items"]["$ref"], json!("#/$defs/GiftIdea"));
    assert_eq!(
        ideas_schema["$defs"]["GiftIdea"]["required"],
        json!(["title", "why"])
    );

    let schema = signature.schema();
    // The property is the field's schema with its definitions lifted out — same reference, one
    // `$defs` block for the document rather than one per field.
    assert_eq!(schema["properties"]["ideas"]["type"], ideas_schema["type"]);
    assert_eq!(
        schema["properties"]["ideas"]["items"],
        ideas_schema["items"]
    );
    assert_eq!(schema["properties"]["tip"], json!({ "type": "string" }));
    // The definitions are lifted to the root, so every `$ref` resolves against the document. Left
    // under the property that carried them, `#/$defs/GiftIdea` would point at nothing — a schema a
    // provider is right to reject, and one this assertion previously could not have caught because
    // nothing was hoisted at all.
    assert_eq!(
        schema["$defs"]["GiftIdea"]["title"],
        json!("GiftIdea"),
        "got: {schema}"
    );
    assert_eq!(
        schema["properties"]["ideas"]["items"]["$ref"],
        json!("#/$defs/GiftIdea")
    );
    assert_eq!(schema["properties"]["ideas"].get("$defs"), None);
    assert!(!schema.to_string().contains("$schema"), "got: {schema}");
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
    let system = &calls[0].system();
    assert!(system.contains("1. `recipient` (Recipient): who the gift is for\n"));
    assert!(system.contains("2. `themes` (list[str]): keywords to build on\n"));
    assert!(
        system.contains("1. `ideas` (list[GiftIdea]): three concrete ideas\n"),
        "got: {system}"
    );
    // The schema reaches the model through the slot note alone, spaced as `json.dumps` writes it —
    // and in pydantic's dialect, because dspy prints what pydantic produced. This assertion carried
    // the inlined, untitled shape until the two were rendered side by side; the comment beside it
    // said so, and nothing acted on it. Verified against `_get_json_schema(list[GiftIdea])`.
    assert!(
        system.contains(
            "{ideas}        # note: the value you produce must adhere to the JSON schema: \
             {\"type\": \"array\", \"$defs\": {\"GiftIdea\": {\"type\": \"object\", \
             \"properties\": {\"title\": {\"type\": \"string\", \"title\": \"Title\"}, \
             \"why\": {\"type\": \"string\", \"title\": \"Why\"}}, \"required\": \
             [\"title\", \"why\"], \"title\": \"GiftIdea\"}}, \"items\": \
             {\"$ref\": \"#/$defs/GiftIdea\"}}"
        ),
        "got: {system}"
    );

    // `json.dumps` spacing, because the adapter renders the value rather than receiving text
    // some other serializer already wrote.
    let opening = calls[0].turns()[0].text().unwrap();
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
    let retry = &calls[1].turns();
    assert_eq!(retry[1].role, "assistant");
    assert_eq!(retry[1].text().unwrap(), bad);
    assert!(
        retry[2]
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
async fn a_shape_mismatch_refuses_at_parse_and_re_asks_through_the_json_fallback() {
    // dspy 3.3.1 casts every field inside `parse`, so a `list[GiftIdea]` missing `why` is a parse
    // failure and `ChatAdapter.__call__` answers it by re-asking through `JSONAdapter` — measured
    // against dspy, which makes exactly these two calls for this reply.
    let shallow = marker_reply(r#"[{"title":"Fly rod"}]"#);
    let corrected = format!(r#"{{ "ideas": {GOOD_IDEAS}, "tip": "Wrap it well." }}"#);
    let lm = Scripted::new(&[&shallow, &corrected]);
    let outputs = IdeasTask::predict()
        .call_inputs_with(&lm, &inputs())
        .await
        .expect("the fallback's reply deserializes");
    assert_eq!(outputs.ideas.len(), 3);

    let calls = lm.calls();
    assert_eq!(calls.len(), 2);
    let modes: Vec<bool> = calls.iter().map(|call| call.json_mode).collect();
    assert_eq!(modes, [false, true], "the second ask is the JSON fallback");
    // The fallback re-asks the original exchange rather than showing the model its failure: it is
    // a different adapter asking the same question, not a correction.
    assert_eq!(calls[1].turns().len(), 1);
}

#[tokio::test]
async fn a_shape_mismatch_with_the_feedback_ask_carries_pydantics_complaint() {
    // With the feedback ask the model is shown what went wrong instead, and what it is shown is
    // the cast's own words — `list[GiftIdea]`'s missing member, not a serde message.
    let shallow = marker_reply(r#"[{"title":"Fly rod"}]"#);
    let lm = Scripted::new(&[&shallow, &marker_reply(GOOD_IDEAS)]);
    let outputs = IdeasTask::predict()
        .feedback_retry()
        .call_inputs_with(&lm, &inputs())
        .await
        .expect("corrected reply deserializes");
    assert_eq!(outputs.ideas.len(), 3);

    let calls = lm.calls();
    assert_eq!(calls.len(), 2);
    let retry = &calls[1].turns();
    assert_eq!(retry.len(), 3);
    assert_eq!(retry[1].role, "assistant");
    assert_eq!(retry[1].text().unwrap(), shallow);
    assert!(
        retry[2].text().unwrap().contains("ideas.0.why is required"),
        "got: {:?}",
        retry[2].parts
    );
}

#[tokio::test]
async fn a_second_shape_failure_is_final_with_no_third_ask() {
    let shallow = marker_reply(r#"[{"title":"Fly rod"}]"#);
    let lm = Scripted::new(&[&shallow, &shallow]);
    let error = IdeasTask::predict()
        .feedback_retry()
        .call_inputs_with(&lm, &inputs())
        .await
        .expect_err("second bad shape is final");
    assert!(
        error.to_string().contains("ideas.0.why is required"),
        "got: {error}"
    );
    assert_eq!(lm.calls().len(), 2);
}

#[tokio::test]
async fn typed_calls_stay_bounded_at_two_provider_calls() {
    // The ceiling is the ask plus one feedback retry. Both failures this reply could have — a
    // field left out and a member of a structured one left out — are parse failures on dspy 3.3.1,
    // so they are one stage rather than two, and the retry is shown both complaints in turn.
    let script = [
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
        .expect("second reply lands");
    assert_eq!(outputs.ideas.len(), 3);

    let calls = lm.calls();
    assert_eq!(calls.len(), 2);
    let modes: Vec<bool> = calls.iter().map(|call| call.json_mode).collect();
    assert_eq!(modes, [true, true], "the chosen adapter is used throughout");
    assert!(
        calls[1]
            .turns()
            .last()
            .expect("turns")
            .text()
            .unwrap()
            .contains("ideas.0.why is required")
    );
}

#[tokio::test]
async fn a_field_left_out_is_the_other_thing_the_feedback_ask_names() {
    // The other parse failure, so the two are told apart: a declared field the reply omits is
    // named by `ensure`, not by a cast.
    let lm = Scripted::new(&[
        &format!(r#"{{ "ideas": {GOOD_IDEAS} }}"#),
        &format!(r#"{{ "ideas": {GOOD_IDEAS}, "tip": "Wrap it well." }}"#),
    ]);
    let outputs = IdeasTask::predict()
        .feedback_retry()
        .adapter(JsonAdapter::default())
        .call_inputs_with(&lm, &inputs())
        .await
        .expect("second reply lands");
    assert_eq!(outputs.tip, "Wrap it well.");
    let calls = lm.calls();
    assert_eq!(calls.len(), 2);
    assert!(
        calls[1]
            .turns()
            .last()
            .expect("turns")
            .text()
            .unwrap()
            .contains("the tip field is missing")
    );
}

#[tokio::test]
async fn chain_of_thought_re_asks_a_shape_mismatch_through_the_json_fallback() {
    // A reasoned reply whose structured field is short a member fails the same way a plain one
    // does — inside `parse` — so the chat adapter's fallback re-asks the whole exchange in JSON,
    // and the reasoning still comes back from the reply that lands.
    let reasoned_bad = format!(
        "[[ ## reasoning ## ]]\nthinking hard\n\n{}",
        marker_reply(r#"[{"title":"Fly rod"}]"#)
    );
    let corrected = format!(
        r#"{{ "reasoning": "thinking again", "ideas": {GOOD_IDEAS}, "tip": "Wrap it well." }}"#
    );
    let lm = Scripted::new(&[&reasoned_bad, &corrected]);
    let outputs = IdeasTask::chain_of_thought()
        .call_inputs_with(&lm, &inputs())
        .await
        .expect("the fallback's reply deserializes");
    assert_eq!(outputs.ideas.len(), 3);
    assert_eq!(outputs.tip, "Wrap it well.");

    let calls = lm.calls();
    assert_eq!(calls.len(), 2);
    let modes: Vec<bool> = calls.iter().map(|call| call.json_mode).collect();
    assert_eq!(modes, [false, true], "the second ask is the JSON fallback");
}
