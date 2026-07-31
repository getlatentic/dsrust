//! A `Citations` output asks Anthropic a different question than it asks anyone else.
//!
//! `Citations.adapt_to_native_lm_feature` deletes the field from the rendered signature for an
//! `anthropic/` model and `Citations.parse_lm_response` fills it afterwards from the citations the
//! provider attached to its own text blocks. Every other provider renders the field and asks for it
//! in prose. Both arms are held here, because deleting it everywhere and deleting it nowhere each
//! pass a one-provider fixture.
//!
//! The goldens come from running the pinned dspy — `scripts/generate_citations_fixture.py`.

use std::sync::Arc;

use anyhow::Result;
use dsrust::adapter::Input;
use dsrust::adapter::native_citations;
use dsrust::lm::api::{self, LmOutput, LmPart, LmResponse};
use dsrust::lm::{ChatModel, DynChatModel, global};
use dsrust::signature::{FieldKind, JsonType, OutField, Signature, TypeDescription};
use dsrust::{Adapter, ChatAdapter, Example, Module, example};
use serde_json::Value;

fn golden() -> Value {
    serde_json::from_str(include_str!("conformance/adapter/citations_native.json"))
        .expect("the golden parses")
}

/// The signature dspy's fixture used: a question in, an answer and a `Citations` out.
fn cited() -> Signature {
    let mut signature: Signature = "question -> answer".parse().expect("parses");
    signature.instructions = "Answer the question with sources.".to_owned();
    // The reflected type, carried across from the golden. A Rust-declared signature has no Python
    // to reflect, and `Citations` is a pydantic model whose docstring and schema both reach the
    // prompt — so a byte comparison of the arm that *keeps* the field needs them.
    let reflected = &golden()["citations_type"];
    signature.outputs.push(OutField {
        name: "citations".into(),
        kind: FieldKind::Json(JsonType {
            annotation: "Citations".into(),
            descriptions: vec![TypeDescription {
                name: "Citations".into(),
                text: reflected["description"]
                    .as_str()
                    .expect("description")
                    .into(),
                replaces_schema: false,
            }],
            reflection: Some(reflected["schema"].clone()),
        }),
        // The note beside the field line reads this, not the type's reflection: dspy prints the
        // schema of the *field*, which for a custom type is the type's own.
        schema: Some(reflected["schema"].clone()),
        ..Default::default()
    });
    signature
}

/// Whichever arm the golden recorded for `model`.
fn arm(model: &str) -> Value {
    golden()["renders"]
        .as_array()
        .expect("renders")
        .iter()
        .find(|render| render["model"] == model)
        .unwrap_or_else(|| panic!("no recorded render for {model}"))
        .clone()
}

/// dspy drops the field for an Anthropic model, so the prompt never names it. This is the byte that
/// moves: rendering it anyway asks the model for something the provider will also send.
#[test]
fn an_anthropic_model_is_not_asked_for_the_field() {
    let recorded = arm("anthropic/claude-sonnet-4-5");
    assert_eq!(recorded["renders_the_field"], Value::Bool(false));

    let planned = native_citations::plan(&cited(), true).expect("a Citations output on Anthropic");
    let (system, _) = ChatAdapter::default()
        .format(
            &planned.signature,
            &[],
            &[Input::new("question", serde_json::json!("Who wrote it?"))],
        )
        .expect("renders");

    // Byte for byte against what dspy rendered, not merely "the word is absent": dropping the
    // field changes the numbering of every output line after it, and a `contains` check would pass
    // for a render that dropped the field and got the rest wrong.
    assert_eq!(
        system,
        recorded["system"].as_str().expect("system"),
        "the Anthropic render diverges from dspy's"
    );
}

/// Every other provider renders it, so the same signature on OpenAI still asks in prose.
#[test]
fn any_other_provider_is_still_asked_for_the_field() {
    let recorded = arm("openai/gpt-4o-mini");
    assert_eq!(recorded["renders_the_field"], Value::Bool(true));

    assert_eq!(native_citations::plan(&cited(), false), None);
    let (system, _) = ChatAdapter::default()
        .format(
            &cited(),
            &[],
            &[Input::new("question", serde_json::json!("Who wrote it?"))],
        )
        .expect("renders");
    assert_eq!(
        system,
        recorded["system"].as_str().expect("system"),
        "the non-Anthropic render diverges from dspy's"
    );
}

/// A model that cites: it answers with text and the citation parts Anthropic attaches to it, and
/// it reports itself as one whose citations arrive natively.
struct Citing;

impl ChatModel for Citing {
    async fn forward(&self, _request: &api::LmRequest) -> Result<LmResponse> {
        let citations = golden()["response"]["citations"].clone();
        let mut parts = vec![LmPart::text(
            "[[ ## answer ## ]]\nBede wrote it.\n\n[[ ## completed ## ]]",
        )];
        for citation in citations.as_array().expect("citations") {
            parts.push(LmPart::citation(citation));
        }
        Ok(LmResponse {
            outputs: vec![LmOutput {
                parts,
                ..LmOutput::default()
            }],
            ..LmResponse::text("")
        })
    }

    fn native_citations_usable(&self) -> bool {
        true
    }
}

/// The parse half: the field the render dropped is filled from the provider's own channel, with
/// the citations dspy's `parse_lm_response` reads out of the same reply.
///
/// Without this the field would come back null — the prompt never asked for it — so the two halves
/// only work together.
#[tokio::test]
async fn the_field_is_filled_from_the_providers_own_channel() {
    let model = Arc::new(Citing);
    global::configure_model(
        reqwest::Client::new(),
        model.clone() as Arc<dyn DynChatModel>,
    );

    let predict = dsrust::predict::Predict::from_signature(cited()).set_lm(model);
    let answered = predict
        .forward(example! { question: "Who wrote it?" }.with_inputs(["question"]))
        .await
        .expect("the reply parses");

    let filled = answered.get("citations").expect("a citations field");
    let cited: &Vec<Value> = filled.as_array().expect("citations are a list");
    assert_eq!(cited.len(), 2, "both citations should arrive: {filled}");
    assert_eq!(cited[0]["cited_text"], "Bede completed it in 731.");
    assert_eq!(cited[0]["document_title"], "Ecclesiastical History");
    assert_eq!(cited[1]["cited_text"], "written at Jarrow");

    assert_eq!(
        answered.get("answer").and_then(Value::as_str),
        Some("Bede wrote it."),
        "the rendered fields still parse from the text"
    );
}

/// The same reply on a provider that does not cite natively leaves the field to the prompt, so a
/// run that read the channel unconditionally would fill a field the model was asked for.
#[tokio::test]
async fn a_provider_that_does_not_cite_natively_keeps_the_field_in_the_prompt() {
    let _ = Example::default();
    assert_eq!(native_citations::plan(&cited(), false), None);
    assert_eq!(
        native_citations::citations_output_field(&cited()),
        Some("citations"),
        "the field is found whatever the provider; only the plan is conditional"
    );
}
