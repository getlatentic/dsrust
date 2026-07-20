//! dspy `TwoStepAdapter`: let the main model answer in prose, then have a second model pull the
//! fields out of what it said.
//!
//! Every other adapter asks one model to answer *and* to obey a wire format at the same time. A
//! model strong at the task but weak at formatting loses marks for the second job. This adapter
//! splits them: the first ask carries no markers, no JSON and no tags — just a description of
//! what the answer must cover — and a second, smaller model reads that prose through the chat
//! adapter to produce the fields.
//!
//! The second ask is a model call. This crate puts model calls in the module rather than the
//! adapter, so [`Adapter::extraction`] hands the module what to ask and leaves the asking to it.

use std::sync::Arc;

use anyhow::Result;
use serde_json::Value;

use crate::adapter::python_json::format_field_value;
use crate::adapter::{Adapter, ChatAdapter, Extraction};
use crate::example::Example;
use crate::lm::{ChatTurn, DynChatModel};
use crate::signature::{FieldKind, InField, Signature};

use super::prompt::{numbered_input_lines, numbered_output_lines};

/// The field the extraction is handed, holding whatever the first model wrote.
const TEXT: &str = "text";

/// An adapter that asks for prose first and structure second.
///
/// ```
/// # use dsrs::{DummyLM, TwoStepAdapter, example};
/// # use std::sync::Arc;
/// let smaller = Arc::new(DummyLM::new([example! { answer: "Paris" }]));
/// let adapter = TwoStepAdapter::new(smaller);
/// ```
pub struct TwoStepAdapter {
    /// The model asked to read the first reply and name the fields in it.
    pub extraction_model: Arc<dyn DynChatModel>,
    /// The wire format the extraction speaks. dspy hard-codes its `ChatAdapter`; keeping it a
    /// field lets a caller extract through any format without a second adapter type.
    pub extraction_adapter: Box<dyn Adapter>,
}

impl TwoStepAdapter {
    pub fn new(extraction_model: Arc<dyn DynChatModel>) -> Self {
        Self {
            extraction_model,
            extraction_adapter: Box::new(ChatAdapter::default()),
        }
    }
}

/// dspy `format_task_description`: what the first model is told, in prose.
///
/// No wire format appears here at all — that is the point of the adapter. The model is told what
/// it will receive, what its answer must cover, and to say it fully enough that another agent
/// can read it, which is exactly what the second model then does.
fn task_description(signature: &Signature) -> String {
    let mut parts = vec![
        "You are a helpful assistant that can solve tasks based on user input.".to_owned(),
        format!(
            "As input, you will be provided with:\n{}",
            numbered_input_lines(signature)
        ),
        format!(
            "Your outputs must contain:\n{}",
            numbered_output_lines(signature)
        ),
        "You should lay out your outputs in detail so that your answer can be understood by \
         another agent"
            .to_owned(),
    ];
    if !signature.instructions.is_empty() {
        parts.push(format!("Specific instructions: {}", signature.instructions));
    }
    parts.join("\n")
}

/// dspy `format_user_message_content`: one `name: value` line per input, a blank line apart.
fn user_message(signature: &Signature, inputs: &[(&str, Value)]) -> String {
    signature
        .inputs
        .iter()
        .filter_map(|field| {
            let (_, value) = inputs.iter().find(|(name, _)| *name == field.name)?;
            Some(format!("{}: {}", field.name, format_field_value(value)))
        })
        .collect::<Vec<_>>()
        .join("\n\n")
        .trim()
        .to_owned()
}

/// The assistant half of a demo, in the same `name: value` prose the request uses.
fn demo_answer(signature: &Signature, example: &Example, missing: Option<&str>) -> ChatTurn {
    let parts: Vec<String> = signature
        .outputs
        .iter()
        .filter_map(|field| {
            let value = match example.get(&field.name) {
                Some(value) => format_field_value(value),
                None => missing?.to_owned(),
            };
            Some(format!("{}: {value}", field.name))
        })
        .collect();
    ChatTurn::assistant(parts.join("\n\n").trim().to_owned())
}

/// The user half of a demo. dspy reuses `format_user_message_content`, so a demo's request reads
/// exactly like the live one.
fn demo_ask(signature: &Signature, example: &Example, prefix: Option<&str>) -> ChatTurn {
    let values: Vec<(&str, Value)> = signature
        .inputs
        .iter()
        .filter_map(|field| Some((field.name.as_str(), example.get(&field.name)?.clone())))
        .collect();
    let body = user_message(signature, &values);
    let parts: Vec<String> = prefix
        .map(str::to_owned)
        .into_iter()
        .chain(std::iter::once(body))
        .collect();
    ChatTurn::user(parts.join("\n\n").trim().to_owned())
}

/// dspy `_create_extractor_signature`: `text` in, the original signature's outputs out.
///
/// The instruction's long run of spaces is upstream's own — a backslash-continued Python string
/// keeps the next line's indentation — and it reaches the model, so it is reproduced rather than
/// tidied away.
pub fn extractor_signature(signature: &Signature) -> Signature {
    let named: Vec<String> = signature
        .outputs
        .iter()
        .map(|field| format!("`{}`", field.name))
        .collect();
    let instructions = format!(
        "The input is a text that should contain all the necessary information to produce the \
         fields {}.             Your job is to extract the fields from the text verbatim. \
         Extract precisely the appropriate value (content) for each field.",
        named.join(", ")
    );
    Signature {
        instructions,
        inputs: vec![InField {
            name: TEXT.to_owned(),
            desc: String::new(),
            kind: FieldKind::Str,
            values: None,
        }],
        outputs: signature.outputs.clone(),
    }
}

impl Adapter for TwoStepAdapter {
    fn format(
        &self,
        signature: &Signature,
        demos: &[Example],
        inputs: &[(&str, Value)],
    ) -> (String, Vec<ChatTurn>) {
        let mut turns: Vec<ChatTurn> = demos
            .iter()
            .flat_map(|demo| {
                [
                    demo_ask(signature, demo, None),
                    demo_answer(signature, demo, None),
                ]
            })
            .collect();
        turns.push(ChatTurn::user(user_message(signature, inputs)));
        (task_description(signature), turns)
    }

    fn system_message(&self, signature: &Signature) -> String {
        task_description(signature)
    }

    /// The first reply is prose, so there is nothing to read out of it here. dspy returns the
    /// extraction's result from `parse`; this crate cannot, because the extraction is a model
    /// call — [`Adapter::extraction`] hands it to the module instead, and the module puts the
    /// text it produced through the extraction adapter's own `parse`.
    fn parse(&self, _signature: &Signature, raw: &str) -> Result<Value> {
        Ok(Value::String(raw.to_owned()))
    }

    fn extraction(&self, signature: &Signature) -> Option<Extraction<'_>> {
        Some(Extraction {
            signature: extractor_signature(signature),
            adapter: self.extraction_adapter.as_ref(),
            model: self.extraction_model.as_ref(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::example;
    use crate::lm::dummy::DummyLM;
    use crate::signature::OutField;
    use serde_json::json;

    fn signature() -> Signature {
        let mut signature = Signature::single_input(
            "Solve it.",
            vec![OutField {
                name: "answer".into(),
                desc: "the reply".into(),
                kind: FieldKind::Str,
                values: None,
                schema: None,
            }],
        );
        signature.inputs = vec![InField {
            name: "question".into(),
            desc: "the ask".into(),
            kind: FieldKind::Str,
            values: None,
        }];
        signature
    }

    fn adapter() -> TwoStepAdapter {
        TwoStepAdapter::new(Arc::new(DummyLM::new([example! { answer: "Paris" }])))
    }

    #[test]
    fn the_first_ask_describes_the_task_without_naming_a_wire_format() {
        // Copied from `TwoStepAdapter.format_task_description` on dspy 3.2.1. A marker or a
        // brace here would put the formatting burden back on the model this adapter exists to
        // relieve of it.
        assert_eq!(
            task_description(&signature()),
            "You are a helpful assistant that can solve tasks based on user input.\n\
             As input, you will be provided with:\n1. `question` (str): the ask\n\
             Your outputs must contain:\n1. `answer` (str): the reply\n\
             You should lay out your outputs in detail so that your answer can be understood by \
             another agent\n\
             Specific instructions: Solve it."
        );
    }

    #[test]
    fn a_signature_with_no_instructions_omits_that_line() {
        let mut plain = signature();
        plain.instructions = String::new();
        assert!(!task_description(&plain).contains("Specific instructions"));
    }

    #[test]
    fn the_request_is_named_values_rather_than_sections() {
        let inputs = vec![("question", json!("Why?"))];
        assert_eq!(user_message(&signature(), &inputs), "question: Why?");
    }

    #[test]
    fn the_extractor_asks_for_the_original_outputs_from_one_text_field() {
        let extractor = extractor_signature(&signature());
        assert_eq!(extractor.inputs.len(), 1);
        assert_eq!(extractor.inputs[0].name, TEXT);
        let names: Vec<&str> = extractor.outputs.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(names, ["answer"]);
    }

    #[test]
    fn the_extractor_instruction_keeps_upstreams_spacing() {
        // The run of spaces comes from a backslash-continued Python string upstream, and the
        // model reads it, so it is part of the bytes rather than something to tidy.
        assert_eq!(
            extractor_signature(&signature()).instructions,
            "The input is a text that should contain all the necessary information to produce \
             the fields `answer`.             Your job is to extract the fields from the text \
             verbatim. Extract precisely the appropriate value (content) for each field."
        );
    }

    #[test]
    fn the_adapter_offers_an_extraction_where_the_others_offer_none() {
        let adapter = adapter();
        assert!(adapter.extraction(&signature()).is_some());
        assert!(ChatAdapter::default().extraction(&signature()).is_none());
    }

    #[test]
    fn a_demo_reads_in_the_same_prose_the_request_uses() {
        let demo = example! { question: "Where?", answer: "Paris" };
        let (_, turns) = adapter().format(&signature(), &[demo], &[("question", json!("Why?"))]);
        assert_eq!(turns[0].content.text(), Some("question: Where?"));
        assert_eq!(turns[1].content.text(), Some("answer: Paris"));
        assert_eq!(turns[2].content.text(), Some("question: Why?"));
    }
}
