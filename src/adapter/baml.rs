//! dspy `BAMLAdapter`: the JSON adapter, with each output's type spelled out rather than
//! schema'd.
//!
//! Upstream subclasses its JSON adapter and changes exactly two things. The structure section of
//! the system message states every output type in a compact notation, which costs a fraction of
//! the tokens a JSON schema does and which smaller models follow more reliably. And an input
//! that is a whole record reaches the model as indented JSON rather than the single line
//! `json.dumps` writes, so a nested value stays readable. Everything else — the field lists,
//! the demos, the conversation history, and how a reply is read back — is the JSON adapter's,
//! which is why a parse failure here reports itself under that adapter's name.
//!
//! Named for the project whose formatter it follows: <https://github.com/BoundaryML/baml>.

mod notation;

use anyhow::Result;
use serde_json::Value;

use crate::example::Example;
use crate::lm::{ChatTurn, OutputMode};
use crate::signature::Signature;

use super::exchange::{Style, json_answer};
use super::python_json::format_field_value;
use super::{Adapter, JsonAdapter, blocks, conversation, live_inputs, marker, section};

/// Marker sections and a JSON reply, as the JSON adapter has it, laying a record out over
/// several lines wherever one appears — in a demo and a replayed conversation as much as in the
/// live request, since upstream renders all three through the one method.
const STYLE: Style = Style {
    wrap: section,
    value: input_value,
    answer: json_answer,
};

/// The JSON adapter, stating a structured output's type as a compact notation.
#[derive(Debug, Clone, Copy, Default)]
pub struct BamlAdapter;

impl BamlAdapter {
    /// dspy `format_field_structure`: how an interaction is laid out, with each output's type
    /// written where the other adapters leave a placeholder and a note.
    ///
    /// A type the notation refuses — a model that reaches itself — is an error here, as it is
    /// upstream, because the alternative is a prompt confidently describing a different type.
    pub fn field_structure(&self, signature: &Signature) -> Result<String> {
        let mut sections = vec![format!(
            "All interactions will be structured in the following way, with the appropriate \
             values filled in.\n"
        )];
        for field in &signature.inputs {
            sections.extend([
                marker(&field.name),
                format!("{{{}}}", field.name),
                String::new(),
            ]);
        }
        for field in &signature.outputs {
            sections.push(marker(&field.name));
            sections.push(format!(
                "Output field `{}` should be of type: {}\n",
                field.name,
                notation::output_type(field)?
            ));
        }
        sections.push(marker("completed"));
        Ok(sections.join("\n"))
    }
}

impl Adapter for BamlAdapter {
    /// A signature the notation refuses states why where its structure would go: this method
    /// cannot report the failure, and a prompt that quietly described some other type would be
    /// worse than one that says what went wrong. [`BamlAdapter::field_structure`] hands the same
    /// message to a caller that can act on it.
    fn system_message(&self, signature: &Signature) -> String {
        let structure = self
            .field_structure(signature)
            .unwrap_or_else(|error| error.to_string());
        super::system_message(signature, &structure)
    }

    fn format(
        &self,
        signature: &Signature,
        demos: &[Example],
        inputs: &[(&str, Value)],
    ) -> (String, Vec<ChatTurn>) {
        let (asked, mut turns) = conversation(signature, demos, inputs, STYLE);
        turns.push(ChatTurn::user(user_message(
            &asked,
            &live_inputs(&asked, inputs),
        )));
        (
            self.system_message(signature),
            blocks::split_custom_types(turns),
        )
    }

    /// A reply is one JSON object, read exactly as the adapter this one is built on reads it.
    /// Upstream inherits the method outright, errors and error name included.
    fn parse(&self, signature: &Signature, raw: &str) -> Result<Value> {
        JsonAdapter.parse(signature, raw)
    }

    fn output_mode<'a>(&self, schema: &'a Value) -> OutputMode<'a> {
        JsonAdapter.output_mode(schema)
    }
}

/// The request: each input in its own marker section, closed by the JSON adapter's reminder that
/// the reply is one object.
fn user_message(signature: &Signature, inputs: &[(&str, Value)]) -> String {
    let mut parts: Vec<String> = inputs
        .iter()
        .map(|(name, value)| section(name, &input_value(signature, name, value)))
        .collect();
    parts.push(super::json_output_requirements(signature));
    parts.join("\n\n").trim().to_owned()
}

/// How one input value is written.
///
/// A record is laid out over several lines, which is the half of this format that faces the
/// input side: a nested value crammed onto one line is what the models upstream targets read
/// worst. Everything else takes the ordinary formatter.
///
/// dspy asks whether the value is a pydantic instance. The crate has no pydantic instance to
/// ask about, so it asks the declaration instead — a field whose type is a model, carrying an
/// object — which picks out the same values except where an input does not match what it was
/// declared as.
fn input_value(signature: &Signature, name: &str, value: &Value) -> String {
    let declared_record = signature
        .inputs
        .iter()
        .find(|field| field.name == name)
        .is_some_and(|field| notation::is_record(&field.kind));
    match declared_record && value.is_object() {
        true => serde_json::to_string_pretty(value).unwrap_or_else(|_| format_field_value(value)),
        false => format_field_value(value),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signature::{FieldKind, InField, JsonType, OutField};
    use serde_json::json;

    /// `PatientDetails` as `bridge/python/rust_adapter.py` reflects it, trimmed to the members
    /// the assertions here read.
    fn patient_reflection() -> Value {
        json!({
            "type": { "kind": "model", "model": 0 },
            "models": [{
                "doc": "Patient Details model docstring",
                "fields": [
                    { "name": "name", "desc": "Full name of the patient", "alias": null,
                      "type": { "kind": "str" } },
                    { "name": "age", "desc": null, "alias": null, "type": { "kind": "int" } },
                ],
            }],
        })
    }

    fn patient_kind() -> FieldKind {
        FieldKind::Json(JsonType {
            annotation: "PatientDetails".into(),
            descriptions: Vec::new(),
            reflection: Some(patient_reflection()),
        })
    }

    /// Upstream's signature: a record in, a record out, and a plain question beside it.
    fn signature() -> Signature {
        Signature {
            instructions: "Extract the patient.".into(),
            inputs: vec![
                InField {
                    name: "patient".into(),
                    desc: String::new(),
                    kind: patient_kind(),
                    values: None,
                },
                InField {
                    name: "question".into(),
                    desc: String::new(),
                    kind: FieldKind::Str,
                    values: None,
                },
            ],
            outputs: vec![OutField {
                name: "answer".into(),
                desc: String::new(),
                kind: FieldKind::Str,
                values: None,
                schema: None,
            }],
        }
    }

    fn patient() -> Value {
        json!({ "name": "John Doe", "age": 45 })
    }

    /// The bytes `BAMLAdapter().format_field_structure` writes: every input a placeholder, every
    /// output its type, and the closing marker.
    #[test]
    fn the_structure_states_a_placeholder_per_input_and_a_type_per_output() {
        let mut signature = signature();
        signature.outputs[0].kind = patient_kind();
        assert_eq!(
            BamlAdapter.field_structure(&signature).expect("renders"),
            "All interactions will be structured in the following way, with the appropriate \
             values filled in.\n\n\
             [[ ## patient ## ]]\n\
             {patient}\n\n\
             [[ ## question ## ]]\n\
             {question}\n\n\
             [[ ## answer ## ]]\n\
             Output field `answer` should be of type: # Patient Details model docstring\n\
             {\n\
             \x20 # Full name of the patient\n\
             \x20 name: string,\n\
             \x20 age: int,\n\
             }\n\n\
             [[ ## completed ## ]]"
        );
    }

    /// The structure section reaches the system message inside the frame every adapter shares,
    /// which is where the numbered field lists and the objective come from.
    #[test]
    fn the_system_message_carries_the_structure_between_the_field_lists_and_the_objective() {
        let system = BamlAdapter.system_message(&signature());
        // The trailing space on every line but a block's last is upstream's: it strips the
        // block, not the lines.
        assert!(
            system.starts_with(
                "Your input fields are:\n\
                 1. `patient` (PatientDetails): \n\
                 2. `question` (str):\n\
                 Your output fields are:\n\
                 1. `answer` (str):\n\
                 All interactions will be structured"
            ),
            "got: {system}"
        );
        assert!(system.contains("Output field `answer` should be of type: string"));
        assert!(system.ends_with(
            "In adhering to this structure, your objective is: \n        Extract the patient."
        ));
    }

    /// A type the notation refuses cannot be reported by an infallible method, so the message
    /// stands where the structure would — the same message `field_structure` returns.
    #[test]
    fn a_refused_type_states_its_refusal_in_place_of_the_structure() {
        let mut signature = signature();
        signature.outputs[0].kind = FieldKind::Json(JsonType {
            annotation: "CircularModel".into(),
            descriptions: Vec::new(),
            reflection: Some(json!({
                "type": { "kind": "model", "model": 0 },
                "models": [{ "doc": null, "fields": [
                    { "name": "field", "desc": null, "alias": null,
                      "type": { "kind": "model", "model": 0 } },
                ]}],
            })),
        });
        assert!(BamlAdapter.field_structure(&signature).is_err());
        assert!(
            BamlAdapter
                .system_message(&signature)
                .contains("BAMLAdapter cannot handle recursive pydantic models")
        );
    }

    #[test]
    fn a_record_input_is_laid_out_over_lines_and_everything_else_is_not() {
        let inputs = vec![
            ("patient", patient()),
            ("question", json!("What is the diagnosis?")),
        ];
        let (_, turns) = BamlAdapter.format(&signature(), &[], &inputs);
        assert_eq!(
            turns[0].content.text().expect("one text turn"),
            "[[ ## patient ## ]]\n\
             {\n  \"name\": \"John Doe\",\n  \"age\": 45\n}\n\n\
             [[ ## question ## ]]\n\
             What is the diagnosis?\n\n\
             Respond with a JSON object in the following order of fields: `answer`."
        );
    }

    /// The record test alone would pass if every object were laid out, so a field declared as
    /// something other than a model has to stay on one line beside it.
    #[test]
    fn an_object_under_a_non_record_declaration_stays_on_one_line() {
        let mut signature = signature();
        signature.inputs[0].kind = FieldKind::Json(JsonType::plain("dict[str, str]"));
        let inputs = vec![("patient", patient())];
        let (_, turns) = BamlAdapter.format(&signature, &[], &inputs);
        assert!(
            turns[0]
                .content
                .text()
                .expect("one text turn")
                .starts_with("[[ ## patient ## ]]\n{\"name\": \"John Doe\", \"age\": 45}"),
            "got: {:?}",
            turns[0].content.text()
        );
    }

    /// A record's value is where a multimodal type's blocks are embedded, and laying the record
    /// out must not stop the message being split around them.
    #[test]
    fn a_custom_type_inside_a_record_still_splits_into_its_own_block() {
        let embedded = json!({
            "images": ["<<CUSTOM-TYPE-START-IDENTIFIER>>[{\"type\": \"image_url\", \
                       \"image_url\": {\"url\": \"https://example.com/a.jpg\"}}]\
                       <<CUSTOM-TYPE-END-IDENTIFIER>>"],
        });
        let (_, turns) = BamlAdapter.format(&signature(), &[], &[("patient", embedded)]);
        let crate::lm::Content::Blocks(blocks) = &turns[0].content else {
            panic!("got: {:?}", turns[0].content)
        };
        assert!(blocks.contains(&json!({
            "type": "image_url",
            "image_url": { "url": "https://example.com/a.jpg" },
        })));
    }

    /// dspy renders a demo through the same method as the live request, so a record in one is
    /// laid out too, and answers it the JSON adapter's way — an object rather than the marker
    /// sections the chat adapter reads back.
    #[test]
    fn a_demo_is_laid_out_the_way_the_request_it_stands_for_would_be() {
        let demo = crate::example! {
            patient: json!({ "name": "Jane Doe", "age": 30 }),
            question: "Who?",
            answer: "Jane Doe",
        };
        let (_, turns) = BamlAdapter.format(&signature(), &[demo], &[("patient", patient())]);
        assert_eq!(
            turns[0].content.text().expect("a rendered demo"),
            "[[ ## patient ## ]]\n\
             {\n  \"name\": \"Jane Doe\",\n  \"age\": 30\n}\n\n\
             [[ ## question ## ]]\nWho?"
        );
        assert_eq!(
            turns[1].content.text().expect("a rendered demo"),
            "{\n  \"answer\": \"Jane Doe\"\n}"
        );
    }

    /// Upstream inherits its parse outright, so a reply naming the wrong fields fails the same
    /// way and a well-formed one reads the same values.
    #[test]
    fn a_reply_is_read_exactly_as_the_json_adapter_reads_it() {
        let signature = signature();
        let raw = r#"{"answer": "a fever"}"#;
        assert_eq!(
            BamlAdapter.parse(&signature, raw).expect("parses"),
            JsonAdapter.parse(&signature, raw).expect("parses")
        );
        assert!(BamlAdapter.parse(&signature, r#"{"other": 1}"#).is_err());
    }

    /// The provider is asked for structured output, as it is for the adapter this is built on:
    /// only how the type is described in the prompt differs.
    #[test]
    fn the_provider_is_still_asked_for_structured_output() {
        let schema = json!({ "type": "object" });
        assert!(matches!(
            BamlAdapter.output_mode(&schema),
            OutputMode::Json { .. }
        ));
        assert!(BamlAdapter.json_fallback().is_none());
    }
}
