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
use crate::lm::api::LmMessage;
use crate::lm::messages_of;
use crate::lm::{ChatTurn, OutputMode};
use crate::signature::Signature;

use super::exchange::{Style, json_answer};
use super::python_json::format_value;
use super::{Adapter, Input, JsonAdapter, blocks, conversation, live_inputs, marker, section};

/// Marker sections and a JSON reply, as the JSON adapter has it, laying a record out over
/// several lines wherever one appears — in a demo and a replayed conversation as much as in the
/// live request, since upstream renders all three through the one method.
const STYLE: Style = Style {
    wrap: section,
    value: demo_value,
    answer: json_answer,
};

/// The JSON adapter, stating a structured output's type as a compact notation.
///
/// Carries [`JsonAdapter`]'s settings, because upstream's `BAMLAdapter(JSONAdapter)` defines no
/// `__init__` and inherits its base's — including the default that sets native function calling
/// **on**, which is `JSONAdapter`'s and not the base `Adapter`'s. No `use_json_adapter_fallback`:
/// `ChatAdapter.__call__` skips the fallback for anything that `isinstance` says is a
/// `JSONAdapter`, and this is one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BamlAdapter {
    /// dspy `use_native_function_calling`, **on** by default — inherited from `JSONAdapter`, which
    /// says so in its own constructor.
    pub use_native_function_calling: bool,
    /// dspy `parallel_tool_calls`: `None` leaves the provider option unset, which is not the same
    /// as `Some(false)`.
    pub parallel_tool_calls: Option<bool>,
}

impl Default for BamlAdapter {
    fn default() -> Self {
        Self {
            use_native_function_calling: true,
            parallel_tool_calls: None,
        }
    }
}

impl BamlAdapter {
    /// The JSON adapter this one is built on, carrying these same settings — what upstream reaches
    /// by inheriting it.
    fn base(&self) -> JsonAdapter {
        JsonAdapter {
            use_native_function_calling: self.use_native_function_calling,
            parallel_tool_calls: self.parallel_tool_calls,
        }
    }
}

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
    /// dspy's class name, not this crate's type name: it is what a callback watcher
    /// reads, where upstream hands the handler the instance and it takes
    /// `type(instance).__name__`.
    fn name(&self) -> &'static str {
        "BAMLAdapter"
    }

    fn system_message(&self, signature: &Signature) -> Result<String> {
        Ok(super::system_message(
            signature,
            &self.field_structure(signature)?,
        ))
    }

    fn format(
        &self,
        signature: &Signature,
        demos: &[Example],
        inputs: &[Input<'_>],
    ) -> Result<Vec<LmMessage>> {
        let (asked, mut turns) = conversation(signature, demos, inputs, STYLE, false);
        turns.push(ChatTurn::user(user_message(
            &asked,
            &live_inputs(&asked, inputs),
        )));
        Ok(messages_of(
            &self.system_message(signature)?,
            &blocks::split_custom_types(turns),
        ))
    }

    /// A reply is one JSON object, read exactly as the adapter this one is built on reads it.
    /// Upstream inherits the method outright, errors and error name included.
    fn parse(&self, signature: &Signature, raw: &str) -> Result<Value> {
        self.base().parse(signature, raw)
    }

    fn output_mode<'a>(&self, schema: &'a Value) -> OutputMode<'a> {
        self.base().output_mode(schema)
    }

    fn native_function_calling(&self) -> super::NativeFunctionCalling {
        self.base().native_function_calling()
    }
}

/// The request: each input in its own marker section, closed by the JSON adapter's reminder that
/// the reply is one object.
fn user_message(signature: &Signature, inputs: &[Input<'_>]) -> String {
    let mut parts: Vec<String> = inputs
        .iter()
        .map(|input| section(input.name, &input_value(input)))
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
/// dspy asks whether the value *is* a pydantic instance — `isinstance(value, BaseModel)` — and
/// this asks the same question of the same thing. It cannot be asked of the JSON, where a
/// serialized struct and a hand-written map are one and the same; it is answered where the
/// answer still exists and carried here on [`Input::record`]. Asking the field's declared type
/// instead would lay out a loose mapping handed to a record-declared field over several lines,
/// where upstream writes it inline.

/// A demo's or a replayed turn's value.
///
/// These come from an [`Example`], which holds JSON a caller put there rather than a record
/// instance, so they take the ordinary formatter — the same answer upstream reaches for a demo
/// built from a plain mapping. The provenance that earns the multi-line layout rides only a live
/// input, which a demo value is not.
fn demo_value(_signature: &Signature, _name: &str, value: &Value) -> String {
    format_value(value)
}

fn input_value(input: &Input<'_>) -> String {
    match input.is_record_object() {
        true => serde_json::to_string_pretty(&input.value)
            .unwrap_or_else(|_| format_value(&input.value)),
        false => format_value(&input.value),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signature::{FieldKind, InField, JsonType, OutField};
    use serde_json::json;

    /// `PatientDetails` as `crates/dsrs-bridge/python/rust_adapter.py` reflects it, trimmed to the members
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
                    kind: patient_kind(),
                    ..Default::default()
                },
                InField {
                    name: "question".into(),
                    ..Default::default()
                },
            ],
            outputs: vec![OutField {
                name: "answer".into(),
                ..Default::default()
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
            BamlAdapter::default()
                .field_structure(&signature)
                .expect("renders"),
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
        let system = BamlAdapter::default()
            .system_message(&signature())
            .expect("renders");
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
        // The refusal reaches every caller, not just the one that asks for the structure
        // alone: a prompt built around a type the notation could not write would describe
        // something the model is then asked to produce.
        assert!(BamlAdapter::default().field_structure(&signature).is_err());
        assert!(BamlAdapter::default().system_message(&signature).is_err());
        let asked =
            BamlAdapter::default().format(&signature, &[], &[Input::new("question", json!("x"))]);
        assert!(
            format!(
                "{:#}",
                asked.expect_err("a recursive model has no notation")
            )
            .contains("BAMLAdapter cannot handle recursive pydantic models")
        );
    }

    /// The byte rule this adapter exists for, and the one `Input::record` decides.
    ///
    /// Both forms below were taken from dspy 3.2.1 rather than reasoned about: rendering the same
    /// field through `BAMLAdapter` with a `BaseModel` gives `{\n  "name": ...\n}` and with an
    /// equivalent `dict` gives `{"name": ..., "age": ...}` on one line. Upstream's own test for
    /// this (`test_baml_adapter_formats_pydantic_inputs_as_clean_json`) asserts only that
    /// `\'"name": "John Doe"\'` appears somewhere in the message, which is true of both forms — so
    /// a green upstream suite does not pin this and these literals are what does.
    #[test]
    fn a_record_input_is_laid_out_over_lines_and_everything_else_is_not() {
        let inputs = vec![
            Input::record("patient", patient()),
            Input::new("question", json!("What is the diagnosis?")),
        ];
        let rendered = BamlAdapter::default()
            .format(&signature(), &[], &inputs)
            .expect("renders");
        let turns = &rendered[1..];
        assert_eq!(
            turns[0].text().expect("one text turn"),
            "[[ ## patient ## ]]\n\
             {\n  \"name\": \"John Doe\",\n  \"age\": 45\n}\n\n\
             [[ ## question ## ]]\n\
             What is the diagnosis?\n\n\
             Respond with a JSON object in the following order of fields: `answer`."
        );
    }

    /// The record test alone would pass if every object were laid out, so an object that did
    /// *not* arrive as a record has to stay on one line beside it. This is upstream's `dict`,
    /// which `isinstance(value, BaseModel)` answers no for however model-shaped it looks.
    #[test]
    fn an_object_that_is_not_a_record_stays_on_one_line() {
        let mut signature = signature();
        signature.inputs[0].kind = FieldKind::Json(JsonType::plain("dict[str, str]"));
        let inputs = vec![Input::new("patient", patient())];
        let rendered = BamlAdapter::default()
            .format(&signature, &[], &inputs)
            .expect("renders");
        let turns = &rendered[1..];
        assert!(
            turns[0]
                .text()
                .expect("one text turn")
                .starts_with("[[ ## patient ## ]]\n{\"name\": \"John Doe\", \"age\": 45}"),
            "got: {:?}",
            turns[0].text().as_deref()
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
        let rendered = BamlAdapter::default()
            .format(&signature(), &[], &[Input::new("patient", embedded)])
            .expect("renders");
        let wire = crate::lm::api::LmRequest::new("", rendered).wire_messages();
        let content = &wire[1]["content"];
        let blocks = content
            .as_array()
            .unwrap_or_else(|| panic!("a record input renders blocks, got: {content}"));
        assert!(blocks.contains(&json!({
            "type": "image_url",
            "image_url": { "url": "https://example.com/a.jpg" },
        })));
    }

    /// A demo is rendered by the same rule as a live request — what the *value* is — and an
    /// [`Example`] holds JSON rather than a record instance, so a demo takes the one-line form
    /// even where the live request beside it does not.
    ///
    /// That asymmetry is upstream's own. Measured against dspy 3.2.1: a demo built from a dict
    /// renders `{"name": "Jane", "age": 30}` while the same call passing a `BaseModel` renders
    /// it over four lines. A trainset of mappings and a typed call is exactly that pairing.
    #[test]
    fn a_demo_holds_json_so_it_takes_the_one_line_form() {
        let demo = crate::example! {
            patient: json!({ "name": "Jane Doe", "age": 30 }),
            question: "Who?",
            answer: "Jane Doe",
        };
        let rendered = BamlAdapter::default()
            .format(
                &signature(),
                &[demo],
                &[Input::record("patient", patient())],
            )
            .expect("renders");
        let turns = &rendered[1..];
        assert_eq!(
            turns[0].text().expect("a rendered demo"),
            "[[ ## patient ## ]]\n\
             {\"name\": \"Jane Doe\", \"age\": 30}\n\n\
             [[ ## question ## ]]\nWho?"
        );
        assert_eq!(
            turns[1].text().expect("a rendered demo"),
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
            BamlAdapter::default()
                .parse(&signature, raw)
                .expect("parses"),
            JsonAdapter::default()
                .parse(&signature, raw)
                .expect("parses")
        );
        assert!(
            BamlAdapter::default()
                .parse(&signature, r#"{"other": 1}"#)
                .is_err()
        );
    }

    /// The provider is asked for structured output, as it is for the adapter this is built on:
    /// only how the type is described in the prompt differs.
    #[test]
    fn the_provider_is_still_asked_for_structured_output() {
        let schema = json!({ "type": "object" });
        assert!(matches!(
            BamlAdapter::default().output_mode(&schema),
            OutputMode::Json { .. }
        ));
        assert!(BamlAdapter::default().json_fallback().is_none());
    }
}
