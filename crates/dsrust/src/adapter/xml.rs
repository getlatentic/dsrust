//! dspy `XMLAdapter`: the same conversation, with each field wrapped in a tag.
//!
//! Upstream builds this on its chat adapter, changing only how one field is written and how a
//! reply is read back. The same holds here — the field lists, the demos, the conversation
//! history and the objective all come from the shared assembly, and only the wrapper differs.
//! It exists because some models follow tags more reliably than they follow marker sections.

use anyhow::Result;
use serde_json::Value;

use crate::example::Example;
use crate::lm::ChatTurn;
use crate::lm::api::LmMessage;
use crate::lm::messages_of;
use crate::signature::Signature;

use super::Input;
use super::exchange::{Style, plain};
use super::{blocks, conversation, live_inputs, output_slot, python_json::format_field_value};

/// One field as a tag pair. dspy puts the value on its own line between the tags.
fn wrap(name: &str, value: &str) -> String {
    format!("<{name}>\n{value}\n</{name}>")
}

/// The assistant turn an example produced, in tags. Unlike the chat adapter's there is no
/// closing marker: a reply ends where its last tag closes.
fn answer(signature: &Signature, example: &Example, missing: Option<&str>) -> ChatTurn {
    let sections: Vec<String> = signature
        .outputs
        .iter()
        .filter_map(|field| {
            let value = match example.get(&field.name) {
                Some(value) => format_field_value(&field.kind, value),
                None => missing?.to_owned(),
            };
            Some(wrap(&field.name, &value))
        })
        .collect();
    ChatTurn::assistant(sections.join("\n\n"))
}

const STYLE: Style = Style {
    wrap,
    value: super::exchange::plain,
    answer,
};

/// Tag-wrapped fields rather than marker sections.
#[derive(Default)]
pub struct XmlAdapter;

impl super::Adapter for XmlAdapter {
    /// dspy's class name, not this crate's type name: it is what a callback watcher
    /// reads, where upstream hands the handler the instance and it takes
    /// `type(instance).__name__`.
    fn name(&self) -> &'static str {
        "XMLAdapter"
    }

    fn system_message(&self, signature: &Signature) -> Result<String> {
        let slots = signature
            .inputs
            .iter()
            .map(|field| wrap(&field.name, &format!("{{{}}}", field.name)))
            .chain(
                signature
                    .outputs
                    .iter()
                    // The slot carries whatever note the field earns, inside the tags.
                    .map(|field| wrap(&field.name, &output_slot(field))),
            )
            .collect::<Vec<_>>()
            .join("\n\n");
        let structure = format!(
            "All interactions will be structured in the following way, with the appropriate \
             values filled in.\n\n{slots}"
        );
        Ok(super::system_message(signature, &structure))
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

    fn parse(&self, signature: &Signature, raw: &str) -> Result<Value> {
        super::parse::parse_tags(signature, raw)
    }
}

/// The request: each input in tags, closed by the reminder naming the tags to answer in.
fn user_message(signature: &Signature, inputs: &[Input<'_>]) -> String {
    let mut parts: Vec<String> = inputs
        .iter()
        .map(|input| wrap(input.name, &plain(signature, input.name, &input.value)))
        .collect();
    parts.push(output_requirements(signature));
    parts.join("\n\n").trim().to_owned()
}

/// dspy `user_message_output_requirements` for XML: the tags, in order.
fn output_requirements(signature: &Signature) -> String {
    let tags: Vec<String> = signature
        .outputs
        .iter()
        .map(|field| format!("`<{}>`", field.name))
        .collect();
    format!(
        "Respond with the corresponding output fields wrapped in XML tags {}.",
        tags.join(", then ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::Adapter;
    use crate::signature::{FieldKind, InField, OutField};
    use serde_json::json;

    fn signature() -> Signature {
        let mut signature = Signature::single_input(
            "Pick a color.",
            vec![
                OutField {
                    name: "color".into(),
                    desc: "the color".into(),
                    ..Default::default()
                },
                OutField {
                    name: "why".into(),
                    ..Default::default()
                },
            ],
        );
        signature.inputs = vec![InField {
            name: "room".into(),
            desc: "the room".into(),
            ..Default::default()
        }];
        signature
    }

    /// The bytes `dspy.adapters.xml_adapter.XMLAdapter().format` writes for this signature.
    #[test]
    fn the_system_message_states_every_field_as_a_tag_pair() {
        let system = XmlAdapter.system_message(&signature()).expect("renders");
        assert!(system.starts_with(
            "Your input fields are:\n1. `room` (str): the room\n\
             Your output fields are:\n1. `color` (str): the color\n2. `why` (str):\n"
        ));
        assert!(system.contains("<room>\n{room}\n</room>\n\n<color>\n{color}\n</color>"));
        assert!(system.ends_with(
            "In adhering to this structure, your objective is: \n        Pick a color."
        ));
    }

    #[test]
    fn the_request_wraps_each_input_then_names_the_tags_to_answer_in() {
        let rendered = XmlAdapter
            .format(&signature(), &[], &[Input::new("room", json!("the study"))])
            .expect("renders");
        let turns = &rendered[1..];
        assert_eq!(
            turns[0].text().unwrap(),
            "<room>\nthe study\n</room>\n\n\
             Respond with the corresponding output fields wrapped in XML tags `<color>`, \
             then `<why>`."
        );
    }

    #[test]
    fn a_typed_output_keeps_its_note_inside_the_tags() {
        let mut signature = signature();
        signature.outputs[0].kind = FieldKind::Int;
        let system = XmlAdapter.system_message(&signature).expect("renders");
        assert!(
            system.contains(
                "<color>\n{color}        # note: the value you produce must be a single int \
                 value\n</color>"
            ),
            "got: {system}"
        );
    }

    #[test]
    fn a_demo_reads_as_a_solved_exchange_in_tags() {
        // No closing marker: an XML reply ends where its last tag closes.
        let demo = crate::example! { room: "the den", color: "green", why: "It rests." };
        let rendered = XmlAdapter
            .format(&signature(), &[demo], &[Input::new("room", json!("study"))])
            .expect("renders");
        let turns = &rendered[1..];
        assert_eq!(turns[0].text().unwrap(), "<room>\nthe den\n</room>");
        assert_eq!(
            turns[1].text().unwrap(),
            "<color>\ngreen\n</color>\n\n<why>\nIt rests.\n</why>"
        );
    }
}
