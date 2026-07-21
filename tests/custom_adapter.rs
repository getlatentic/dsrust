//! A user-defined adapter, written outside the crate.
//!
//! dspy's `Adapter` is a base class people subclass — `XMLAdapter`, `TwoStepAdapter` and
//! `BAMLAdapter` all ship that way. The Rust equivalent has to be a trait a downstream crate
//! can implement without touching this one, and this test is the proof.

use anyhow::{Result, anyhow};
use dsrs::Example;
use dsrs::adapter::Input;
use dsrs::adapter::python_json::format_value;
use dsrs::lm::ChatTurn;
use dsrs::signature::{OutField, Signature};
use dsrs::{Adapter, ChatAdapter};
use serde_json::{Map, Value, json};

/// Fields wrapped in XML tags rather than `[[ ## markers ## ]]`.
struct XmlAdapter;

impl Adapter for XmlAdapter {
    fn format(
        &self,
        signature: &Signature,
        _demos: &[Example],
        inputs: &[Input<'_>],
    ) -> anyhow::Result<(String, Vec<ChatTurn>)> {
        let user = inputs
            .iter()
            .map(|input| {
                let (name, value) = (input.name, &input.value);
                format!("<{name}>{}</{name}>", format_value(value))
            })
            .collect::<Vec<_>>()
            .join("\n");
        Ok((self.system_message(signature)?, vec![ChatTurn::user(user)]))
    }

    fn system_message(&self, signature: &Signature) -> anyhow::Result<String> {
        let tags: Vec<String> = signature
            .outputs
            .iter()
            .map(|field| format!("<{}>...</{}>", field.name, field.name))
            .collect();
        Ok(format!(
            "{}\nReply with {}.",
            signature.instructions,
            tags.join(" then ")
        ))
    }

    fn parse(&self, signature: &Signature, raw: &str) -> Result<Value> {
        let mut found = Map::new();
        for field in &signature.outputs {
            let open = format!("<{}>", field.name);
            let close = format!("</{}>", field.name);
            if let (Some(start), Some(end)) = (raw.find(&open), raw.find(&close)) {
                let text = raw[start + open.len()..end].trim();
                found.insert(field.name.to_owned(), Value::String(text.to_owned()));
            }
        }
        match found.is_empty() {
            true => Err(anyhow!("reply carries no xml field tags")),
            false => Ok(Value::Object(found)),
        }
    }
}

fn signature() -> Signature {
    Signature::single_input(
        "Pick a colour.",
        vec![OutField {
            name: "colour".into(),
            desc: "the chosen colour".into(),
            ..Default::default()
        }],
    )
}

#[test]
fn an_adapter_defined_outside_the_crate_formats_and_parses() {
    let inputs = [Input::new("request", json!("something calm"))];
    let (system, turns) = XmlAdapter
        .format(&signature(), &[], &inputs)
        .expect("renders");
    assert!(system.contains("<colour>...</colour>"));
    assert_eq!(turns.len(), 1);
    assert_eq!(
        turns[0].content.text().unwrap(),
        "<request>something calm</request>"
    );

    let parsed = XmlAdapter
        .parse(&signature(), "<colour>blue</colour>")
        .expect("xml reply parses");
    assert_eq!(parsed["colour"], "blue");
}

#[test]
fn a_custom_adapter_is_interchangeable_with_the_shipped_ones() {
    let adapters: Vec<Box<dyn Adapter>> =
        vec![Box::new(ChatAdapter::default()), Box::new(XmlAdapter)];
    let inputs = [Input::new("request", json!("something calm"))];
    for adapter in &adapters {
        let (system, _) = adapter.format(&signature(), &[], &inputs).expect("renders");
        assert!(
            system.contains("Pick a colour."),
            "every adapter carries the instruction"
        );
    }
}
