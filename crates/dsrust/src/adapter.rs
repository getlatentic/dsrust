use anyhow::Result;
use serde_json::Value;

use crate::example::Example;
use crate::lm::{ChatTurn, DynChatModel, OutputMode};
use crate::signature::Signature;

/// One input on its way to an adapter: what it is called, what it holds, and where it came from.
///
/// The provenance is here rather than read off the signature because dspy reads it off the
/// *value*: `isinstance(value, BaseModel)`. A signature only knows what a field was **declared**
/// as, and the two part company exactly when a caller hands a loose mapping to a field declared
/// as a record — at which point [`baml::BamlAdapter`] would lay out bytes upstream never sends.
///
/// A serialized struct and a hand-written map are the same `Value`, so this cannot be recovered
/// later. It is set where it is still known: the derive marks a struct-typed field, the bridge
/// asks Python, and anything built from loose JSON is not a record because it is not one.
#[derive(Debug, Clone, PartialEq)]
pub struct Input<'a> {
    pub name: &'a str,
    pub value: Value,
    /// Whether this value arrived as a record instance — a Rust struct, or a pydantic model
    /// across the bridge — rather than as JSON a caller wrote by hand.
    pub record: bool,
}

impl<'a> Input<'a> {
    /// Loose JSON: whatever a caller put in an [`Example`], and dspy's plain `dict`.
    pub fn new(name: &'a str, value: Value) -> Self {
        Self {
            name,
            value,
            record: false,
        }
    }

    /// A value that came from a declared record, which is dspy's `BaseModel` instance.
    pub fn record(name: &'a str, value: Value) -> Self {
        Self {
            name,
            value,
            record: true,
        }
    }

    /// Whether this is a record whose value is actually an object, which is the pair of
    /// conditions BAML lays out over several lines rather than one.
    pub fn is_record_object(&self) -> bool {
        self.record && self.value.is_object()
    }
}

impl<'a> From<(&'a str, Value)> for Input<'a> {
    fn from((name, value): (&'a str, Value)) -> Self {
        Self::new(name, value)
    }
}

pub mod baml;
mod blocks;
mod chat;
pub use chat::ChatAdapter;
mod json;
pub mod native_citations;
pub mod native_reasoning;
pub use native_reasoning::{NativeReasoning, ReasoningEffort};
pub mod native_tools;
pub use json::JsonAdapter;
mod two_step;
pub use two_step::{TwoStepAdapter, extractor_signature};
mod demos;
mod exchange;
pub mod types;
pub use types::{
    Audio, Citation, Citations, Code, Document, File, Formatted, History, Image, MediaType,
    Reasoning, ToolCall, ToolCallResult, ToolCallResults, ToolCalls, Type,
};
pub mod stream;
pub use stream::{FieldListener, stream_field};
pub(crate) mod history;
pub mod parse;
pub(crate) mod prompt;
pub use prompt::field_description;
pub mod python_json;
pub mod xml;

use demos::demo_turns;
use json::json_output_requirements;
use prompt::{marker, output_hint, output_slot, section, system_message};

/// How a signature travels over the wire.
///
/// Mirrors DSPy's `Adapter` base class: implement it to teach the crate a new wire format.
/// [`ChatAdapter`] speaks `[[ ## field ## ]]` marker sections any model can produce with no
/// provider support, and is DSPy's default and ours; [`xml::XmlAdapter`] wraps the same fields
/// in tags, which some models follow more reliably. [`JsonAdapter`] engages the provider's
/// native structured output, and [`baml::BamlAdapter`] builds on it, trading the JSON schema it
/// states for a compact notation of the same type.
///
/// Like DSPy, a parse failure is final: there is no silent retry in another format, because a
/// caller who chose an adapter chose the wire contract it implies.
///
/// Formatting and parsing live here; the model call lives in the module that owns the
/// conversation. That split keeps this trait object-safe, so a caller can hold
/// `Box<dyn Adapter>` and swap wire formats at run time.
pub trait Adapter: Send + Sync {
    /// Which wire format this is — dspy reads the same thing off `instance.__class__.__name__`.
    ///
    /// Reaches a caller in two places: the `adapter_name` on a
    /// [`FieldMismatch`](parse::FieldMismatch), which is dspy's own field, and the
    /// [observation spans](crate::observe). Defaulted rather than required so that adding it does
    /// not break an adapter someone already wrote; the built-ins each name themselves.
    fn name(&self) -> &'static str {
        "Adapter"
    }

    /// The whole conversation to send, with no model call: the system message, then the
    /// turns. Mirrors `Adapter.format`, which returns a message list for the same reason —
    /// a demo or a conversation history expands into several turns, not one.
    ///
    /// `demos` are the solved examples that precede the real request. An optimizer's whole
    /// output is a set of these, so an adapter that cannot render them cannot run a compiled
    /// program.
    fn format(
        &self,
        signature: &Signature,
        demos: &[Example],
        inputs: &[Input<'_>],
    ) -> Result<(String, Vec<ChatTurn>)>;

    /// Extract the signature's fields from a raw reply. A reply that does not speak this
    /// adapter's format at all fails here; a reply missing individual fields parses and
    /// leaves those gaps for the signature's own validation, whose failure carries feedback
    /// into a retry.
    fn parse(&self, signature: &Signature, raw: &str) -> Result<Value>;

    /// A signature this adapter cannot render is an error rather than a prompt describing
    /// something else — dspy's own `format` raises for the same reason, and the only adapter
    /// with a type it can refuse is the one whose notation cannot express every type.
    ///
    /// What this adapter tells the model before the conversation starts: the fields, the shape
    /// of an interaction, and the objective. dspy exposes the same method, and its `format`
    /// builds the exchange around it.
    fn system_message(&self, signature: &Signature) -> Result<String>;

    /// How the provider should be asked to shape its reply. Text by default, since a format
    /// carried entirely in the prompt needs nothing from the provider.
    fn output_mode<'a>(&self, _schema: &'a Value) -> OutputMode<'a> {
        OutputMode::Text
    }

    /// The adapter to re-ask through when a reply fails to parse, if any.
    ///
    /// dspy's `ChatAdapter.__call__` catches a parse failure and retries the whole exchange
    /// through `JSONAdapter`, which its `use_json_adapter_fallback` flag disables. Most
    /// adapters have no second opinion to offer, so the default is none.
    fn json_fallback(&self) -> Option<Box<dyn Adapter>> {
        None
    }

    /// Whether this adapter asks the provider to call tools itself, and what it says about
    /// parallel calls while doing so.
    ///
    /// dspy keeps both on its base `Adapter` and reads them in `_call_preprocess`; here they are
    /// one answer because they are only ever consulted together. Off by default, which is the
    /// base class's own default — `JSONAdapter` is the one that overrides it.
    fn native_function_calling(&self) -> NativeFunctionCalling {
        NativeFunctionCalling::default()
    }

    /// A second exchange this adapter needs before its reply carries the signature's fields.
    ///
    /// dspy's `TwoStepAdapter` lets the main model answer in prose, then asks a second model to
    /// pull the fields out of that prose. The second ask is a model call, which this trait
    /// cannot make and stay object-safe — so, exactly as [`Adapter::json_fallback`] hands back
    /// an adapter for the module to re-ask through, this hands back everything the module needs
    /// to run the extraction itself. Most adapters read their own replies and answer none.
    fn extraction(&self, _signature: &Signature) -> Option<Extraction<'_>> {
        None
    }
}

/// What an adapter says about letting the provider call tools itself.
///
/// dspy spells these as two attributes on the base `Adapter`. `parallel` is `None` where upstream
/// leaves the provider option unset, which is not the same as asking for `false`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NativeFunctionCalling {
    pub enabled: bool,
    pub parallel: Option<bool>,
}

/// The second ask an adapter cannot make for itself: what to ask, how to render it, and which
/// model to ask.
pub struct Extraction<'a> {
    /// dspy's `_create_extractor_signature`: `text` in, the original outputs out.
    pub signature: Signature,
    /// The wire format the extraction speaks, which is dspy's `ChatAdapter`.
    pub adapter: &'a dyn Adapter,
    /// The model asked to do the extracting — a smaller one than answered the task.
    pub model: &'a dyn DynChatModel,
}

/// A rejected reply carried into the retry turn: the model sees its own previous output
/// and a precise statement of what was wrong and what is required.
pub struct Feedback {
    pub previous: String,
    pub error: String,
}

/// Everything before the live request: the demos, then any conversation history, and the
/// signature the request itself is rendered against.
///
/// dspy assembles this in its base `format` for both adapters, which is why a history field is
/// replayed and hidden the same way whichever wire format carries the request. Only the
/// assistant half of each exchange differs, so that renderer is the parameter.
fn conversation(
    signature: &Signature,
    demos: &[Example],
    inputs: &[Input<'_>],
    style: exchange::Style,
    native_tools: bool,
) -> (Signature, Vec<ChatTurn>) {
    let mut turns = demo_turns(signature, demos, style);
    let asked = match history::field_name(signature) {
        None => signature.clone(),
        Some(name) => {
            // Every turn dspy builds for a history — the replayed exchanges and the live request
            // alike — is rendered without the history field, which is why it appears in none of them.
            // The system message is built from the original signature and does still announce it.
            let stripped = signature.delete(name);
            if let Some(found) = inputs.iter().find(|input| input.name == name) {
                turns.extend(history::turns(&stripped, &found.value, style, native_tools));
            }
            stripped
        }
    };
    (asked, turns)
}

/// The inputs the request renders: everything the asked-for signature still declares, which
/// drops the history field once its exchanges have been replayed.
///
/// Walked in the signature's order rather than the caller's, because dspy renders from
/// `signature.input_fields` and the order a caller happened to name its values in never reaches
/// a prompt. Filtering the caller's list instead keeps that order, which agrees only while every
/// caller passes values as the signature declares them.
fn live_inputs<'a>(asked: &Signature, inputs: &[Input<'a>]) -> Vec<Input<'a>> {
    asked
        .inputs
        .iter()
        .filter_map(|declared| inputs.iter().find(|input| input.name == declared.name))
        .cloned()
        .collect()
}

/// The turns a module sends for one attempt: whatever the adapter rendered, plus the rejected
/// reply and its error when this is a feedback retry.
pub fn turns_for(mut turns: Vec<ChatTurn>, feedback: Option<&Feedback>) -> Vec<ChatTurn> {
    if let Some(feedback) = feedback {
        turns.push(ChatTurn::assistant(feedback.previous.clone()));
        turns.push(ChatTurn::user(format!(
            "Your previous reply was rejected: {}. Send the corrected reply now, in the same \
             format, with every output field present and valid.",
            feedback.error
        )));
    }
    turns
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversation_appends_previous_output_and_error_on_retry() {
        let feedback = Feedback {
            previous: "[[ ## color ## ]]\ngreen".into(),
            error: "color must be one of red, blue; got \"green\"".into(),
        };
        let turns = turns_for(vec![ChatTurn::user("draft it")], Some(&feedback));
        assert_eq!(turns.len(), 3);
        assert_eq!(turns[1].content.text().unwrap(), "[[ ## color ## ]]\ngreen");
        assert!(
            turns[2]
                .content
                .text()
                .unwrap()
                .contains("color must be one of red, blue")
        );
        assert!(turns_for(vec![ChatTurn::user("draft it")], None).len() == 1);
    }
}

#[cfg(test)]
mod live_input_order {
    use super::*;
    use crate::signature::{InField, OutField};
    use serde_json::json;

    fn two_inputs() -> Signature {
        Signature {
            instructions: "T.".to_owned(),
            inputs: vec![
                InField {
                    name: "alpha".to_owned(),
                    ..Default::default()
                },
                InField {
                    name: "beta".to_owned(),
                    ..Default::default()
                },
            ],
            outputs: vec![OutField {
                name: "answer".to_owned(),
                ..Default::default()
            }],
        }
    }

    /// A field named twice renders once, because upstream's inputs are a dict and cannot hold
    /// the same key twice. `Example::set` replaces rather than appends, so a duplicate cannot
    /// reach here from a module either — this pins the helper against a caller building pairs
    /// by hand.
    #[test]
    fn a_field_named_twice_renders_once() {
        let signature = two_inputs();
        let repeated = [
            Input::new("alpha", json!("first")),
            Input::new("beta", json!("B")),
            Input::new("alpha", json!("second")),
        ];
        let rendered = live_inputs(&signature, &repeated);
        let names: Vec<&str> = rendered.iter().map(|input| input.name).collect();
        assert_eq!(names, ["alpha", "beta"]);
    }

    /// dspy walks the signature's own input list to render the live request, so the order a
    /// caller happened to pass values in never reaches a prompt. Every adapter shares this, and
    /// no existing caller passed them out of order, so nothing caught it.
    #[test]
    fn the_request_renders_in_signature_order_not_call_order() {
        let signature = two_inputs();
        let asked_backwards = [
            Input::new("beta", json!("B")),
            Input::new("alpha", json!("A")),
        ];
        let rendered = live_inputs(&signature, &asked_backwards);
        let names: Vec<&str> = rendered.iter().map(|input| input.name).collect();
        assert_eq!(names, ["alpha", "beta"]);
    }
}

#[cfg(test)]
mod adapter_names {
    use super::*;
    use crate::{BamlAdapter, XmlAdapter};

    /// Each adapter reports the name of the dspy class it is a port of.
    ///
    /// dspy dispatches a callback by the instance's *type* and hands the handler that instance, so
    /// what a watcher reads is `type(instance).__name__`. A Rust callback cannot hand over a `dyn
    /// Adapter` for the caller to downcast, so [`Adapter::name`] stands in — and it is a stand-in
    /// only if it says what upstream's class is called.
    ///
    /// Three of the five said the Rust type's name instead (`JsonAdapter` for `JSONAdapter`, and
    /// so on). Nothing asserted any of them: mutating `ChatAdapter::name` to the empty string left
    /// the whole suite green, which is how the disagreement surfaced.
    #[test]
    fn every_adapter_reports_its_dspy_class_name() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/conformance/constants/tables.json");
        let text = std::fs::read_to_string(&path).expect("the constants golden is committed");
        let tables: serde_json::Value = serde_json::from_str(&text).expect("the golden parses");
        let recorded = tables["adapter_names"].as_object().expect("adapter names");

        let ours: Vec<(&str, &str)> = vec![
            ("chat", ChatAdapter::default().name()),
            ("json", JsonAdapter::default().name()),
            ("xml", XmlAdapter.name()),
            ("baml", BamlAdapter.name()),
        ];
        for (wire, name) in &ours {
            assert_eq!(
                Some(*name),
                recorded[*wire].as_str(),
                "the {wire} adapter should report dspy's class name"
            );
        }
        assert_eq!(
            recorded.len(),
            ours.len() + 1,
            "the golden records TwoStepAdapter too, which needs a model to build"
        );
    }
}
