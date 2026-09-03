//! Asking the provider to call tools itself, rather than rendering the calls as a field.
//!
//! dspy's `Adapter._call_preprocess` decides this before a request goes out: when the adapter is
//! set to native function calling, the signature declares tools coming in and `ToolCalls` going
//! out, and the model can actually take a tool list, the tools move from the prompt onto the
//! request and both fields leave the signature — so the rendered exchange never mentions them and
//! the provider answers with calls of its own.
//!
//! All four of those have to hold. Any one missing and the tools stay in the prompt, which is the
//! path every marker-based adapter takes by default.

use anyhow::{Result, anyhow};
use serde_json::{Map, Value};

use crate::Tool;
use crate::adapter::Input;
use crate::adapter::types::ToolCalls;
use crate::lm::Capabilities;
use crate::lm::api::LmToolSpec;
use crate::signature::Signature;

/// The annotations dspy recognises as "these are the tools", matching its check against the
/// `Tool` class for `list[Tool]` and for a bare `Tool`.
const TOOL_ANNOTATIONS: [&str; 2] = ["list[Tool]", "Tool"];

/// What native function calling changes about a request.
///
/// Both halves matter and the second is the one that surprises: when the provider is going to call
/// tools itself, the *rendered signature* loses both the tool input and the `ToolCalls` output. The
/// model is not asked to write a tool call in prose when it is about to be handed the mechanism for
/// making one, and a signature still declaring those fields would ask for both.
///
/// ```no_run
/// # use dsrust::adapter::native_tools::NativeTools;
/// # fn read(native: NativeTools, original: &dsrust::Signature) {
/// assert!(native.signature.inputs.len() < original.inputs.len());
/// assert!(native.tools.len() > 0, "the list the request carries, in the field's own order");
/// # }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct NativeTools {
    /// The tool list the request carries, in the order the field's value states them.
    pub tools: Vec<LmToolSpec>,
    /// The signature to render, with the tool input and the `ToolCalls` output both gone.
    pub signature: Signature,
}

/// The name of the input field holding the tools, if the signature declares one.
///
/// dspy `Adapter._get_tool_call_input_field_name`, which asks the annotation for `list[Tool]` or
/// `Tool`. The kinds here carry the annotation dspy would print, so the question is the same one.
pub fn tool_call_input_field(signature: &Signature) -> Option<&str> {
    signature
        .inputs
        .iter()
        .find(|field| TOOL_ANNOTATIONS.contains(&field.kind.annotation()))
        .map(|field| field.name.as_str())
}

/// The name of the output field the calls come back on, if the signature declares one.
/// dspy `Adapter._get_tool_call_output_field_name`.
pub fn tool_call_output_field(signature: &Signature) -> Option<&str> {
    signature
        .outputs
        .iter()
        .find(|field| field.kind.annotation() == ToolCalls::ANNOTATION)
        .map(|field| field.name.as_str())
}

/// What the request should carry, or `None` to render the tools into the prompt as usual.
///
/// The error is upstream's own: a signature that asks for `ToolCalls` without declaring any tools
/// to call has stated a contract nothing can satisfy, and dspy refuses it rather than sending a
/// request whose output field can only come back empty. It refuses whether or not the model
/// supports function calling, because the signature is wrong either way.
pub fn plan(
    signature: &Signature,
    inputs: &[Input<'_>],
    capabilities: Capabilities,
) -> Result<Option<NativeTools>> {
    let output_field = tool_call_output_field(signature);
    let input_field = tool_call_input_field(signature);
    let (Some(output_field), Some(input_field)) = (output_field, input_field) else {
        if let Some(output_field) = output_field {
            return Err(anyhow!(
                "You provided an output field {output_field} to receive the tool calls \
                 information, but did not provide any tools as the input. Please provide a list \
                 of tools as the input by adding an input field with type `list[dspy.Tool]`."
            ));
        }
        return Ok(None);
    };
    if !capabilities.function_calling {
        return Ok(None);
    }
    let stated = inputs
        .iter()
        .find(|input| input.name == input_field)
        .map(|input| &input.value);
    Ok(Some(NativeTools {
        tools: stated.map(specs).unwrap_or_default(),
        signature: signature.delete(output_field).delete(input_field),
    }))
}

/// The tool list a field's value states, as specs for the request.
///
/// dspy takes `inputs[field]` and wraps a lone tool in a list before formatting each, so a
/// signature declaring a bare `Tool` travels the same path as one declaring a list.
fn specs(value: &Value) -> Vec<LmToolSpec> {
    match value {
        Value::Array(tools) => tools.iter().filter_map(spec).collect(),
        single => spec(single).into_iter().collect(),
    }
}

/// dspy `Tool.format_as_litellm_function_call`: what one tool looks like on the request.
///
/// An argument is required unless its schema carries a `default`, which is a test for the key
/// rather than for a useful value: a `default` of `null` exempts an argument as surely as a
/// `default` of `""`. Being optional in *type* exempts nothing — a nullable argument with no
/// default is still one the provider must send.
fn spec(tool: &Value) -> Option<LmToolSpec> {
    let name = tool.get("name")?.as_str()?;
    let args = tool
        .get("args")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let description = tool.get("desc").and_then(Value::as_str).map(str::to_owned);
    Some(spec_from(name, description, args))
}

/// dspy `Tool.format_as_litellm_function_call`, for a tool this crate holds: the function spec a
/// provider that calls tools natively is sent. An argument whose schema carries a `default` is not
/// required, and an undocumented tool has no description, as upstream sends `None`.
pub fn spec_of(tool: &dyn Tool) -> LmToolSpec {
    let description = match tool.description().is_empty() {
        true => None,
        false => Some(tool.description().to_owned()),
    };
    let args = tool.args().as_object().cloned().unwrap_or_default();
    spec_from(tool.name(), description, args)
}

/// What a tool is as a value: dspy's `Tool` fields `name`, `desc` and `args`, which is what a
/// `list[Tool]` input field carries here and [`plan`] reads back.
pub fn manifest(tool: &dyn Tool) -> Value {
    serde_json::json!({
        "name": tool.name(),
        "desc": match tool.description().is_empty() {
            true => Value::Null,
            false => Value::String(tool.description().to_owned()),
        },
        "args": tool.args(),
    })
}

fn spec_from(name: &str, description: Option<String>, args: Map<String, Value>) -> LmToolSpec {
    let required: Vec<Value> = args
        .iter()
        .filter(|(_, schema)| schema.get("default").is_none())
        .map(|(name, _)| Value::String(name.clone()))
        .collect();
    let parameters = serde_json::json!({
        "type": "object",
        "properties": Value::Object(args),
        "required": Value::Array(required),
    });
    let mut spec = LmToolSpec::new(name, parameters.as_object().cloned().unwrap_or_default());
    spec.description = description;
    spec
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signature::{FieldKind, InField, JsonType, OutField};
    use serde_json::json;

    fn tool_signature() -> Signature {
        Signature {
            instructions: "Answer the question.".into(),
            inputs: vec![
                InField {
                    name: "question".into(),
                    ..Default::default()
                },
                InField {
                    name: "tools".into(),
                    kind: FieldKind::Json(JsonType {
                        annotation: "list[Tool]".into(),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            ],
            outputs: vec![OutField {
                name: "tool_calls".into(),
                kind: FieldKind::Json(JsonType {
                    annotation: "ToolCalls".into(),
                    ..Default::default()
                }),
                ..Default::default()
            }],
        }
    }

    fn search() -> Value {
        json!({
            "name": "search",
            "desc": "look something up",
            "args": { "query": { "type": "string" } },
        })
    }

    fn able() -> Capabilities {
        Capabilities {
            function_calling: true,
            ..Default::default()
        }
    }

    #[test]
    fn the_tools_move_onto_the_request_and_both_fields_leave_the_signature() {
        let value = json!([search()]);
        let inputs = [
            Input::new("question", json!("what is dspy")),
            Input::new("tools", value),
        ];
        let planned = plan(&tool_signature(), &inputs, able())
            .expect("plans")
            .expect("native function calling");
        assert_eq!(planned.tools.len(), 1);
        assert_eq!(planned.tools[0].name, "search");
        assert_eq!(
            planned.tools[0].description.as_deref(),
            Some("look something up")
        );
        assert_eq!(
            Value::Object(planned.tools[0].parameters.clone()),
            json!({
                "type": "object",
                "properties": { "query": { "type": "string" } },
                "required": ["query"],
            })
        );
        // Neither field is rendered: the provider is being asked, not told.
        let names: Vec<&str> = planned
            .signature
            .inputs
            .iter()
            .map(|f| f.name.as_str())
            .collect();
        assert_eq!(names, ["question"]);
        assert!(planned.signature.outputs.is_empty());
    }

    #[test]
    fn a_model_that_cannot_call_tools_renders_them_instead() {
        let inputs = [Input::new("tools", json!([search()]))];
        assert_eq!(
            plan(&tool_signature(), &inputs, Capabilities::default()).expect("plans"),
            None
        );
    }

    #[test]
    fn a_signature_with_no_tool_fields_is_left_alone() {
        let plain = Signature::single_input("Answer.", vec![OutField::default()]);
        assert_eq!(plan(&plain, &[], able()).expect("plans"), None);
    }

    /// Upstream refuses this rather than sending a request whose output field cannot be filled.
    #[test]
    fn asking_for_calls_without_offering_tools_is_refused() {
        let mut signature = tool_signature();
        signature.inputs.retain(|field| field.name != "tools");
        let error = plan(&signature, &[], able()).expect_err("refused");
        assert!(
            error
                .to_string()
                .contains("did not provide any tools as the input"),
            "{error}"
        );
        // And refused the same way when the model could not have called them anyway.
        assert!(plan(&signature, &[], Capabilities::default()).is_err());
    }

    /// dspy wraps a lone tool in a list before formatting, so a bare `Tool` field works too.
    #[test]
    fn a_single_tool_travels_the_same_path_as_a_list() {
        let mut signature = tool_signature();
        signature.inputs[1].kind = FieldKind::Json(JsonType {
            annotation: "Tool".into(),
            ..Default::default()
        });
        let inputs = [Input::new("tools", search())];
        let planned = plan(&signature, &inputs, able())
            .expect("plans")
            .expect("native");
        assert_eq!(planned.tools.len(), 1);
        assert_eq!(planned.tools[0].name, "search");
    }
}

#[cfg(test)]
mod conformance {
    use super::*;

    /// Six tools as dspy formatted them, recorded by running it. Regenerate with
    /// `scripts/generate_tool_spec_fixture.py`.
    fn golden() -> Value {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/conformance/react/tool_spec.json");
        let text = std::fs::read_to_string(&path).expect("the tool-spec golden is committed");
        serde_json::from_str(&text).expect("the golden parses")
    }

    /// What the provider is told about each tool, against what dspy tells it.
    ///
    /// The corpus separates the two ways an argument can look optional: `by_type` is nullable and
    /// has no default, so it stays required, while `by_default` is exempt. A rule that reads the
    /// type rather than the default agrees with upstream on neither.
    #[test]
    fn states_the_tools_dspy_states() {
        let recorded = golden();
        let tools = recorded["tools"].as_array().expect("tools").clone();
        assert!(!tools.is_empty(), "the golden records no tools");
        for tool in tools {
            let name = tool["name"].as_str().expect("name");
            let ours = spec(&tool).expect("a spec for every recorded tool");
            let theirs = &tool["native"]["function"];
            assert_eq!(
                ours.description.as_deref(),
                theirs["description"].as_str(),
                "{name} describes itself differently"
            );
            assert_eq!(
                serde_json::to_string(&Value::Object(ours.parameters)).unwrap(),
                serde_json::to_string(&theirs["parameters"]).unwrap(),
                "{name} states different parameters"
            );
        }
    }
}
