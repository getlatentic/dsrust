//! What a provider answered on its own channels, and the request knobs that ask for them.
//!
//! dspy's `_call_preprocess` takes a field off the render when the provider will supply it
//! natively — the tool calls, the model's own thinking — and `_call_postprocess` fills that field
//! from the channel instead of from the reply text. Both halves live here: what to put in the
//! request to ask, and how to read what came back.

use anyhow::{Result, anyhow};
use serde_json::{Map, Value};

use super::{Predict, Reply};
use crate::adapter::{ToolCall, ToolCalls, native_tools};
use crate::lm::api;
use crate::signature::FieldKind;

/// dspy sets `parallel_tool_calls` only when the adapter states one and the request has not
/// already asked, so an explicit `false` from the caller is never overwritten by the default.
///
/// The normalized config spells it `tool_choice.parallel`, which is where upstream's own
/// `LMToolChoice.from_value` puts a `parallel_tool_calls` kwarg.
pub(super) fn ask_for_parallel_calls(request: &mut api::LmRequest, parallel: Option<bool>) {
    let Some(parallel) = parallel else {
        return;
    };
    let choice = request
        .config
        .tool_choice
        .get_or_insert_with(Default::default);
    if choice.parallel.is_none() {
        choice.parallel = Some(parallel);
    }
}

/// dspy's forced `tool_choice`: pin the provider to one tool by name. The normalized config states
/// it as the sole allowed tool under `required`, which every provider lowers to
/// `{"type":"function","function":{"name": tool}}`.
pub(super) fn force_tool(request: &mut api::LmRequest, tool: &str) {
    let choice = request
        .config
        .tool_choice
        .get_or_insert_with(Default::default);
    choice.mode = api::ToolChoiceMode::Required;
    choice.allowed = Some(vec![tool.to_owned()]);
}

/// dspy `_provider_tool_call_to_tool_call_dict`: a provider's native call as a [`ToolCall`].
///
/// The reply parser already structured the arguments and settled the id from `id`/`call_id`, so a
/// part carries what upstream reads out of a litellm tool call — no repair or renaming is left to
/// do here. A part that is not a tool call is not one of these and is dropped.
fn as_tool_call(part: &api::LmPart) -> Option<ToolCall> {
    match part {
        api::LmPart::ToolCall { id, name, args, .. } => Some(ToolCall {
            id: id.clone(),
            name: name.clone(),
            args: args.clone(),
        }),
        _ => None,
    }
}

/// dspy `Reasoning.parse_lm_response` reads `reasoning_content` off a reply; here that is the
/// thinking part the providers already lift out of it.
fn thinking_text(output: &api::LmOutput) -> Option<String> {
    output.parts.iter().find_map(|part| match part {
        api::LmPart::Thinking { text, .. } => Some(text.clone()),
        _ => None,
    })
}

impl<S> Predict<S> {
    /// dspy `_call_postprocess`: the reply's value when a native feature answered for a field.
    ///
    /// A field the render dropped is not one the reply spoke, so it is read from its own channel
    /// instead — the provider's tool calls fill the tool-call output, the model's own thinking
    /// fills the reasoning output — while whatever content did come back is parsed against what
    /// remained. Returns `None` when the render kept every output field, which is the marker text
    /// path the caller handles itself.
    pub(super) fn native_value(&self, reply: &Reply) -> Result<Option<Value>> {
        let removed: Vec<&crate::signature::OutField> = self
            .signature
            .outputs
            .iter()
            .filter(|field| {
                !reply
                    .rendered
                    .outputs
                    .iter()
                    .any(|kept| kept.name == field.name)
            })
            .collect();
        if removed.is_empty() {
            return Ok(None);
        }

        let text = reply.response.first_text();
        let mut value = match !text.is_empty() && !reply.rendered.outputs.is_empty() {
            true => self
                .adapter
                .parse(&reply.rendered, &text)
                .unwrap_or_else(|_| Value::Object(Map::new())),
            false => Value::Object(Map::new()),
        };
        let object = value
            .as_object_mut()
            .ok_or_else(|| anyhow!("a parsed reply is an object"))?;
        for field in &self.signature.outputs {
            object.entry(field.name.clone()).or_insert(Value::Null);
        }

        // The provider's own tool calls fill the tool-call output field.
        if let Some(tool_field) = native_tools::tool_call_output_field(&self.signature) {
            let calls: Vec<ToolCall> = reply
                .response
                .outputs
                .first()
                .into_iter()
                .flat_map(api::LmOutput::tool_calls)
                .filter_map(as_tool_call)
                .collect();
            if !calls.is_empty() {
                object.insert(
                    tool_field.to_owned(),
                    ToolCalls::new(calls).to_value_with_ids(),
                );
            }
        }

        // dspy `Reasoning.parse_lm_response`: a reasoning model's thinking fills the reasoning
        // output that left the render for it.
        if let Some(thinking) = reply.response.outputs.first().and_then(thinking_text) {
            for field in &removed {
                if matches!(field.kind, FieldKind::Reasoning) {
                    object.insert(field.name.clone(), Value::String(thinking.clone()));
                }
            }
        }
        Ok(Some(value))
    }
}
