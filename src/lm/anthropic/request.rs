//! The Anthropic messages body, built the way litellm builds it.
//!
//! dspy 3.3 ships no native Anthropic wire — it routes the provider through litellm — so litellm's
//! transform is the reference here. The system prompt and every message's content become block
//! lists, tools take Anthropic's own `input_schema`/`custom` shape, a generation cap is always
//! present (litellm defaults it to 4096), sampling stops are `stop_sequences`, and a requested
//! schema rides as a forced `json_tool_call`. `tests/lm_api_conformance.rs` holds the body to
//! litellm's own, captured case for case.

use serde_json::{Value, json};

use crate::lm::api::{self, Content, LmToolChoice, LmToolSpec, ToolChoiceMode, content_of};

/// litellm's default when a call names no cap, which Anthropic requires.
const DEFAULT_MAX_TOKENS: u32 = 4096;

/// The tool litellm injects to make Anthropic answer in a schema: the model calls it, its arguments
/// are the structured reply. Named exactly as litellm names it, since a caller sees it back.
const JSON_TOOL: &str = "json_tool_call";

/// The Anthropic messages body for one call.
pub(super) fn request(model: &str, call: &api::LmRequest) -> Value {
    let config = &call.config;
    let mut body = json!({
        "model": model,
        "messages": messages(call),
        "max_tokens": config.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
    });
    if let Some(system) = system(call) {
        body["system"] = system;
    }
    if let Some(temperature) = config.temperature {
        body["temperature"] = json!(temperature);
    }
    if let Some(top_p) = config.top_p {
        body["top_p"] = json!(top_p);
    }
    if let Some(stop) = &config.stop
        && !stop.is_empty()
    {
        body["stop_sequences"] = json!(stop);
    }
    let mut tools: Vec<Value> = call.tools.iter().map(tool).collect();
    if let Some(choice) = &config.tool_choice {
        body["tool_choice"] = tool_choice(choice);
    }
    // A requested schema is litellm's `json_tool_call`: a forced tool carrying the schema, the model
    // answering by calling it. `super::reply` reads that call back out as the reply text.
    if let Some(schema) = call.output_schema() {
        tools.push(json!({ "name": JSON_TOOL, "input_schema": schema }));
        body["tool_choice"] = json!({ "type": "tool", "name": JSON_TOOL });
    }
    if !tools.is_empty() {
        body["tools"] = Value::Array(tools);
    }
    body
}

/// Every non-system message as `{role, content: [blocks]}`. The system prompt is lifted into its
/// own field, so it is dropped from the conversation here.
fn messages(call: &api::LmRequest) -> Vec<Value> {
    call.messages
        .iter()
        .filter(|message| message.role != "system")
        .map(|message| json!({ "role": message.role, "content": content(&message.parts) }))
        .collect()
}

/// A message's parts as Anthropic content blocks. They render first to the OpenAI-shaped blocks
/// [`content_of`] already builds — litellm's own input — then each maps to its Anthropic form, so a
/// text-only message becomes a one-block list rather than the bare string it stays on OpenAI.
fn content(parts: &[api::LmPart]) -> Vec<Value> {
    match content_of(parts) {
        Ok(Content::Text(text)) => vec![text_block(&text)],
        Ok(Content::Blocks(blocks)) => blocks.iter().map(block).collect(),
        Err(_) => Vec::new(),
    }
}

/// The system prompt as Anthropic's top-level `system` field, a block list, or `None` when the call
/// carries no system message.
fn system(call: &api::LmRequest) -> Option<Value> {
    let message = call.messages.iter().find(|message| message.role == "system")?;
    Some(Value::Array(content(&message.parts)))
}

/// One OpenAI-shaped content block as its Anthropic form. Text stays text; an `image_url` becomes an
/// `image` with a typed `source`. Any other block — the ones this crate's parts do not produce for a
/// chat turn — is carried through unchanged.
fn block(block: &Value) -> Value {
    match block["type"].as_str() {
        Some("text") => json!({ "type": "text", "text": block["text"] }),
        Some("image_url") => json!({ "type": "image", "source": image_source(&block["image_url"]) }),
        _ => block.clone(),
    }
}

fn text_block(text: &str) -> Value {
    json!({ "type": "text", "text": text })
}

/// An OpenAI `image_url` as Anthropic's `source`: a data URI splits into its media type and base64
/// data, a plain URL stays a URL.
fn image_source(image_url: &Value) -> Value {
    let url = image_url["url"].as_str().unwrap_or_default();
    match url
        .strip_prefix("data:")
        .and_then(|rest| rest.split_once(";base64,"))
    {
        Some((media_type, data)) => json!({ "type": "base64", "media_type": media_type, "data": data }),
        None => json!({ "type": "url", "url": url }),
    }
}

/// litellm's Anthropic tool shape: the parameters under `input_schema`, `type: "custom"`, the
/// description kept when the caller gave one.
fn tool(spec: &LmToolSpec) -> Value {
    let mut tool = json!({
        "name": spec.name,
        "input_schema": spec.parameters,
        "type": "custom",
    });
    if let Some(description) = &spec.description {
        tool["description"] = json!(description);
    }
    tool
}

/// litellm's Anthropic tool-choice: one named tool when exactly one is allowed under `auto`/
/// `required`, `any` for `required`, `auto` for `auto`, `none` to forbid tools.
fn tool_choice(choice: &LmToolChoice) -> Value {
    let single = choice.allowed.as_ref().filter(|allowed| {
        allowed.len() == 1 && matches!(choice.mode, ToolChoiceMode::Auto | ToolChoiceMode::Required)
    });
    match single {
        Some(allowed) => json!({ "type": "tool", "name": allowed[0] }),
        None => match choice.mode {
            ToolChoiceMode::Required => json!({ "type": "any" }),
            ToolChoiceMode::Auto => json!({ "type": "auto" }),
            ToolChoiceMode::None => json!({ "type": "none" }),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Faithfulness to dspy 3.3's actual Anthropic path: our body equals the one litellm puts on the
    /// wire for the same typed request — messages and system as block lists, tools in Anthropic's
    /// shape, the cap defaulted to 4096, stops renamed, a schema forced as `json_tool_call`. Every
    /// expectation is litellm's own captured output (`scripts/generate_litellm_wire_fixture.py`), so
    /// a divergence is litellm's word against ours rather than a hand-written assertion.
    #[test]
    fn our_body_matches_litellm_for_anthropic() {
        crate::lm::tests::each_case("anthropic", request);
    }
}
