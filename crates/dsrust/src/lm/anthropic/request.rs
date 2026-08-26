//! The Anthropic messages body, built the way litellm builds it.
//!
//! dspy 3.3 ships no native Anthropic wire — it routes the provider through litellm — so litellm's
//! transform is the reference here, and this mirrors its pipeline: the OpenAI-shaped messages
//! [`wire_messages`](api::LmRequest::wire_messages) builds — litellm's own input — are remapped to
//! Anthropic's form. The system prompt and every message's content become block lists, an assistant
//! turn's `tool_calls` become `tool_use` blocks and a tool result a `tool_result` block, tools take
//! Anthropic's `input_schema`/`custom` shape, a cap is always present (litellm defaults it to 4096),
//! stops are `stop_sequences`, and a requested schema rides as a forced `json_tool_call`.
//! `tests/lm_api_conformance.rs` holds the body to litellm's own, captured case for case.

use serde_json::{Value, json};

use crate::lm::api::{self, LmToolChoice, LmToolSpec, ToolChoiceMode};

/// litellm's default when a call names no cap, which Anthropic requires.
const DEFAULT_MAX_TOKENS: u32 = 4096;

/// The tool litellm injects to make Anthropic answer in a schema: the model calls it, its arguments
/// are the structured reply. Named exactly as litellm names it, since a caller sees it back.
const JSON_TOOL: &str = "json_tool_call";

/// The Anthropic messages body for one call.
pub(super) fn request(model: &str, call: &api::LmRequest) -> Value {
    let config = &call.config;
    let messages = call.wire_messages();
    let mut body = json!({
        "model": model,
        "messages": conversation(&messages),
        "max_tokens": config.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
    });
    if let Some(system) = system(&messages) {
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

/// The conversation as Anthropic messages: the system prompt is lifted into its own field, so it is
/// dropped here; every other OpenAI-shaped message is remapped by [`anthropic_message`].
fn conversation(messages: &[Value]) -> Vec<Value> {
    messages
        .iter()
        .filter(|message| message["role"] != "system")
        .map(anthropic_message)
        .collect()
}

/// One OpenAI-shaped message as its Anthropic form — the remap litellm performs. A tool result
/// becomes a `user` message carrying a `tool_result` block; an assistant turn's `tool_calls` become
/// `tool_use` blocks beside its content; anything else is its content as a block list.
fn anthropic_message(message: &Value) -> Value {
    if message["role"] == "tool" {
        return json!({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": message["tool_call_id"],
                "content": message["content"],
            }],
        });
    }
    let mut blocks = content(&message["content"]);
    for call in message["tool_calls"].as_array().into_iter().flatten() {
        blocks.push(json!({
            "type": "tool_use",
            "id": call["id"],
            "name": call["function"]["name"],
            "input": parsed_args(&call["function"]["arguments"]),
        }));
    }
    json!({ "role": message["role"], "content": blocks })
}

/// OpenAI-shaped `content` as Anthropic content blocks: a bare string becomes one text block, a block
/// list maps each block, and `null` — an assistant turn that is only tool calls — becomes no blocks.
fn content(content: &Value) -> Vec<Value> {
    match content {
        Value::String(text) => vec![text_block(text)],
        Value::Array(blocks) => blocks.iter().map(block).collect(),
        _ => Vec::new(),
    }
}

/// The system prompt as Anthropic's top-level `system` field, a block list, or `None` when the call
/// carries no system message.
fn system(messages: &[Value]) -> Option<Value> {
    let message = messages
        .iter()
        .find(|message| message["role"] == "system")?;
    Some(Value::Array(content(&message["content"])))
}

/// A tool call's arguments — the `json.dumps` string OpenAI carries — parsed back to the object that
/// Anthropic's `tool_use` `input` expects.
fn parsed_args(arguments: &Value) -> Value {
    arguments
        .as_str()
        .and_then(|text| serde_json::from_str(text).ok())
        .unwrap_or_else(|| json!({}))
}

/// One OpenAI-shaped content block as its Anthropic form. An `image_url` becomes an `image` with a
/// typed `source`; everything else is carried through unchanged — including text.
///
/// Text passes through rather than being rebuilt, because the two shapes already agree on
/// `{type, text}` and a rebuild strips whatever rides alongside. That is not hypothetical:
/// Anthropic's prompt caching is a `cache_control` key *on a content block*, so a caller who put
/// one on a carried text block had asked for caching, and the rebuild silently un-asked. A
/// surviving mutant found this — deleting the text arm changed nothing any test could see, which
/// meant the rebuild's only observable effect was the stripping.
fn block(block: &Value) -> Value {
    match block["type"].as_str() {
        Some("image_url") => {
            json!({ "type": "image", "source": image_source(&block["image_url"]) })
        }
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
        Some((media_type, data)) => {
            json!({ "type": "base64", "media_type": media_type, "data": data })
        }
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

    /// The named-tool shortcut needs *both* halves: exactly one allowed tool, and a mode that
    /// permits calling it. Two allowed under `required` is `any`; one allowed under `none` is
    /// still `none`, because the caller forbade tools and the allowlist does not un-forbid them.
    #[test]
    fn tool_choice_names_a_tool_only_when_one_is_allowed_and_callable() {
        use crate::lm::api::{LmToolChoice, ToolChoiceMode};
        let one = |mode| LmToolChoice {
            mode,
            allowed: Some(vec!["get_weather".to_owned()]),
            ..LmToolChoice::default()
        };
        assert_eq!(
            tool_choice(&one(ToolChoiceMode::Required)),
            json!({ "type": "tool", "name": "get_weather" })
        );
        assert_eq!(
            tool_choice(&one(ToolChoiceMode::None)),
            json!({ "type": "none" })
        );
        let two = LmToolChoice {
            mode: ToolChoiceMode::Required,
            allowed: Some(vec!["a".to_owned(), "b".to_owned()]),
            ..LmToolChoice::default()
        };
        assert_eq!(tool_choice(&two), json!({ "type": "any" }));
    }

    /// A carried text block keeps whatever rides on it — `cache_control` is Anthropic's prompt
    /// caching, declared per block, and rebuilding `{type, text}` silently un-asks for it.
    #[test]
    fn a_text_blocks_cache_control_reaches_the_request() {
        let carried = json!({
            "type": "text",
            "text": "the corpus",
            "cache_control": { "type": "ephemeral" },
        });
        assert_eq!(block(&carried), carried);
    }

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
