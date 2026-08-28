//! The Responses API's tool dialect, which is not the chat API's.
//!
//! dspy shipped one renderer for both wires until 3.3.0. The two split when `strict` arrived, and
//! they disagree about shape rather than content: this one is flat where the chat dialect nests
//! under `function`, and it refuses a `tool_choice` the wire cannot express where the chat one
//! falls back. A separate file because "which dialect" is the only question either function asks.

use anyhow::Result;
use serde_json::{Value, json};

use super::super::provider_extras;
use crate::lm::api;

/// dspy 3.3.0's `tool_to_openai_responses`: the Responses API's function-tool shape.
///
/// Flat, where the chat dialect nests under `function` — `{type, name, parameters}` with the
/// optional fields beside them rather than inside. Until 3.3.0 both wires shared
/// `tool_to_openai`, so this wire sent the chat shape and OpenAI took it; the two renderers split
/// when `strict` arrived, and the provider extras land at the top level here for the same reason
/// they land under `function` there — each dialect puts them where it puts its function fields.
pub(super) fn tool_item(tool: &api::LmToolSpec) -> Value {
    let mut data = json!({
        "type": tool.r#type,
        "name": tool.name,
        "parameters": tool.parameters,
    });
    if let Some(description) = &tool.description {
        data["description"] = json!(description);
    }
    if let Some(strict) = tool.strict {
        data["strict"] = json!(strict);
    }
    for (key, value) in provider_extras(tool) {
        data[key] = value.clone();
    }
    data
}

/// dspy 3.3.0's `tool_choice_to_openai_responses`: the Responses API's flat `tool_choice`.
///
/// `{"type": "function", "name": …}`, where the chat dialect nests the name under `"function"`.
/// Both wires shared one renderer until 3.3.0.
///
/// And this one **refuses** a constraint the wire cannot express, where the chat renderer falls
/// back to the bare mode. Reproduced rather than softened: the body builder already returns a
/// `Result`, so there is no reason to answer a request upstream rejects.
pub(super) fn apply_responses_tool_choice(
    body: &mut Value,
    choice: &api::LmToolChoice,
) -> Result<()> {
    match choice.allowed.as_deref() {
        Some(allowed) => {
            if allowed.len() != 1
                || !matches!(
                    choice.mode,
                    api::ToolChoiceMode::Required | api::ToolChoiceMode::Auto
                )
            {
                anyhow::bail!(
                    "OpenAI Responses tool_choice only supports constraining to a single allowed \
                     tool with mode 'required' or 'auto'."
                );
            }
            body["tool_choice"] = json!({ "type": "function", "name": allowed[0] });
        }
        None => body["tool_choice"] = json!(choice.mode),
    }
    if let Some(parallel) = choice.parallel {
        body["parallel_tool_calls"] = json!(parallel);
    }
    Ok(())
}
