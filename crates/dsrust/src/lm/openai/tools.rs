//! How a tool reaches OpenAI's two wires, and what comes back when its arguments will not read.
//!
//! Its own module because these are *rules about a tool spec* rather than steps in building a
//! request: which keys a dialect writes itself and so `provider_data` may not shadow, where the
//! extras go in each dialect, and how a tool choice is spelled. Both wires read them, and both
//! meet the same tool call whose arguments the model malformed.

use serde_json::{Value, json};

use crate::lm::api;

/// The wire keys a dialect writes itself, so `provider_data` cannot overwrite one.
///
/// dspy 3.3.0's `_TOOL_SPEC_WIRE_KEYS`. Before it, `data.update(tool.provider_data)` merged the
/// whole map at the top level and a caller could shadow `parameters` from provider data.
const TOOL_WIRE_KEYS: [&str; 5] = ["type", "name", "description", "parameters", "strict"];

/// A tool call whose arguments would not read: what the model wrote, and why it did not parse.
///
/// dspy records both in `provider_data` and empties the args, so a caller can tell a call made
/// with no arguments from one whose arguments could not be read — the two are the same empty map
/// otherwise, and only one of them is the model's fault. Shared by both readers because both
/// dialects meet the same malformed call.
///
/// The *reason* is this parser's, where upstream's is CPython's `json` — a message no port can
/// reproduce without reimplementing another language's error prose. Upstream's own test asserts
/// the key is there rather than what it says, which is the line this holds to.
pub(in crate::lm) fn unreadable_arguments(
    provider_data: &mut api::Metadata,
    arguments: &str,
    why: Option<serde_json::Error>,
) -> api::Metadata {
    provider_data.insert("raw_arguments".to_owned(), json!(arguments));
    let why = why.map_or_else(
        || "tool-call arguments are not a JSON object".to_owned(),
        |error| error.to_string(),
    );
    provider_data.insert("arguments_parse_error".to_owned(), json!(why));
    api::Metadata::new()
}

/// `provider_data` minus what the dialect writes itself — dspy's `_tool_provider_extras`.
pub(in crate::lm) fn provider_extras(
    tool: &api::LmToolSpec,
) -> impl Iterator<Item = (&String, &Value)> {
    tool.provider_data
        .iter()
        .filter(|(key, _)| !TOOL_WIRE_KEYS.contains(&key.as_str()))
}

/// dspy's `tool_to_openai`: a function tool, its provider extras merged **under `function`**.
///
/// They went at the top level until 3.3.0, which moved them beside the other function fields —
/// "each dialect emits them where it puts function fields", nested here and flattened in the
/// Responses shape.
pub(in crate::lm) fn tool_json(tool: &api::LmToolSpec) -> Value {
    let mut data = json!({
        "type": tool.r#type,
        "function": { "name": tool.name, "parameters": tool.parameters },
    });
    if let Some(description) = &tool.description {
        data["function"]["description"] = json!(description);
    }
    // Only when set: `strict` is a field of the typed spec, where it serialises as `null`, and a
    // key of the body only when the caller asked for it.
    if let Some(strict) = tool.strict {
        data["function"]["strict"] = json!(strict);
    }
    for (key, value) in provider_extras(tool) {
        data["function"][key] = value.clone();
    }
    data
}

/// dspy's `tool_choice_to_openai`: one named tool when exactly one is allowed under `auto`/
/// `required`, otherwise the bare mode. A wider constraint OpenAI cannot express falls back to the
/// mode rather than raising, since the request builder has no way to report an error.
pub(in crate::lm) fn apply_tool_choice(request: &mut Value, choice: &api::LmToolChoice) {
    let single = choice.allowed.as_ref().filter(|allowed| {
        allowed.len() == 1
            && matches!(
                choice.mode,
                api::ToolChoiceMode::Auto | api::ToolChoiceMode::Required
            )
    });
    request["tool_choice"] = match single {
        Some(allowed) => json!({ "type": "function", "function": { "name": allowed[0] } }),
        None => serde_json::to_value(choice.mode).unwrap_or(Value::Null),
    };
    if let Some(parallel) = choice.parallel {
        request["parallel_tool_calls"] = json!(parallel);
    }
}
