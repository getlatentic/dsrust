//! One output item of a Responses reply, as the part it stands for.
//!
//! The three shapes dspy names — text, refusal, function call — plus whatever `media` can make of
//! the rest. Split from the reply reader because that reads the envelope and this reads one item,
//! and only this half has to know what an item's `type` can say.

use serde_json::Value;

use super::media;
use crate::lm::api::{self, Metadata};

/// dspy's `response_content_item_to_parts`: a text item (or a bare `{text}`) as text, a refusal as a
/// refusal part, a function call as a tool call. Image/audio/file output items are not modelled here.
pub(super) fn content_item_parts(item: &Value) -> Vec<api::LmPart> {
    let item_type = item["type"].as_str();
    let text = item["text"].as_str();
    if matches!(item_type, Some("text" | "output_text" | "input_text"))
        || (text.is_some() && item_type.is_none())
    {
        return vec![api::LmPart::text(text.unwrap_or_default())];
    }
    match item_type {
        Some("refusal" | "output_refusal") => vec![refusal(item)],
        Some("tool_call" | "function_call") => vec![function_call_part(item)],
        Some(other) => media::part(other, item).into_iter().collect(),
        None => Vec::new(),
    }
}

/// dspy's `refusal_to_part`: the decline text read from whichever field the item carried it in.
fn refusal(item: &Value) -> api::LmPart {
    let text = item["refusal"]
        .as_str()
        .or_else(|| item["text"].as_str())
        .or_else(|| item["content"].as_str())
        .unwrap_or_default();
    api::LmPart::refusal(text)
}

/// dspy's `responses_function_call_to_part`: name and arguments at the top level (not under
/// `function`), the whole raw item kept as provider data.
pub(super) fn function_call_part(item: &Value) -> api::LmPart {
    let arguments = item["arguments"].as_str().unwrap_or("{}");
    let mut provider_data: Metadata = item.as_object().cloned().unwrap_or_default();
    let args = match serde_json::from_str::<Value>(arguments) {
        Ok(Value::Object(map)) => map,
        parsed => super::super::unreadable_arguments(&mut provider_data, arguments, parsed.err()),
    };
    api::LmPart::ToolCall {
        id: item["call_id"].as_str().map(str::to_owned),
        name: item["name"].as_str().unwrap_or_default().to_owned(),
        args,
        provider_data,
        metadata: Metadata::new(),
    }
}
