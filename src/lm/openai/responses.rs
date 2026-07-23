//! The OpenAI Responses API — dspy 3.3's `to_openai_responses_request` / `responses_to_lm_response`.
//!
//! OpenAI's second wire, used for reasoning models. The request is a flat `input` list of items
//! rather than `messages`: `input_text`/`input_image` content, `function_call`/`function_call_output`
//! items for a tool exchange, `max_output_tokens`, and `reasoning: {effort, summary}`. The reply is
//! one output whose parts are the response's output items — reasoning as thinking, message content as
//! text, a function_call as a tool call, a refusal as its own part. Both are built from the same
//! OpenAI-shaped pieces the chat wire uses, and `tests/lm_api_conformance.rs` holds them to dspy's.

use anyhow::{Result, anyhow};
use serde_json::{Value, json};

use super::{apply_tool_choice, response, tool_json};
use crate::lm::api::{self, Metadata};

// -------- request: LmRequest -> Responses body --------

/// The Responses request body for one call. `response_format` is not built here — like the chat wire
/// fixture, dspy carries the whole `text.format` envelope while this crate stores the bare schema.
pub(super) fn request(model: &str, call: &api::LmRequest) -> Value {
    let config = &call.config;
    let mut body = json!({ "model": model, "input": input(&call.wire_messages()) });
    // dspy's `responses_config_kwargs` opens with the extensions, unknown kwargs passing through.
    for (key, value) in &config.extensions {
        body[key] = value.clone();
    }
    if let Some(temperature) = config.temperature {
        body["temperature"] = json!(temperature);
    }
    if let Some(top_p) = config.top_p {
        body["top_p"] = json!(top_p);
    }
    if let Some(n) = config.n {
        body["n"] = json!(n);
    }
    if let Some(logprobs) = &config.logprobs {
        body["logprobs"] = serde_json::to_value(logprobs).unwrap_or(Value::Null);
    }
    if let Some(stop) = &config.stop
        && !stop.is_empty()
    {
        body["stop"] = json!(stop);
    }
    // The cap is `max_output_tokens` here, not `max_tokens`.
    if let Some(max_tokens) = config.max_tokens {
        body["max_output_tokens"] = json!(max_tokens);
    }
    if let Some(reasoning) = reasoning(call) {
        body["reasoning"] = reasoning;
    }
    if let Some(cache) = &config.prompt_cache
        && let Some(key) = &cache.key
    {
        body["prompt_cache_key"] = json!(key);
    }
    if let Some(choice) = &config.tool_choice {
        apply_tool_choice(&mut body, choice);
    }
    if !call.tools.is_empty() {
        body["tools"] = Value::Array(call.tools.iter().map(tool_json).collect());
    }
    body
}

/// The OpenAI-shaped messages as Responses input items — dspy's `message_to_responses_input_items`,
/// applied to the already-rendered messages. A tool result is a `function_call_output`; an assistant
/// turn's tool calls are `function_call` items beside its content (dropped when the turn is only
/// calls); anything else is its content as `input_*` blocks.
fn input(messages: &[Value]) -> Vec<Value> {
    let mut items = Vec::new();
    for message in messages {
        if message["role"] == "tool" {
            let mut item = json!({ "type": "function_call_output", "output": output_text(&message["content"]) });
            if let Some(id) = message.get("tool_call_id") {
                item["call_id"] = id.clone();
            }
            items.push(item);
            continue;
        }
        let tool_calls = message["tool_calls"].as_array();
        let content = &message["content"];
        // The content item is emitted unless this is an assistant turn of only tool calls.
        let assistant_only_calls =
            message["role"] == "assistant" && falsy(content) && tool_calls.is_some();
        if !assistant_only_calls {
            let mut item = json!({ "role": message["role"], "content": content_blocks(content) });
            if let Some(name) = message.get("name") {
                item["name"] = name.clone();
            }
            items.push(item);
        }
        for call in tool_calls.into_iter().flatten() {
            items.push(function_call(call));
        }
    }
    items
}

/// dspy's truthiness for a message's content: a null, an empty string, or an empty list is falsy, so
/// an assistant turn carrying only tool calls drops its content item.
fn falsy(content: &Value) -> bool {
    content.is_null()
        || content.as_str() == Some("")
        || content.as_array().is_some_and(|blocks| blocks.is_empty())
}

/// OpenAI-shaped `content` as Responses input blocks: a bare string is one `input_text`, a block list
/// maps each — text to `input_text`, an image to `input_image` (its url a bare string, not an object).
fn content_blocks(content: &Value) -> Vec<Value> {
    match content {
        Value::String(text) => vec![json!({ "type": "input_text", "text": text })],
        Value::Array(blocks) => blocks.iter().map(content_block).collect(),
        _ => Vec::new(),
    }
}

fn content_block(block: &Value) -> Value {
    match block["type"].as_str() {
        Some("text") => json!({ "type": "input_text", "text": block["text"] }),
        Some("image_url") => {
            let mut out = json!({ "type": "input_image", "image_url": block["image_url"]["url"] });
            if let Some(detail) = block["image_url"].get("detail").filter(|detail| !detail.is_null()) {
                out["detail"] = detail.clone();
            }
            out
        }
        _ => block.clone(),
    }
}

/// One OpenAI-shaped tool call as a Responses `function_call` item: name and arguments at the top
/// level rather than under `function`, the id kept from either spelling.
fn function_call(call: &Value) -> Value {
    let function = &call["function"];
    let mut item = json!({
        "type": "function_call",
        "name": function["name"],
        "arguments": function["arguments"],
    });
    if let Some(id) = call["id"]
        .as_str()
        .or_else(|| call["call_id"].as_str())
    {
        item["call_id"] = json!(id);
    }
    item
}

/// dspy's `responses_tool_output_text`: a bare string as itself, a block list as its joined text.
fn output_text(content: &Value) -> String {
    match content {
        Value::String(text) => text.clone(),
        Value::Array(blocks) => blocks.iter().filter_map(|block| block["text"].as_str()).collect(),
        _ => String::new(),
    }
}

/// dspy's `reasoning_to_responses_kwargs`: the effort and the summary — not the max-tokens, which the
/// Responses API does not take — under a `reasoning` object, or `None` when neither was set.
fn reasoning(call: &api::LmRequest) -> Option<Value> {
    let reasoning = call.config.reasoning.as_ref()?;
    let mut data = serde_json::Map::new();
    if let Some(effort) = &reasoning.effort {
        data.insert("effort".to_owned(), json!(effort));
    }
    if let Some(summary) = &reasoning.summary {
        data.insert("summary".to_owned(), json!(summary));
    }
    (!data.is_empty()).then(|| Value::Object(data))
}

// -------- reply: Responses body -> LmResponse --------

/// The reply as a typed response, or the message the service itself gave for refusing the call.
pub(super) fn reply(
    label: &str,
    model: &str,
    status: reqwest::StatusCode,
    body: &Value,
) -> Result<api::LmResponse> {
    if !status.is_success() {
        let detail = body["error"]["message"].as_str().unwrap_or("unknown error");
        return Err(anyhow!("{label} {status}: {detail}"));
    }
    let response = responses_to_lm_response(body, model);
    if response.outputs.iter().all(|output| output.parts.is_empty()) {
        return Err(anyhow!("{label} returned no content"));
    }
    Ok(response)
}

/// dspy's `responses_to_lm_response`: one output whose parts are the response's output items in
/// order, with usage, id and cache flag alongside.
fn responses_to_lm_response(body: &Value, fallback_model: &str) -> api::LmResponse {
    let mut parts = Vec::new();
    for item in body["output"].as_array().into_iter().flatten() {
        match item["type"].as_str() {
            Some("message") => {
                for content in item["content"].as_array().into_iter().flatten() {
                    parts.extend(content_item_parts(content));
                    for annotation in content["annotations"].as_array().into_iter().flatten() {
                        parts.push(api::LmPart::citation(annotation));
                    }
                }
            }
            Some("function_call") => parts.push(function_call_part(item)),
            Some("reasoning") => {
                let source = item["content"].as_array().or_else(|| item["summary"].as_array());
                for entry in source.into_iter().flatten() {
                    if let Some(text) = entry["text"].as_str().filter(|text| !text.is_empty()) {
                        parts.push(api::LmPart::thinking(text, false));
                    }
                }
            }
            _ => {}
        }
    }
    let output = api::LmOutput {
        parts,
        provider_output: Some(body.clone()),
        ..api::LmOutput::default()
    };
    api::LmResponse {
        model: Some(body["model"].as_str().unwrap_or(fallback_model).to_owned()),
        outputs: vec![output],
        usage: response::usage(&body["usage"]),
        cache_hit: body["cache_hit"].as_bool().unwrap_or(false),
        response_id: body["id"].as_str().map(str::to_owned),
        provider_response: Some(body.clone()),
        ..api::LmResponse::default()
    }
}

/// dspy's `response_content_item_to_parts`: a text item (or a bare `{text}`) as text, a refusal as a
/// refusal part, a function call as a tool call. Image/audio/file output items are not modelled here.
fn content_item_parts(item: &Value) -> Vec<api::LmPart> {
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
        _ => Vec::new(),
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
fn function_call_part(item: &Value) -> api::LmPart {
    let arguments = item["arguments"].as_str().unwrap_or("{}");
    let mut provider_data: Metadata = item.as_object().cloned().unwrap_or_default();
    let args = match serde_json::from_str::<Value>(arguments) {
        Ok(Value::Object(map)) => map,
        _ => {
            provider_data.insert("raw_arguments".to_owned(), json!(arguments));
            Metadata::new()
        }
    };
    api::LmPart::ToolCall {
        id: item["call_id"].as_str().map(str::to_owned),
        name: item["name"].as_str().unwrap_or_default().to_owned(),
        args,
        provider_data,
        metadata: Metadata::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Faithfulness to dspy 3.3's Responses request: our body equals `to_openai_responses_request`'s
    /// for the same typed request — the `input` list, `input_text`/`input_image` blocks, function
    /// call items, `max_output_tokens`, `reasoning`. Generated by running dspy.
    #[test]
    fn our_body_matches_dspy_33_to_openai_responses_request() {
        for case in request_fixture()["request_cases"].as_array().expect("request cases") {
            let name = case["name"].as_str().expect("a case name");
            let call: api::LmRequest = serde_json::from_value(case["lm_request"].clone())
                .unwrap_or_else(|error| panic!("{name}: the typed request did not parse: {error}"));
            assert_eq!(
                request(&call.model, &call),
                case["expected"],
                "{name}: our Responses body diverges from dspy's"
            );
        }
    }

    /// Faithfulness to dspy 3.3's Responses reply: our `reply` parses each raw Responses object into
    /// the same `LMResponse` `responses_to_lm_response` builds — output items as parts, usage aliased,
    /// id and cache flag kept. Structural compare; runtime-only provider fields cleared.
    #[test]
    fn our_reply_matches_dspy_33_responses_to_lm_response() {
        for case in request_fixture()["reply_cases"].as_array().expect("reply cases") {
            let name = case["name"].as_str().expect("a case name");
            let expected: api::LmResponse = serde_json::from_value(case["lm_response"].clone())
                .unwrap_or_else(|error| panic!("{name}: dspy's LMResponse did not parse: {error}"));
            let mut ours = responses_to_lm_response(&case["response"], "openai/gpt-5");
            ours.provider_response = None;
            ours.outputs.iter_mut().for_each(|output| output.provider_output = None);
            assert_eq!(ours, expected, "{name}: our Responses reply diverges from dspy's");
        }
    }

    fn request_fixture() -> Value {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/conformance/lm_api/openai_responses.json");
        serde_json::from_str(&std::fs::read_to_string(path).expect("fixture is readable"))
            .expect("fixture is valid json")
    }
}
