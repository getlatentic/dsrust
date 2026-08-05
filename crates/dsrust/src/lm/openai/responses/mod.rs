//! The OpenAI Responses API — dspy 3.3's `to_openai_responses_request` / `responses_to_lm_response`.
//!
//! OpenAI's second wire, used for reasoning models. The request is a flat `input` list of items
//! rather than `messages`: `input_text`/`input_image` content, `function_call`/`function_call_output`
//! items for a tool exchange, `max_output_tokens`, and `reasoning: {effort, summary}`. The reply is
//! one output whose parts are the response's output items — reasoning as thinking, message content as
//! text, a function_call as a tool call, a refusal as its own part. Both are built from the same
//! OpenAI-shaped pieces the chat wire uses, and `tests/lm_api_conformance.rs` holds them to dspy's.

use std::time::Duration;

use anyhow::{Result, anyhow};
use futures_util::Stream;
use serde_json::{Value, json};

use super::{JsonFormat, provider_extras, response};
use crate::lm::api::{self, Metadata};
use crate::lm::streaming::Framing;

mod media;
mod stream;

use stream::frame;

// -------- request: LmRequest -> Responses body --------

/// The Responses request body for one call. A requested schema rides under `text.format`, built from
/// the bare schema by `text_format` — this crate builds the envelope rather than carrying dspy's
/// whole one, the same split the chat wire makes.
pub fn request(
    model: &str,
    call: &api::LmRequest,
    json_format: JsonFormat,
) -> anyhow::Result<Value> {
    // As the chat builder: dspy calls the same check from `responses_config_kwargs`, naming the
    // endpoint it was reached from.
    super::reasoning_temperature::checked(&call.config, model, "responses")?;
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
    if let Some(schema) = call.output_schema() {
        body["text"] = text_format(schema, json_format);
    }
    if let Some(cache) = &config.prompt_cache
        && let Some(key) = &cache.key
    {
        body["prompt_cache_key"] = json!(key);
    }
    if let Some(choice) = &config.tool_choice {
        apply_responses_tool_choice(&mut body, choice)?;
    }
    if !call.tools.is_empty() {
        body["tools"] = Value::Array(call.tools.iter().map(tool_item).collect());
    }
    Ok(body)
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
            if let Some(detail) = block["image_url"]
                .get("detail")
                .filter(|detail| !detail.is_null())
            {
                out["detail"] = detail.clone();
            }
            out
        }
        // A file is the one block whose *keys* move: the chat wire nests them under `file`, and the
        // Responses wire carries them at the top of an `input_file`. Each is emitted even when
        // absent, as upstream emits them — a `.get` that found nothing is a null, not a gap.
        Some("file") => {
            let file = &block["file"];
            json!({
                "type": "input_file",
                "file_data": file.get("file_data").cloned().unwrap_or(Value::Null),
                "filename": file.get("filename").cloned().unwrap_or(Value::Null),
                "file_id": file.get("file_id").cloned().unwrap_or(Value::Null),
            })
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
    if let Some(id) = call["id"].as_str().or_else(|| call["call_id"].as_str()) {
        item["call_id"] = json!(id);
    }
    item
}

/// dspy's `responses_tool_output_text`: a bare string as itself, a block list as its joined text.
fn output_text(content: &Value) -> String {
    match content {
        Value::String(text) => text.clone(),
        Value::Array(blocks) => blocks
            .iter()
            .filter_map(|block| block["text"].as_str())
            .collect(),
        _ => String::new(),
    }
}

/// The Responses `text.format` for json mode, built from the bare schema the way the chat wire builds
/// its `response_format`: the object envelope by default, a strict schema when [`JsonFormat::Schema`]
/// is asked for. This is the flat shape the Responses API — and litellm's own transform — expect,
/// name/schema/strict at the format level rather than the chat wire's nested `json_schema`.
fn text_format(schema: &Value, json_format: JsonFormat) -> Value {
    match json_format {
        JsonFormat::Object => json!({ "format": { "type": "json_object" } }),
        JsonFormat::Schema => json!({
            "format": { "type": "json_schema", "name": "response", "schema": schema, "strict": true },
        }),
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

// -------- streaming: Responses SSE -> typed events --------

/// The request body with the streaming flag set.
pub(super) fn streaming_body(
    model: &str,
    call: &api::LmRequest,
    json_format: JsonFormat,
) -> anyhow::Result<Value> {
    let mut body = request(model, call, json_format)?;
    body["stream"] = json!(true);
    Ok(body)
}

/// The typed events of one streaming Responses call. Text and reasoning arrive as deltas for live
/// display; `response.completed` carries the whole reply, parsed by [`responses_to_lm_response`] and
/// handed back as the authoritative answer — so a streamed reply equals a non-streamed one exactly.
pub(super) fn stream<'h>(
    http: &'h reqwest::Client,
    url: String,
    key: Option<String>,
    label: String,
    model: String,
    body: Value,
    timeout: Duration,
) -> impl Stream<Item = Result<api::LmStreamEvent>> + Send + 'h {
    let mut request = http.post(url).timeout(timeout).json(&body);
    if let Some(key) = key {
        request = request.bearer_auth(key);
    }
    crate::lm::streaming::events(
        request.send(),
        label,
        model,
        Framing {
            separator: b"\n\n",
            frame,
        },
    )
}

/// The reply as a typed response, or the message the service itself gave for refusing the call.
pub(super) fn reply(
    label: &str,
    model: &str,
    status: reqwest::StatusCode,
    headers: &reqwest::header::HeaderMap,
    body: &Value,
) -> Result<api::LmResponse> {
    if !status.is_success() {
        if let Some(too_long) = crate::lm::ContextWindowExceeded::detected(model, body) {
            return Err(too_long.into());
        }
        return Err(
            crate::lm::LmFailure::from_body(status.as_u16(), model, label, body)
                .headers(headers)
                .into(),
        );
    }
    let response = responses_to_lm_response(body, model);
    if response
        .outputs
        .iter()
        .all(|output| output.parts.is_empty())
    {
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
                let source = item["content"]
                    .as_array()
                    .or_else(|| item["summary"].as_array());
                for entry in source.into_iter().flatten() {
                    if let Some(text) = entry["text"].as_str().filter(|text| !text.is_empty()) {
                        parts.push(api::LmPart::thinking(text, false));
                    }
                }
            }
            // A generated image, audio or file output item; anything else contributes nothing.
            Some(other) => parts.extend(media::part(other, item)),
            None => {}
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
    use crate::lm::api::{LmDelta, LmStreamEvent};
    use crate::lm::streaming::{Framed, StreamState};

    /// Faithfulness to dspy 3.3's Responses request: our body equals `to_openai_responses_request`'s
    /// for the same typed request — the `input` list, `input_text`/`input_image` blocks, function
    /// call items, `max_output_tokens`, `reasoning`. Generated by running dspy.
    #[test]
    fn our_body_matches_dspy_33_to_openai_responses_request() {
        for case in request_fixture()["request_cases"]
            .as_array()
            .expect("request cases")
        {
            let name = case["name"].as_str().expect("a case name");
            let call: api::LmRequest = serde_json::from_value(case["lm_request"].clone())
                .unwrap_or_else(|error| panic!("{name}: the typed request did not parse: {error}"));
            assert_eq!(
                request(&call.model, &call, JsonFormat::Object).expect("the body builds"),
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
        for case in request_fixture()["reply_cases"]
            .as_array()
            .expect("reply cases")
        {
            let name = case["name"].as_str().expect("a case name");
            let expected: api::LmResponse = serde_json::from_value(case["lm_response"].clone())
                .unwrap_or_else(|error| panic!("{name}: dspy's LMResponse did not parse: {error}"));
            let mut ours = responses_to_lm_response(&case["response"], "openai/gpt-5");
            ours.provider_response = None;
            ours.outputs
                .iter_mut()
                .for_each(|output| output.provider_output = None);
            assert_eq!(
                ours, expected,
                "{name}: our Responses reply diverges from dspy's"
            );
        }
    }

    /// A requested schema rides under `text.format` in the flat shape the Responses API takes — the
    /// object envelope by default, the strict named schema on [`JsonFormat::Schema`].
    #[test]
    fn a_requested_schema_rides_under_text_format() {
        let schema = json!({ "type": "object", "properties": { "answer": { "type": "string" } } });
        let call = crate::lm::api::interop::raise_request(
            "be helpful",
            &[crate::lm::ChatTurn::user("hi")],
            crate::lm::OutputMode::Json { schema: &schema },
            &crate::lm::Sampling::default(),
        );
        assert_eq!(
            request("gpt-5", &call, JsonFormat::Object).expect("builds")["text"],
            json!({ "format": { "type": "json_object" } })
        );
        assert_eq!(
            request("gpt-5", &call, JsonFormat::Schema).expect("builds")["text"],
            json!({ "format": { "type": "json_schema", "name": "response", "schema": schema, "strict": true } })
        );
    }

    fn request_fixture() -> Value {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/conformance/lm_api/openai_responses.json");
        serde_json::from_str(&std::fs::read_to_string(path).expect("fixture is readable"))
            .expect("fixture is valid json")
    }

    fn framed(event: Value) -> Framed {
        frame(&format!("data: {event}"), &mut StreamState::default())
    }

    /// The reply's items each stream their own delta — reasoning and text under their own part
    /// index, a tool call's arguments as a tool-call delta — for live display.
    #[test]
    fn each_output_item_streams_its_own_delta() {
        assert_eq!(
            framed(json!({ "type": "response.reasoning_summary_text.delta", "output_index": 0, "delta": "hmm" })).events,
            vec![LmStreamEvent::Delta { output_index: 0, part_index: 0, delta: LmDelta::thinking("hmm") }]
        );
        assert_eq!(
            framed(json!({ "type": "response.output_text.delta", "output_index": 1, "delta": "Sunny." })).events,
            vec![LmStreamEvent::Delta { output_index: 0, part_index: 1, delta: LmDelta::text("Sunny.") }]
        );
        assert_eq!(
            framed(json!({ "type": "response.function_call_arguments.delta", "output_index": 2, "delta": "{\"city\"" })).events,
            vec![LmStreamEvent::Delta {
                output_index: 0,
                part_index: 2,
                delta: LmDelta::ToolCallDelta { id: None, name: None, args_delta: Some("{\"city\"".to_owned()) },
            }]
        );
    }

    /// `response.completed` closes the stream with the whole reply, and it is exactly the reply the
    /// non-streamed parser builds from the same object — so a streamed answer matches a non-streamed.
    #[test]
    fn the_completed_frame_carries_the_authoritative_reply() {
        let completed = json!({
            "id": "resp_1", "model": "gpt-5",
            "output": [
                { "type": "reasoning", "summary": [{ "type": "summary_text", "text": "Let me think." }] },
                { "type": "message", "role": "assistant", "content": [{ "type": "output_text", "text": "It is sunny.", "annotations": [] }] },
            ],
            "usage": { "input_tokens": 10, "output_tokens": 5, "total_tokens": 15 },
        });
        let done = framed(json!({ "type": "response.completed", "response": completed }));
        assert!(done.done, "the completed frame closes the stream");

        let mut carried = *done.response.expect("the completed reply");
        let mut expected = responses_to_lm_response(&completed, "");
        for response in [&mut carried, &mut expected] {
            response.provider_response = None;
            response
                .outputs
                .iter_mut()
                .for_each(|output| output.provider_output = None);
        }
        assert_eq!(
            carried, expected,
            "the streamed reply equals the non-streamed one"
        );
    }
}

/// dspy 3.3.0's `tool_to_openai_responses`: the Responses API's function-tool shape.
///
/// Flat, where the chat dialect nests under `function` — `{type, name, parameters}` with the
/// optional fields beside them rather than inside. Until 3.3.0 both wires shared
/// `tool_to_openai`, so this wire sent the chat shape and OpenAI took it; the two renderers split
/// when `strict` arrived, and the provider extras land at the top level here for the same reason
/// they land under `function` there — each dialect puts them where it puts its function fields.
fn tool_item(tool: &api::LmToolSpec) -> Value {
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
fn apply_responses_tool_choice(body: &mut Value, choice: &api::LmToolChoice) -> Result<()> {
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
