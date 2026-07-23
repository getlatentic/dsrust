//! Reading an OpenAI chat-completion response into the typed [`LmResponse`](api::LmResponse), the way
//! dspy 3.3's `completion_to_lm_response` does: reasoning content as a thinking part first, then text,
//! then tool calls each carrying the raw call as provider data, then citations; usage with every
//! counter aliased; the response id, finish reason and cache flag kept. `tests/lm_api_conformance.rs`
//! holds this to dspy's own output, so a dropped reasoning part or a mis-aliased count is a failure.

use anyhow::{Result, anyhow};
use serde_json::Value;

use super::{LmUsage, api};
use crate::lm::api::Metadata;

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
    let response = completion_to_lm_response(body, model);
    if response.outputs.iter().all(|output| output.parts.is_empty()) {
        // A reply with neither text, a tool call, nor reasoning is nothing to parse — surfaced as the
        // error it is rather than an empty answer a caller would read as a blank completion.
        return Err(anyhow!("{label} returned no content"));
    }
    Ok(response)
}

/// dspy's `completion_to_lm_response`: the choices as outputs, usage with every counter, the id and
/// cache flag, and the model falling back to the request's when the response omits its own.
fn completion_to_lm_response(body: &Value, fallback_model: &str) -> api::LmResponse {
    let outputs = body["choices"]
        .as_array()
        .map(|choices| choices.iter().map(choice_to_lm_output).collect())
        .unwrap_or_default();
    api::LmResponse {
        model: Some(body["model"].as_str().unwrap_or(fallback_model).to_owned()),
        outputs,
        usage: usage(&body["usage"]),
        cache_hit: body["cache_hit"].as_bool().unwrap_or(false),
        response_id: body["id"].as_str().map(str::to_owned),
        provider_response: Some(body.clone()),
        ..api::LmResponse::default()
    }
}

/// dspy's `choice_to_lm_output`: reasoning first, then content, then tool calls, then citations, and
/// why generation stopped — `length` being this format's name for a reply cut off at the cap.
fn choice_to_lm_output(choice: &Value) -> api::LmOutput {
    let message = &choice["message"];
    let mut parts = Vec::new();
    if let Some(reasoning) = message["reasoning_content"].as_str().filter(|text| !text.is_empty()) {
        parts.push(api::LmPart::thinking(reasoning, false));
    }
    if let Some(content) = message["content"].as_str().filter(|text| !text.is_empty()) {
        parts.push(api::LmPart::text(content));
    }
    for call in message["tool_calls"].as_array().into_iter().flatten() {
        parts.push(tool_call(call));
    }
    parts.extend(citations(choice));
    let reason = choice["finish_reason"].as_str();
    api::LmOutput {
        parts,
        finish_reason: reason.map(str::to_owned),
        truncated: reason == Some("length"),
        logprobs: nonnull(&choice["logprobs"]),
        provider_output: Some(choice.clone()),
        ..api::LmOutput::default()
    }
}

/// dspy's `provider_tool_call_to_part`: the arguments parsed from their JSON string into an object,
/// the whole raw call kept as provider data, the id read from either spelling.
fn tool_call(call: &Value) -> api::LmPart {
    let function = &call["function"];
    let arguments = function["arguments"]
        .as_str()
        .or_else(|| call["arguments"].as_str())
        .unwrap_or("{}");
    let mut provider_data: Metadata = call.as_object().cloned().unwrap_or_default();
    let args = match serde_json::from_str::<Value>(arguments) {
        Ok(Value::Object(map)) => map,
        _ => {
            // dspy keeps the unparsed string beside the raw call and empties the args.
            provider_data.insert("raw_arguments".to_owned(), Value::String(arguments.to_owned()));
            Metadata::new()
        }
    };
    api::LmPart::ToolCall {
        id: call["call_id"]
            .as_str()
            .or_else(|| call["id"].as_str())
            .map(str::to_owned),
        name: function["name"]
            .as_str()
            .or_else(|| call["name"].as_str())
            .unwrap_or_default()
            .to_owned(),
        args,
        provider_data,
        metadata: Metadata::new(),
    }
}

/// dspy's `extract_citations_from_choice`: the litellm-populated `provider_specific_fields.citations`,
/// each entry a citation or a list of them, as [`Citation`](api::LmPart::Citation) parts.
fn citations(choice: &Value) -> Vec<api::LmPart> {
    let Some(list) = choice["message"]["provider_specific_fields"]["citations"].as_array() else {
        return Vec::new();
    };
    let mut parts = Vec::new();
    for item in list {
        match item.as_array() {
            Some(inner) => parts.extend(inner.iter().map(api::LmPart::citation)),
            None => parts.push(api::LmPart::citation(item)),
        }
    }
    parts
}

/// dspy's `usage_from_response`: every counter the provider sent, deserialized whole (unknown ones
/// kept), then the two spellings mirrored and the total derived. `None` when no usage was reported.
pub(super) fn usage(usage: &Value) -> Option<LmUsage> {
    if usage.is_null() {
        return None;
    }
    serde_json::from_value::<LmUsage>(usage.clone())
        .ok()
        .map(LmUsage::fill_aliases)
}

/// A JSON null reads as absent rather than as a present null, which is how an omitted `logprobs`
/// stays `None` instead of becoming `Some(Value::Null)`.
fn nonnull(value: &Value) -> Option<Value> {
    (!value.is_null()).then(|| value.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Faithfulness to dspy 3.3's response boundary: our `reply` parses each raw response into the
    /// same `LMResponse` dspy's `completion_to_lm_response` builds — reasoning part first, tool calls
    /// carrying the raw call, usage aliased, the id and finish reason kept. The fixture is generated
    /// by running dspy, and the compare is structural (parse dspy's dump, compare values) so pydantic's
    /// `"metadata": {}` convention is not a false divergence. The runtime-only `provider_response`/
    /// `provider_output` (which dspy excludes from its dump) are cleared before the compare.
    #[test]
    fn our_reply_matches_dspy_33_completion_to_lm_response() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/conformance/lm_api/openai_response.json");
        let fixture: Value =
            serde_json::from_str(&std::fs::read_to_string(path).expect("fixture is readable"))
                .expect("fixture is valid json");

        for case in fixture["cases"].as_array().expect("cases array") {
            let name = case["name"].as_str().expect("a case name");
            let expected: api::LmResponse = serde_json::from_value(case["lm_response"].clone())
                .unwrap_or_else(|error| panic!("{name}: dspy's LMResponse did not parse: {error}"));
            // The fixture's fallback model is the request's, "openai/gpt-4o".
            let mut ours = completion_to_lm_response(&case["response"], "openai/gpt-4o");
            ours.provider_response = None;
            ours.outputs.iter_mut().for_each(|output| output.provider_output = None);
            assert_eq!(
                ours, expected,
                "{name}: our reply diverges from dspy's completion_to_lm_response"
            );
        }
    }

    #[test]
    fn a_refused_call_carries_the_status_and_the_services_own_message() {
        let body = serde_json::json!({ "error": { "message": "Incorrect API key provided" } });
        let error = reply("openai", "gpt-4o-mini", reqwest::StatusCode::UNAUTHORIZED, &body)
            .expect_err("401 is a failure");
        assert!(error.to_string().contains("openai 401"), "got: {error}");
        assert!(error.to_string().contains("Incorrect API key provided"), "got: {error}");
    }

    #[test]
    fn a_success_carrying_no_content_is_an_error_rather_than_an_empty_reply() {
        let error = reply(
            "openai",
            "gpt-4o-mini",
            reqwest::StatusCode::OK,
            &serde_json::json!({ "choices": [] }),
        )
        .expect_err("nothing to read");
        assert!(error.to_string().contains("openai returned no content"), "got: {error}");
    }
}
