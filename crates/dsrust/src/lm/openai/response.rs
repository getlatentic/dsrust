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
    let response = completion_to_lm_response(body, model);
    if response
        .outputs
        .iter()
        .all(|output| output.parts.is_empty())
    {
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
    if let Some(reasoning) = message["reasoning_content"]
        .as_str()
        .filter(|text| !text.is_empty())
    {
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
        // dspy keeps the unparsed string beside the raw call, says why it could not read it, and
        // empties the args — so a caller can tell "the model called this with nothing" from "the
        // model called this and we could not read what with".
        parsed => super::unreadable_arguments(&mut provider_data, arguments, parsed.err()),
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

    use serde_json::json;

    /// The raw body rides along at both levels — dspy keeps the provider response for callers
    /// that read fields the typed model does not carry, and deleting either field lost it
    /// silently because nothing read it back.
    #[test]
    fn the_provider_body_rides_along_at_both_levels() {
        let body = json!({
            "id": "chatcmpl-1",
            "model": "gpt-4o-mini",
            "choices": [{ "message": { "content": "hi" }, "finish_reason": "stop" }],
        });
        let response = completion_to_lm_response(&body, "fallback");
        assert_eq!(response.provider_response, Some(body.clone()));
        assert_eq!(
            response.outputs[0].provider_output,
            Some(body["choices"][0].clone())
        );
    }

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
            ours.outputs
                .iter_mut()
                .for_each(|output| output.provider_output = None);
            assert_eq!(
                ours, expected,
                "{name}: our reply diverges from dspy's completion_to_lm_response"
            );
        }
    }

    #[test]
    fn a_refused_call_carries_the_status_and_the_services_own_message() {
        let body = serde_json::json!({ "error": { "message": "Incorrect API key provided" } });
        let error = reply(
            "openai",
            "gpt-4o-mini",
            reqwest::StatusCode::UNAUTHORIZED,
            &reqwest::header::HeaderMap::new(),
            &body,
        )
        .expect_err("401 is a failure");
        // dspy 3.3 normalizes an LM failure: `[model] message`, with everything else typed
        // beside it rather than spelled into the text.
        assert_eq!(
            error.to_string(),
            "[gpt-4o-mini] Incorrect API key provided"
        );
        let failed = error
            .downcast_ref::<crate::lm::LmFailure>()
            .expect("a typed LM failure");
        assert_eq!(failed.kind, crate::lm::LmErrorKind::Auth);
        assert_eq!(failed.status, Some(401));
        assert_eq!(failed.provider.as_deref(), Some("openai"));
        assert!(
            !failed.is_retryable(),
            "a rejected key fails the same way twice"
        );
    }

    /// A refusal's headers reach the failure, which is what lets the retry wait for what the
    /// provider asked for rather than guessing — dspy's `_exception_retry_after` and
    /// `_exception_request_id`, from the response instead of from a litellm exception.
    #[test]
    fn a_refusals_headers_reach_the_failure() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("retry-after", "30".parse().expect("a header value"));
        headers.insert(
            "x-request-id",
            "req_abc123".parse().expect("a header value"),
        );

        let error = reply(
            "openai",
            "gpt-4o-mini",
            reqwest::StatusCode::TOO_MANY_REQUESTS,
            &headers,
            &serde_json::json!({ "error": { "message": "slow down" } }),
        )
        .expect_err("429 is a failure");
        let failed = error
            .downcast_ref::<crate::lm::LmFailure>()
            .expect("a typed LM failure");
        assert_eq!(failed.retry_after, Some(30.0));
        assert_eq!(failed.request_id.as_deref(), Some("req_abc123"));
        assert!(failed.is_retryable());
    }

    #[test]
    fn a_success_carrying_no_content_is_an_error_rather_than_an_empty_reply() {
        let error = reply(
            "openai",
            "gpt-4o-mini",
            reqwest::StatusCode::OK,
            &reqwest::header::HeaderMap::new(),
            &serde_json::json!({ "choices": [] }),
        )
        .expect_err("nothing to read");
        assert!(
            error.to_string().contains("openai returned no content"),
            "got: {error}"
        );
    }
}
