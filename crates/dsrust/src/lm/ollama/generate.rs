//! The ollama `/api/generate` route — litellm's `ollama/` provider, distinct from `ollama_chat/`.
//!
//! This route takes one prompt string rather than a message list, so a conversation is flattened
//! into it. litellm does that flattening with `ollama_pt`, and this reproduces it byte for byte:
//! consecutive user turns merged under `### User:`, system under `### System:`, assistant under
//! `### Assistant:`, an assistant turn's tool calls appended as `Tool Calls: {json}`. `images`
//! ride beside the prompt, sampling under `options`, a schema as `format`.
//!
//! There is no native tool field on this route — litellm renders tools into the prompt instead —
//! so a model reached this way reports no function-calling capability, and tools arrive already
//! rendered by the adapter, folded into the prompt like any other content. `ollama_chat/` is the
//! route to reach for native calls.

use std::future::Future;
use std::time::Duration;

use anyhow::{Context, Result, anyhow};

use super::refusal;
use futures_util::Stream;
use serde_json::{Map, Value, json};

use super::{authorized, provider_data, usage};
use crate::lm::ChatModel;
use crate::lm::api::{self, LmConfig, LmDelta, LmStreamEvent};
use crate::lm::streaming::{Framed, Framing, StreamState};

/// An ollama server reached over `/api/generate`, the model and host beside it.
pub(crate) struct Generate<'a> {
    pub model: &'a str,
    pub host: &'a str,
    pub api_key: Option<&'a str>,
    pub timeout: Duration,
}

impl ChatModel for Generate<'_> {
    fn forward<'a>(
        &'a self,
        call: &'a api::LmRequest,
    ) -> impl Future<Output = Result<api::LmResponse>> + Send + 'a {
        async move {
            let http = &crate::lm::global::client();
            let request = http
                .post(format!("{}/api/generate", self.host))
                .timeout(self.timeout)
                .json(&request(self.model, call));
            let response = authorized(request, self.api_key)
                .send()
                .await
                .map_err(|error| {
                    crate::lm::LmFailure::from_transport(&error, self.model, "ollama")
                })?;
            let status = response.status();
            let headers = response.headers().clone();
            let body: Value = response
                .json()
                .await
                .context("ollama response was not JSON")?;
            if !status.is_success() {
                if let Some(too_long) =
                    crate::lm::ContextWindowExceeded::detected(self.model, &body)
                {
                    return Err(too_long.into());
                }
                return Err(
                    crate::lm::LmFailure::from_status(status.as_u16(), refusal(&body))
                        .on_model(self.model)
                        .from_provider("ollama")
                        .headers(&headers)
                        .into(),
                );
            }
            reply(self.model, &body)
        }
    }
}

/// The streaming form: the same body with `stream` set, ollama's line-delimited JSON read back.
/// Each line carries a piece of `response`; the last carries `done` and the counts.
pub(crate) fn stream<'h>(
    http: &'h reqwest::Client,
    model: &str,
    host: &str,
    api_key: Option<&str>,
    timeout: Duration,
    call: &api::LmRequest,
) -> impl Stream<Item = Result<api::LmStreamEvent>> + Send + use<'h> {
    let mut body = request(model, call);
    body["stream"] = json!(true);
    let connect = authorized(
        http.post(format!("{host}/api/generate"))
            .timeout(timeout)
            .json(&body),
        api_key,
    )
    .send();
    crate::lm::streaming::events(
        connect,
        "ollama".to_owned(),
        model.to_owned(),
        Framing {
            separator: b"\n",
            frame,
        },
    )
}

/// The `/api/generate` body for one call: the conversation as a single prompt, the images pulled
/// beside it, sampling under `options`, a schema as `format`.
pub(super) fn request(model: &str, call: &api::LmRequest) -> Value {
    let Prompt { text, images } = flatten(&call.wire_messages());
    let mut body = json!({
        "model": model,
        "prompt": text,
        "options": options(&call.config),
        "stream": false,
        "images": images,
    });
    if let Some(schema) = call.output_schema() {
        body["format"] = schema.clone();
    }
    body
}

/// A conversation reduced to one prompt string and the images that rode alongside it.
struct Prompt {
    text: String,
    images: Vec<String>,
}

/// litellm's `ollama_pt`: the message list as one prompt. The three roles are merged in the fixed
/// order user, system, assistant within each pass over the list, and the pass repeats until the
/// list is spent — so a system turn ahead of a user turn still reads system-first, one pass later.
fn flatten(messages: &[Value]) -> Prompt {
    let mut text = String::new();
    let mut images = Vec::new();
    let mut i = 0;
    // A `for` over a range that cannot grow: each round consumes at least one message (the three
    // merges, or the stall guard for a role none of them place), so `len` rounds always suffice —
    // and a round that consumes nothing runs out of rounds instead of spinning. Three mutants of
    // the cursor arithmetic hung for the full timeout under the `while` this replaces; under a
    // bounded loop the same mutants fall out early and the assert below names them.
    for _round in 0..messages.len() {
        if i >= messages.len() {
            break;
        }
        let start = i;
        let user = merge(messages, &mut i, is_user, |message| {
            content_and_images(message, &mut images)
        });
        push_section(&mut text, "User", &user);
        let system = merge(messages, &mut i, is_system, content_str);
        push_section(&mut text, "System", &system);
        let assistant = merge(messages, &mut i, is_assistant, assistant_str);
        push_section(&mut text, "Assistant", &assistant);
        // ollama_pt raises on a role it cannot place rather than loop forever; a stalled cursor
        // here means a message that is none of the three, which the adapters do not produce.
        if i == start {
            i += 1;
        }
    }
    assert!(
        i >= messages.len(),
        "the prompt walk stalled at {i} of {} — a round consumed nothing",
        messages.len()
    );
    Prompt { text, images }
}

/// litellm folds a tool result into the same run as a user turn, so all three read as "user".
fn is_user(role: &str) -> bool {
    matches!(role, "user" | "tool" | "function")
}

fn is_system(role: &str) -> bool {
    role == "system"
}

fn is_assistant(role: &str) -> bool {
    role == "assistant"
}

/// Concatenate the run of consecutive messages the predicate accepts, advancing the cursor past
/// them and rendering each with `render`.
fn merge(
    messages: &[Value],
    i: &mut usize,
    accepts: fn(&str) -> bool,
    mut render: impl FnMut(&Value) -> String,
) -> String {
    // Progress is structural: the run is measured off the slice and the cursor advances by its
    // length, so there is no per-iteration `+= 1` for a mutant to break into a spin. The old shape
    // hung for the whole timeout under exactly that mutation, which is detection only in the sense
    // that the suite never finished.
    let run = messages[*i..]
        .iter()
        .take_while(|message| accepts(message["role"].as_str().unwrap_or_default()))
        .count();
    let merged = messages[*i..*i + run].iter().map(&mut render).collect();
    *i += run;
    merged
}

/// A section header and its content, added only when the content is non-empty — litellm skips an
/// empty run rather than emitting a bare header.
fn push_section(prompt: &mut String, role: &str, content: &str) {
    if !content.is_empty() {
        prompt.push_str(&format!("### {role}:\n{content}\n\n"));
    }
}

/// A user/tool turn's text, its images drained into `images`. litellm pulls an `image_url` block
/// out to the top-level list and keeps only the text in the prompt.
fn content_and_images(message: &Value, images: &mut Vec<String>) -> String {
    match &message["content"] {
        Value::String(text) => text.clone(),
        Value::Array(blocks) => {
            let mut text = String::new();
            for block in blocks {
                match block["type"].as_str() {
                    Some("text") => text.push_str(block["text"].as_str().unwrap_or_default()),
                    Some("image_url") => images.push(image_url(&block["image_url"])),
                    _ => {}
                }
            }
            text
        }
        _ => String::new(),
    }
}

/// A message's content as a string, concatenating the text of a block list — litellm's
/// `convert_content_list_to_str`.
fn content_str(message: &Value) -> String {
    match &message["content"] {
        Value::String(text) => text.clone(),
        Value::Array(blocks) => blocks
            .iter()
            .filter_map(|block| block["text"].as_str())
            .collect(),
        _ => String::new(),
    }
}

/// An assistant turn's content, with any tool calls appended the way litellm does: the calls as a
/// list of `{id, type, function:{name, arguments}}`, pretty-printed under a `Tool Calls:` label.
fn assistant_str(message: &Value) -> String {
    let mut text = content_str(message);
    let Some(calls) = message["tool_calls"]
        .as_array()
        .filter(|calls| !calls.is_empty())
    else {
        return text;
    };
    let rendered = Value::Array(calls.iter().map(ollama_tool_call).collect());
    // litellm's `json.dumps(..., indent=2)`; serde's pretty printer is the same two-space,
    // `": "`-separated form, byte for byte.
    text.push_str(&format!(
        "Tool Calls: {}",
        serde_json::to_string_pretty(&rendered).unwrap_or_default()
    ));
    text
}

/// One assistant `tool_calls` entry in the shape litellm builds before printing it: the id and
/// type kept, the arguments parsed back from the JSON string OpenAI carries them as.
fn ollama_tool_call(call: &Value) -> Value {
    json!({
        "id": call["id"],
        "type": "function",
        "function": {
            "name": call["function"]["name"],
            "arguments": call["function"]["arguments"]
                .as_str()
                .and_then(|text| serde_json::from_str::<Value>(text).ok())
                .unwrap_or(Value::Null),
        },
    })
}

/// The base64 ollama takes: the payload of a `data:` URI, or the reference itself when it is not
/// one — the same reading the chat route does.
fn image_url(image_url: &Value) -> String {
    let url = image_url["url"].as_str().unwrap_or_default();
    url.split_once(";base64,")
        .map_or(url, |(_, data)| data)
        .to_owned()
}

/// LmConfig under `options`, only the keys the caller set — litellm defaults none, and the cap is
/// `num_predict`, exactly as on the chat route.
fn options(config: &LmConfig) -> Value {
    let mut options = Map::new();
    if let Some(temperature) = config.temperature {
        options.insert("temperature".to_owned(), json!(temperature));
    }
    if let Some(stop) = &config.stop
        && !stop.is_empty()
    {
        options.insert("stop".to_owned(), json!(stop));
    }
    if let Some(max_tokens) = config.max_tokens {
        options.insert("num_predict".to_owned(), json!(max_tokens));
    }
    Value::Object(options)
}

/// The `/api/generate` reply: its `response` field is the whole answer, with the counts and stop
/// reason beside it the way the chat route carries them.
fn reply(model: &str, body: &Value) -> Result<api::LmResponse> {
    let content = body["response"].as_str().unwrap_or_default();
    if content.is_empty() {
        return Err(anyhow!("ollama returned no content"));
    }
    let reason = body["done_reason"].as_str();
    let output = api::LmOutput {
        parts: vec![api::LmPart::text(content)],
        truncated: reason == Some("length"),
        finish_reason: reason.map(str::to_owned),
        ..api::LmOutput::default()
    };
    Ok(api::LmResponse {
        outputs: vec![output],
        ..api::LmResponse::default()
    }
    .usage(usage(body))
    .provider_response(provider_data(body))
    .model(model))
}

/// One `/api/generate` line as its events: a piece of `response` as a text delta, the `done` line
/// closing with the counts and stop reason.
fn frame(line: &str, state: &mut StreamState) -> Framed {
    let Ok(chunk) = serde_json::from_str::<Value>(line.trim()) else {
        return Framed::of(Vec::new());
    };
    let mut events = Vec::new();
    if let Some(content) = chunk["response"].as_str().filter(|text| !text.is_empty()) {
        events.push(LmStreamEvent::Delta {
            output_index: 0,
            part_index: 0,
            delta: LmDelta::text(content),
        });
    }
    if chunk["done"].as_bool() != Some(true) {
        return Framed::of(events);
    }
    if let Some(reported) = usage(&chunk) {
        state.usage = Some(reported);
    }
    if let Some(reason) = chunk["done_reason"].as_str() {
        events.push(LmStreamEvent::OutputEnd {
            output_index: 0,
            finish_reason: Some(reason.to_owned()),
            truncated: reason == "length",
        });
    }
    Framed::closing(events)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A role none of the three merges accept must be *skipped*, not spun on. The stall guard is
    /// the loop's whole progress for that case, and two arithmetic mutants of it survived because
    /// no test ever presented an unplaceable role.
    #[test]
    fn an_unknown_role_is_skipped_rather_than_looped_on() {
        let prompt = flatten(&[
            json!({ "role": "user", "content": "first" }),
            json!({ "role": "developer", "content": "unplaceable" }),
            json!({ "role": "user", "content": "second" }),
        ]);
        assert!(prompt.text.contains("first"));
        assert!(prompt.text.contains("second"), "the walk continued past it");
        assert!(!prompt.text.contains("unplaceable"));
    }

    /// litellm's `convert_content_list_to_str`: a block-list content flattens to its text runs.
    ///
    /// On a **system** turn, deliberately: the user path reads blocks through
    /// `content_and_images`, so a user-turn version of this test left `content_str`'s Array arm
    /// unpinned while looking like it covered it — the mutant survived the first version of this
    /// very test.
    #[test]
    fn block_list_content_flattens_to_its_text() {
        let prompt = flatten(&[
            json!({
                "role": "system",
                "content": [
                    { "type": "text", "text": "look at " },
                    { "type": "text", "text": "this" },
                ],
            }),
            json!({ "role": "user", "content": "ok" }),
        ]);
        assert!(prompt.text.contains("look at this"), "{}", prompt.text);
    }

    /// The generate reply carries why generation stopped, and `length` means cut off.
    #[test]
    fn the_reply_keeps_done_reason_and_length_means_truncated() {
        let cut =
            reply("m", &json!({ "response": "part", "done_reason": "length" })).expect("content");
        assert!(cut.outputs[0].truncated);
        assert_eq!(cut.outputs[0].finish_reason.as_deref(), Some("length"));
        let done =
            reply("m", &json!({ "response": "whole", "done_reason": "stop" })).expect("content");
        assert!(!done.outputs[0].truncated);
        assert_eq!(done.outputs[0].finish_reason.as_deref(), Some("stop"));
    }

    /// The closing line: an empty `response` adds no empty delta, a non-done line does not close,
    /// and the reason's truncation reads both ways.
    #[test]
    fn the_done_line_closes_with_the_reason_and_adds_no_empty_delta() {
        let mut state = StreamState::default();
        let open = frame(r#"{"response": "hi", "done": false}"#, &mut state);
        assert!(!open.done, "not closed yet");
        assert_eq!(open.events.len(), 1);

        let mut state = StreamState::default();
        let cut = frame(
            r#"{"response": "", "done": true, "done_reason": "length"}"#,
            &mut state,
        );
        assert!(cut.done);
        let [
            LmStreamEvent::OutputEnd {
                finish_reason,
                truncated,
                ..
            },
        ] = &cut.events[..]
        else {
            panic!("an empty response adds no delta: {:?}", cut.events)
        };
        assert_eq!(finish_reason.as_deref(), Some("length"));
        assert!(truncated);

        let mut state = StreamState::default();
        let done = frame(
            r#"{"response": "", "done": true, "done_reason": "stop"}"#,
            &mut state,
        );
        let [LmStreamEvent::OutputEnd { truncated, .. }] = &done.events[..] else {
            panic!("one closing event: {:?}", done.events)
        };
        assert!(!truncated);
    }

    /// Faithfulness to litellm's `ollama/` path: our body equals the one litellm puts on the wire
    /// for the same typed request — the conversation flattened by `ollama_pt`, `options` with
    /// nothing defaulted. Every expectation is litellm's captured output.
    #[test]
    fn our_body_matches_litellm_for_ollama_generate() {
        crate::lm::tests::each_case("ollama_generate", |model, call| request(model, call));
    }

    /// The reply's `response` field is the whole answer, the counts and stop reason beside it.
    #[test]
    fn the_response_field_is_the_reply() {
        let body = json!({
            "response": "the answer",
            "prompt_eval_count": 12,
            "eval_count": 5,
            "done_reason": "stop",
        });
        let answered = reply("llama3.2", &body).expect("a reply");
        assert_eq!(answered.first_text(), "the answer");
        let usage = answered.usage.expect("counts");
        assert_eq!(usage.input_tokens, Some(12));
        assert_eq!(usage.output_tokens, Some(5));
    }

    #[test]
    fn a_reply_with_an_empty_response_is_an_error() {
        assert!(reply("llama3.2", &json!({ "response": "" })).is_err());
    }
}
