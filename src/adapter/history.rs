//! dspy `format_conversation_history`: a prior conversation reaches the model as the turns it
//! actually was, not as a field value.
//!
//! A `History` input holds the exchanges that came before. Rendering it into a marker section
//! would show the model a transcript inside a single request; dspy instead deletes the field
//! from the signature and replays each exchange as its own user and assistant turn, so the
//! conversation reads to the model as a conversation.

use serde_json::Value;

use crate::example::Example;
use crate::lm::{ChatTurn, Content, LmPart, Role};
use crate::signature::Signature;

use super::exchange::{Style, ask};
use super::python_json::json_dumps;
use super::types::ToolCalls;

/// The annotation dspy prints for a history field, and so the name this crate recognises it by.
const ANNOTATION: &str = "History";

/// dspy's `missing_field_message` for history, standing in for a field an exchange never had.
/// The trailing space is upstream's, and survives because the block is stripped as a whole.
const NOT_SUPPLIED: &str = "Not supplied for this conversation history message. ";

/// dspy renders a round of tool results through a one-field signature named for them, so the
/// section a replayed conversation carries is this.
const TOOL_CALL_RESULTS: &str = "tool_call_results";

/// The name of the signature's history input, if it declares one.
///
/// dspy matches on the field's annotation being `History` itself, so a signature carries at most
/// one and the first wins.
pub(super) fn field_name(signature: &Signature) -> Option<&str> {
    signature
        .inputs
        .iter()
        .find(|field| field.kind.annotation() == ANNOTATION)
        .map(|field| field.name.as_str())
}


/// The exchanges a history value carries, in order.
///
/// dspy's `History.messages` is a list of dicts keyed by the signature's own field names, so an
/// exchange is an [`Example`] in every respect but the name. A value that is not a list of
/// objects contributes nothing rather than erroring: it reached here as a model input, and a
/// malformed one should not take down a request that has everything else it needs.
fn exchanges(value: &Value) -> Vec<Example> {
    value
        .get("messages")
        .and_then(Value::as_array)
        .map(|messages| {
            messages
                .iter()
                .filter_map(Value::as_object)
                .map(|fields| {
                    Example::new(fields.iter().map(|(name, value)| (name, value.clone())))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// The replayed turns, a user and an assistant turn per exchange.
///
/// `stripped` is the signature without the history field, as [`Signature::delete`] leaves it.
pub(super) fn turns(
    stripped: &Signature,
    value: &Value,
    style: Style,
    native: bool,
) -> Vec<ChatTurn> {
    exchanges(value)
        .iter()
        .flat_map(|exchange| replay(stripped, exchange, style, native))
        .collect()
}

/// dspy `format_conversation_history` for one exchange, without native function calling.
///
/// An exchange that called tools splits into three: what was asked, the calls the model made, and
/// what those calls returned. The assistant turn states the calls *without* their results — the
/// model produced the calls, not the answers — and the results follow as their own user turn, the
/// way a tool's output arrives in a conversation. A turn whose content comes out empty is left
/// out entirely rather than sent blank.
fn replay(stripped: &Signature, exchange: &Example, style: Style, native: bool) -> Vec<ChatTurn> {
    let mut turns = Vec::new();
    push_non_empty(&mut turns, ask(stripped, exchange, None, style));

    let called = tool_calls_of(exchange);
    if native && let Some((name, calls)) = &called {
        native_replay(&mut turns, stripped, exchange, style, name, calls);
        return turns;
    }
    let answered = called.as_ref().and_then(|(name, calls)| {
        calls.tool_call_results.as_ref().map(|results| (name.as_str(), results.clone()))
    });
    // The calls replay without their results, so the assistant turn reads as the model wrote it.
    let mut stated = exchange.clone();
    if let (Some((name, calls)), Some(_)) = (&called, &answered)
        && let Ok(without) = serde_json::to_value(calls.without_results())
    {
        stated.set(name.clone(), without);
    }
    push_non_empty(&mut turns, (style.answer)(stripped, &stated, Some(NOT_SUPPLIED)));

    if let Some((_, results)) = answered {
        let rendered = json_dumps(&serde_json::to_value(&results).unwrap_or(Value::Null));
        turns.push(ChatTurn::user((style.wrap)(TOOL_CALL_RESULTS, &rendered)));
    }
    turns
}

/// dspy `format_conversation_history`'s native branch: once the provider calls tools itself, the
/// calls travel beside the assistant's content rather than inside it, and each result comes back
/// as its own `tool` message naming the call it answers.
///
/// The assistant's content is rendered from the outputs that are *not* the calls and that this
/// exchange actually recorded, so a turn that was only a tool call carries no content at all.
fn native_replay(
    turns: &mut Vec<ChatTurn>,
    stripped: &Signature,
    exchange: &Example,
    style: Style,
    field: &str,
    calls: &ToolCalls,
) {
    let mut spoken = stripped.clone();
    spoken
        .outputs
        .retain(|output| output.name != field && exchange.get(&output.name).is_some());
    let content = match spoken.outputs.is_empty() {
        true => String::new(),
        false => (style.answer)(&spoken, exchange, None)
            .content
            .text()
            .unwrap_or_default()
            .to_owned(),
    };

    // Results only replay when they answer exactly the calls made: a provider pairs each `tool`
    // message to a call by id, so anything else would attribute a result to the wrong call.
    let answered = calls.results_match_calls();
    if content.is_empty() && !answered {
        return;
    }

    let mut parts: Vec<LmPart> = Vec::new();
    if !content.is_empty() {
        parts.push(LmPart::text(content));
    }
    if answered {
        parts.extend(calls.tool_calls.iter().map(|call| LmPart::ToolCall {
            id: call.id.clone(),
            name: call.name.clone(),
            args: call.args.clone(),
            provider_data: Default::default(),
            metadata: Default::default(),
        }));
    }
    turns.push(ChatTurn {
        role: Role::Assistant,
        content: Content::Parts(parts),
    });

    if !answered {
        return;
    }
    for result in &calls.tool_call_results.iter().flat_map(|r| r.tool_call_results.clone()).collect::<Vec<_>>() {
        turns.push(ChatTurn {
            role: Role::Tool,
            content: Content::Parts(vec![LmPart::ToolResult {
                call_id: result.call_id.clone(),
                name: Some(result.name.clone()),
                content: vec![LmPart::text(tool_result_content(&result.value))],
                is_error: result.is_error,
                provider_data: Default::default(),
                metadata: Default::default(),
            }]),
        });
    }
}

/// dspy `_tool_result_content`: a string result is its own text; anything else is written as JSON.
fn tool_result_content(value: &Value) -> String {
    match value.as_str() {
        Some(text) => text.to_owned(),
        None => json_dumps(value),
    }
}

/// dspy appends a replayed turn only when it has content: an exchange that recorded no inputs, or
/// whose outputs were all tool calls, would otherwise send a blank message.
fn push_non_empty(turns: &mut Vec<ChatTurn>, turn: ChatTurn) {
    if !turn.content.text().unwrap_or_default().trim().is_empty() {
        turns.push(turn);
    }
}

/// dspy `_tool_calls_from_message`: the first field of the exchange whose value carries tool
/// calls. Upstream asks the *value*, not the signature — a mapping with a `tool_calls` key is one
/// — so a replayed exchange is read the same way whether it came from a typed field or a plain
/// dict, and a value that does not parse contributes nothing rather than failing the request.
fn tool_calls_of(exchange: &Example) -> Option<(String, ToolCalls)> {
    exchange.fields().find_map(|(name, value)| {
        let carries_calls = value.get("tool_calls").is_some();
        let calls = carries_calls.then(|| serde_json::from_value(value.clone()).ok())??;
        Some((name.to_owned(), calls))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signature::{FieldKind, InField, JsonType, OutField};
    use serde_json::json;

    fn signature() -> Signature {
        let mut signature = Signature::single_input(
            "Answer the question.",
            vec![OutField {
                name: "answer".into(),
                ..Default::default()
            }],
        );
        signature.inputs = vec![
            InField {
                name: "question".into(),
                ..Default::default()
            },
            InField {
                name: "history".into(),
                kind: FieldKind::Json(JsonType::plain(ANNOTATION)),
                ..Default::default()
            },
        ];
        signature
    }

    fn history() -> Value {
        json!({
            "messages": [
                { "question": "What is the capital of France?", "answer": "Paris" },
                { "question": "What is the capital of Germany?", "answer": "Berlin" },
            ]
        })
    }

    #[test]
    fn the_history_field_is_found_by_its_annotation() {
        assert_eq!(field_name(&signature()), Some("history"));
    }

    #[test]
    fn a_signature_without_one_reports_none() {
        let mut plain = signature();
        plain.inputs.retain(|field| field.name != "history");
        assert_eq!(field_name(&plain), None);
    }

    #[test]
    fn stripping_leaves_every_other_input_in_order() {
        let stripped = signature().delete("history");
        let names: Vec<&str> = stripped.inputs.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(names, ["question"]);
    }

    #[test]
    fn each_exchange_becomes_a_user_and_an_assistant_turn() {
        let stripped = signature().delete("history");
        let turns = turns(&stripped, &history(), crate::adapter::chat::MARKER_STYLE, false);

        assert_eq!(turns.len(), 4);
        assert_eq!(
            turns[0].content.text().unwrap(),
            "[[ ## question ## ]]\nWhat is the capital of France?"
        );
        assert_eq!(
            turns[1].content.text().unwrap(),
            "[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]\n"
        );
        assert_eq!(
            turns[2].content.text().unwrap(),
            "[[ ## question ## ]]\nWhat is the capital of Germany?"
        );
        assert_eq!(
            turns[3].content.text().unwrap(),
            "[[ ## answer ## ]]\nBerlin\n\n[[ ## completed ## ]]\n"
        );
    }

    #[test]
    fn the_history_field_itself_never_reaches_a_turn() {
        // Rendering it would show the model a transcript inside one request, which is the thing
        // replaying the exchanges exists to avoid.
        let stripped = signature().delete("history");
        let turns = turns(&stripped, &history(), crate::adapter::chat::MARKER_STYLE, false);
        assert!(
            !turns
                .iter()
                .any(|turn| turn.content.text().unwrap().contains("history"))
        );
    }

    #[test]
    fn an_exchange_missing_an_output_says_so_rather_than_going_blank() {
        let stripped = signature().delete("history");
        let turns = turns(
            &stripped,
            &json!({ "messages": [{ "question": "Where?" }] }),
            crate::adapter::chat::MARKER_STYLE,
            false,
        );
        assert_eq!(
            turns[1].content.text().unwrap(),
            "[[ ## answer ## ]]\nNot supplied for this conversation history message.\n\n[[ ## completed ## ]]\n"
        );
    }

    #[test]
    fn a_history_with_no_messages_contributes_no_turns() {
        let stripped = signature().delete("history");
        assert!(
            turns(
                &stripped,
                &json!({ "messages": [] }),
                crate::adapter::chat::MARKER_STYLE,
                false
            )
            .is_empty()
        );
        assert!(turns(&stripped, &json!({}), crate::adapter::chat::MARKER_STYLE, false).is_empty());
    }
}
