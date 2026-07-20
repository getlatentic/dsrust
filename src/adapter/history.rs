//! dspy `format_conversation_history`: a prior conversation reaches the model as the turns it
//! actually was, not as a field value.
//!
//! A `History` input holds the exchanges that came before. Rendering it into a marker section
//! would show the model a transcript inside a single request; dspy instead deletes the field
//! from the signature and replays each exchange as its own user and assistant turn, so the
//! conversation reads to the model as a conversation.

use serde_json::Value;

use crate::example::Example;
use crate::lm::ChatTurn;
use crate::signature::Signature;

use super::exchange::{Style, ask};

/// The annotation dspy prints for a history field, and so the name this crate recognises it by.
const ANNOTATION: &str = "History";

/// dspy's `missing_field_message` for history, standing in for a field an exchange never had.
/// The trailing space is upstream's, and survives because the block is stripped as a whole.
const NOT_SUPPLIED: &str = "Not supplied for this conversation history message. ";

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

/// The signature the turns are rendered against: the caller's, without the history field.
///
/// Every turn dspy builds for a history — the replayed exchanges and the live request alike —
/// is rendered from this, which is why the history field appears in none of them. The system
/// message is built from the original signature and does still announce the field.
pub(super) fn without_field(signature: &Signature, name: &str) -> Signature {
    let mut stripped = signature.clone();
    stripped.inputs.retain(|field| field.name != name);
    stripped
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
/// `stripped` is the signature without the history field, as [`without_field`] returns it.
pub(super) fn turns(stripped: &Signature, value: &Value, style: Style) -> Vec<ChatTurn> {
    exchanges(value)
        .iter()
        .flat_map(|exchange| {
            [
                ask(stripped, exchange, None, style),
                (style.answer)(stripped, exchange, Some(NOT_SUPPLIED)),
            ]
        })
        .collect()
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
                desc: String::new(),
                kind: FieldKind::Str,
                values: None,
                schema: None,
            }],
        );
        signature.inputs = vec![
            InField {
                name: "question".into(),
                desc: String::new(),
                kind: FieldKind::Str,
                values: None,
            },
            InField {
                name: "history".into(),
                desc: String::new(),
                kind: FieldKind::Json(JsonType::plain(ANNOTATION)),
                values: None,
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
        let stripped = without_field(&signature(), "history");
        let names: Vec<&str> = stripped.inputs.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(names, ["question"]);
    }

    #[test]
    fn each_exchange_becomes_a_user_and_an_assistant_turn() {
        let stripped = without_field(&signature(), "history");
        let turns = turns(&stripped, &history(), crate::adapter::MARKER_STYLE);

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
        let stripped = without_field(&signature(), "history");
        let turns = turns(&stripped, &history(), crate::adapter::MARKER_STYLE);
        assert!(
            !turns
                .iter()
                .any(|turn| turn.content.text().unwrap().contains("history"))
        );
    }

    #[test]
    fn an_exchange_missing_an_output_says_so_rather_than_going_blank() {
        let stripped = without_field(&signature(), "history");
        let turns = turns(
            &stripped,
            &json!({ "messages": [{ "question": "Where?" }] }),
            crate::adapter::MARKER_STYLE,
        );
        assert_eq!(
            turns[1].content.text().unwrap(),
            "[[ ## answer ## ]]\nNot supplied for this conversation history message.\n\n[[ ## completed ## ]]\n"
        );
    }

    #[test]
    fn a_history_with_no_messages_contributes_no_turns() {
        let stripped = without_field(&signature(), "history");
        assert!(
            turns(
                &stripped,
                &json!({ "messages": [] }),
                crate::adapter::MARKER_STYLE
            )
            .is_empty()
        );
        assert!(turns(&stripped, &json!({}), crate::adapter::MARKER_STYLE).is_empty());
    }
}
