//! dspy `adapters/types/history.py`: the `History` type a signature carries as an input.
//!
//! A `History` holds the exchanges that came before, each a mapping keyed by the signature's own
//! field names. The [formatter](crate::adapter::history) is what turns those exchanges into the
//! user and assistant turns a model reads; this is the value the caller constructs and a module
//! holds, mutates, and hands back.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// dspy's `History`: the conversation so far, as a list of messages keyed by a signature's fields.
///
/// Upstream's model is `frozen`, which stops the field from being *reassigned* but not the list
/// from being appended to — `history.messages.append(event)` is how an agent loop records a turn,
/// and that mutates the list in place. Exposing `messages` gives the same reach; the frozen-ness
/// upstream relies on was never about the list's contents.
///
/// `extra="forbid"` upstream is [`deny_unknown_fields`](serde) here: a serialized history states
/// exactly `messages` and nothing else, so a dict carrying a stray key is refused rather than
/// silently dropped. `messages` is required, matching that it has no default upstream — validating
/// a mapping without it is an error, not an empty history.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct History {
    pub messages: Vec<Map<String, Value>>,
}

impl History {
    pub fn new(messages: Vec<Map<String, Value>>) -> Self {
        Self { messages }
    }

    /// dspy's `_append_history_event` guard is the caller's — a module decides whether an event is
    /// worth recording — so this only records the append itself.
    pub fn push(&mut self, event: Map<String, Value>) {
        self.messages.push(event);
    }

    /// The value a history reaches a request as: `{"messages": [...]}`, which is what the
    /// [formatter](crate::adapter::history) reads back. Serialization cannot fail for a value built
    /// from JSON objects, so the object form is returned directly.
    pub fn to_value(&self) -> Value {
        serde_json::to_value(self).unwrap_or_else(|_| Value::Object(Map::new()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn event(fields: Value) -> Map<String, Value> {
        fields.as_object().expect("an object").clone()
    }

    /// dspy's example: a history is a list of dicts keyed by the signature's fields, and it writes
    /// back out under `messages`.
    #[test]
    fn it_carries_messages_keyed_by_the_signatures_fields() {
        let history = History::new(vec![
            event(json!({ "question": "What is the capital of France?", "answer": "Paris" })),
            event(json!({ "question": "What is the capital of Germany?", "answer": "Berlin" })),
        ]);
        assert_eq!(
            history.to_value(),
            json!({
                "messages": [
                    { "question": "What is the capital of France?", "answer": "Paris" },
                    { "question": "What is the capital of Germany?", "answer": "Berlin" },
                ]
            })
        );
    }

    /// dspy `History.model_validate({"messages": [...]})`: a serialized history reads back into the
    /// type, which is how a caller can hand one in as a plain mapping.
    #[test]
    fn it_reads_back_from_a_serialized_mapping() {
        let history: History =
            serde_json::from_value(json!({ "messages": [{ "question": "old" }] })).expect("parses");
        assert_eq!(history.messages.len(), 1);
        assert_eq!(history.messages[0]["question"], json!("old"));
    }

    /// `extra="forbid"`: a mapping with a key the model never declared is refused, not dropped.
    #[test]
    fn it_refuses_a_mapping_with_an_unknown_key() {
        let refused = serde_json::from_value::<History>(json!({ "messages": [], "extra": 1 }));
        assert!(refused.is_err());
    }

    /// `messages` has no default upstream, so a mapping without it is an error rather than empty.
    #[test]
    fn it_requires_messages() {
        assert!(serde_json::from_value::<History>(json!({})).is_err());
    }

    /// An agent loop records a turn by appending to the list, which upstream's frozen model still
    /// allows because the list itself is mutable.
    #[test]
    fn a_turn_is_recorded_by_appending_an_event() {
        let mut history = History::default();
        history.push(event(json!({ "question": "cats", "answer": "found cats" })));
        assert_eq!(history.messages.len(), 1);
        assert_eq!(history.messages[0]["answer"], json!("found cats"));
    }
}
