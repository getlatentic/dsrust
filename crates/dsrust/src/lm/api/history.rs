//! `LMHistoryEntry` — one call kept for inspection after the fact.

use super::part::Metadata;
use super::request::LmRequest;
use super::response::LmResponse;

/// `extra="allow"` upstream, so anything a caller attaches survives rather than being dropped —
/// ```
/// use dsrust::lm::api::LmHistoryEntry;
///
/// // `extra="allow"` upstream, so a field a caller attached survives a round trip rather than
/// // being dropped — which is what makes a history written by one tool readable by another.
/// let written = serde_json::json!({
///     "request": { "model": "openai/gpt-4o-mini", "messages": [] },
///     "response": { "outputs": [] },
///     "timestamp": "2026-01-01T00:00:00Z",
///     "uuid": "abc",
///     "model": "openai/gpt-4o-mini",
///     "a_field_this_crate_does_not_model": 1,
/// });
/// let entry: LmHistoryEntry = serde_json::from_value(written.clone()).expect("unknowns allowed");
/// assert_eq!(entry.uuid, "abc");
/// ```
/// deliberately the opposite of the request and response it holds, which both forbid unknowns.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LmHistoryEntry {
    pub request: LmRequest,
    pub response: LmResponse,
    pub timestamp: String,
    pub uuid: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_type: Option<String>,
    #[serde(flatten)]
    pub extra: Metadata,
}

impl LmHistoryEntry {
    /// The clock and the identifier are the caller's: there is no ambient one to read here, and
    /// a fixed pair is what lets a test compare two entries at all.
    pub fn new(
        request: LmRequest,
        response: LmResponse,
        timestamp: impl Into<String>,
        uuid: impl Into<String>,
    ) -> Self {
        Self {
            request,
            response,
            timestamp: timestamp.into(),
            uuid: uuid.into(),
            model_type: None,
            extra: Metadata::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::message::LmMessage;
    use super::super::part::LmPart;
    use super::*;
    use serde_json::json;

    fn entry() -> LmHistoryEntry {
        LmHistoryEntry::new(
            LmRequest::new(
                "openai/gpt-4o",
                vec![LmMessage::user(vec![LmPart::text("Why?")])],
            ),
            LmResponse::text("Because."),
            "2026-07-21T10:00:00Z",
            "01J0000000000000000000",
        )
    }

    #[test]
    fn an_entry_keeps_both_halves_of_the_call() {
        let entry = entry();
        assert_eq!(entry.request.model, "openai/gpt-4o");
        assert_eq!(entry.response.first_text(), "Because.");

        let written = serde_json::to_value(&entry).expect("serializes");
        assert_eq!(
            serde_json::from_value::<LmHistoryEntry>(written).expect("round-trips"),
            entry
        );
    }

    /// `extra="allow"`: a field this crate does not model is kept, where the request and response
    /// inside it would reject the same key.
    #[test]
    fn a_field_nobody_modelled_survives_the_round_trip() {
        let mut written = serde_json::to_value(entry()).expect("serializes");
        written["trace_id"] = json!("abc123");

        let entry: LmHistoryEntry = serde_json::from_value(written).expect("unknowns are allowed");
        assert_eq!(entry.extra["trace_id"], json!("abc123"));

        let back = serde_json::to_value(&entry).expect("serializes");
        assert_eq!(back["trace_id"], json!("abc123"), "at the top level");
    }
}
