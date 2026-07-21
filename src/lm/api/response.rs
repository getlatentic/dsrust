//! `LMOutput` and `LMResponse`.

use serde_json::Value;

use super::part::{LmPart, Metadata};
use crate::lm::LmUsage;

/// One candidate completion. A candidate is a structure rather than a string, which is what
/// makes `finish_reason` and `truncated` readable at all.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LmOutput {
    pub parts: Vec<LmPart>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub truncated: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_output: Option<Value>,
    #[serde(default, skip_serializing_if = "Metadata::is_empty")]
    pub provider_data: Metadata,
    #[serde(default, skip_serializing_if = "Metadata::is_empty")]
    pub metadata: Metadata,
}

impl LmOutput {
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            parts: vec![LmPart::text(text)],
            ..Self::default()
        }
    }

    pub fn as_text(&self) -> String {
        self.parts.iter().filter_map(LmPart::as_text).collect()
    }

    /// Every tool the model asked for, in the order it asked.
    pub fn tool_calls(&self) -> impl Iterator<Item = &LmPart> {
        self.parts
            .iter()
            .filter(|part| matches!(part, LmPart::ToolCall { .. }))
    }
}

#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LmResponse {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub outputs: Vec<LmOutput>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<LmUsage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost: Option<f64>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub cache_hit: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_response: Option<Value>,
    #[serde(default, skip_serializing_if = "Metadata::is_empty")]
    pub provider_data: Metadata,
    #[serde(default, skip_serializing_if = "Metadata::is_empty")]
    pub metadata: Metadata,
}

impl LmResponse {
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            outputs: vec![LmOutput::text(text)],
            ..Self::default()
        }
    }

    /// The candidate an adapter parses, which is the only one unless several were asked for.
    pub fn first_text(&self) -> String {
        self.outputs.first().map(LmOutput::as_text).unwrap_or_default()
    }

    /// A replay is not billed, so anything totalling spend reads this rather than `usage`.
    pub fn spend(&self) -> Option<LmUsage> {
        self.usage.clone().filter(|_| !self.cache_hit)
    }
}

fn is_false(value: &bool) -> bool {
    !*value
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_candidate_carries_why_it_stopped() {
        let output = LmOutput {
            finish_reason: Some("length".to_owned()),
            truncated: true,
            ..LmOutput::text("as far as it got")
        };
        assert_eq!(output.as_text(), "as far as it got");
        assert_eq!(output.finish_reason.as_deref(), Some("length"));
        assert!(output.truncated);
    }

    #[test]
    fn a_replayed_reply_reports_its_worth_but_costs_nothing() {
        let response = LmResponse {
            usage: Some(LmUsage::counted(10, 4)),
            cache_hit: true,
            ..LmResponse::text("replayed")
        };
        assert_eq!(response.usage.as_ref().and_then(LmUsage::total), Some(14));
        assert_eq!(response.spend(), None);
    }

    #[test]
    fn tool_calls_are_readable_off_a_candidate() {
        let output = LmOutput {
            parts: vec![
                LmPart::text("calling"),
                LmPart::ToolCall {
                    id: Some("call_1".to_owned()),
                    name: "search".to_owned(),
                    args: Metadata::new(),
                    provider_data: Metadata::new(),
                    metadata: Metadata::new(),
                },
            ],
            ..LmOutput::default()
        };
        assert_eq!(output.tool_calls().count(), 1);
        assert_eq!(output.as_text(), "calling");
    }

    #[test]
    fn a_response_round_trips_through_json() {
        let response = LmResponse {
            model: Some("openai/gpt-4o".to_owned()),
            cost: Some(0.002),
            response_id: Some("resp_1".to_owned()),
            ..LmResponse::text("hello")
        };
        let written = serde_json::to_value(&response).expect("serializes");
        assert_eq!(written["outputs"][0]["parts"][0]["text"], json!("hello"));
        assert_eq!(
            serde_json::from_value::<LmResponse>(written).expect("parses"),
            response
        );
    }
}
