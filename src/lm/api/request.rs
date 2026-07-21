//! `LMRequest`.

use super::config::LmConfig;
use super::message::{LmMessage, LmToolSpec};
use super::part::Metadata;

/// One call as a value. The model is part of it, which is what lets a request be routed and
/// cached without ambient state deciding who answers.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LmRequest {
    pub model: String,
    pub messages: Vec<LmMessage>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<LmToolSpec>,
    #[serde(default)]
    pub config: LmConfig,
    #[serde(default, skip_serializing_if = "Metadata::is_empty")]
    pub metadata: Metadata,
}

impl LmRequest {
    pub fn new(model: impl Into<String>, messages: Vec<LmMessage>) -> Self {
        Self {
            model: model.into(),
            messages,
            tools: Vec::new(),
            config: LmConfig::default(),
            metadata: Metadata::new(),
        }
    }

    pub fn configured(mut self, config: LmConfig) -> Self {
        self.config = config;
        self
    }

    pub fn with_tools(mut self, tools: Vec<LmToolSpec>) -> Self {
        self.tools = tools;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lm::api::{LmPart, RolloutId};
    use serde_json::json;

    fn request() -> LmRequest {
        LmRequest::new(
            "openai/gpt-4o",
            vec![LmMessage::user(vec![LmPart::text("Why?")])],
        )
    }

    #[test]
    fn a_request_round_trips_through_json() {
        let request = request().configured(
            LmConfig::from_kwargs([
                ("temperature".to_owned(), json!(0.7)),
                ("rollout_id".to_owned(), json!(3)),
            ])
            .expect("builds"),
        );
        let written = serde_json::to_value(&request).expect("serializes");
        assert_eq!(written["model"], json!("openai/gpt-4o"));
        assert_eq!(written["config"]["cache"]["rollout_id"], json!(3));
        assert_eq!(
            serde_json::from_value::<LmRequest>(written).expect("parses"),
            request
        );
    }

    #[test]
    fn the_rollout_id_reads_back_from_where_it_folded_to() {
        let request = request().configured(
            LmConfig::from_kwargs([("rollout_id".to_owned(), json!(9))]).expect("builds"),
        );
        assert_eq!(request.config.rollout_id(), Some(&RolloutId::Number(9)));
    }

    #[test]
    fn a_request_forbids_what_it_does_not_declare() {
        assert!(
            serde_json::from_value::<LmRequest>(json!({
                "model": "m",
                "messages": [],
                "prompt": "the old spelling",
            }))
            .is_err()
        );
    }
}
