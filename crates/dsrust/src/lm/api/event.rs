//! `LMStreamEvent` — the five things a stream emits.

use super::delta::LmDelta;
use super::response::LmResponse;
use crate::lm::LmUsage;

/// Upstream's error event carries a live `Exception` to re-raise. A Rust stream carries the
/// message instead, since there is no exception object to hand back.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LmStreamEvent {
    Start {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        model: Option<String>,
    },
    Delta {
        #[serde(default)]
        output_index: usize,
        part_index: usize,
        delta: LmDelta,
    },
    OutputEnd {
        #[serde(default)]
        output_index: usize,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        finish_reason: Option<String>,
        #[serde(default)]
        truncated: bool,
    },
    End {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        usage: Option<LmUsage>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cost: Option<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        response: Option<Box<LmResponse>>,
    },
    Error {
        error: String,
    },
}

impl LmStreamEvent {
    pub fn delta(part_index: usize, delta: LmDelta) -> Self {
        Self::Delta {
            output_index: 0,
            part_index,
            delta,
        }
    }

    pub fn end() -> Self {
        Self::End {
            usage: None,
            cost: None,
            response: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn each_event_travels_under_the_tag_upstream_gives_it() {
        for (event, tag) in [
            (LmStreamEvent::Start { model: None }, "start"),
            (LmStreamEvent::delta(0, LmDelta::text("hi")), "delta"),
            (
                LmStreamEvent::OutputEnd {
                    output_index: 0,
                    finish_reason: Some("stop".to_owned()),
                    truncated: false,
                },
                "output_end",
            ),
            (LmStreamEvent::end(), "end"),
            (
                LmStreamEvent::Error {
                    error: "boom".to_owned(),
                },
                "error",
            ),
        ] {
            let written = serde_json::to_value(&event).expect("serializes");
            assert_eq!(written["type"], json!(tag));
            assert_eq!(
                serde_json::from_value::<LmStreamEvent>(written).expect("round-trips"),
                event
            );
        }
    }

    #[test]
    fn an_output_index_defaults_to_the_first_candidate() {
        let event: LmStreamEvent = serde_json::from_value(json!({
            "type": "delta",
            "part_index": 0,
            "delta": { "type": "text_delta", "text": "hi" },
        }))
        .expect("parses");
        assert_eq!(event, LmStreamEvent::delta(0, LmDelta::text("hi")));
    }
}
