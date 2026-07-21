//! `LMDelta` — the six increments a streamed reply arrives in.

use super::part::LmPart;

/// Upstream types the three part-carrying deltas as their specific part class. Rust has one
/// `LmPart` enum rather than eleven types, so the variant is checked where the delta is applied
/// — which is where upstream's `isinstance` check happens too.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LmDelta {
    TextDelta {
        text: String,
    },
    ThinkingDelta {
        text: String,
    },
    ToolCallDelta {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        args_delta: Option<String>,
    },
    CitationDelta {
        citation: LmPart,
    },
    ImageDelta {
        image: LmPart,
    },
    AudioDelta {
        audio: LmPart,
    },
}

impl LmDelta {
    pub fn text(text: impl Into<String>) -> Self {
        Self::TextDelta { text: text.into() }
    }

    pub fn thinking(text: impl Into<String>) -> Self {
        Self::ThinkingDelta { text: text.into() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn each_delta_travels_under_the_tag_upstream_gives_it() {
        let tags = [
            (LmDelta::text("hi"), "text_delta"),
            (LmDelta::thinking("hmm"), "thinking_delta"),
            (
                LmDelta::ToolCallDelta {
                    id: None,
                    name: Some("search".to_owned()),
                    args_delta: Some("{\"q\"".to_owned()),
                },
                "tool_call_delta",
            ),
        ];
        for (delta, tag) in tags {
            let written = serde_json::to_value(&delta).expect("serializes");
            assert_eq!(written["type"], json!(tag));
            assert_eq!(
                serde_json::from_value::<LmDelta>(written).expect("round-trips"),
                delta
            );
        }
    }

    /// `extra="ignore"` upstream, which is serde's default — an unknown key must not fail a
    /// stream mid-flight.
    #[test]
    fn an_unknown_key_does_not_fail_a_delta() {
        let delta: LmDelta =
            serde_json::from_value(json!({ "type": "text_delta", "text": "hi", "seq": 3 }))
                .expect("parses");
        assert_eq!(delta, LmDelta::text("hi"));
    }
}
