//! `LMRequestPatch` — what a type strategy contributes while an adapter is still building a call.

use super::config::LmConfig;
use super::message::{LmMessage, LmToolSpec};
use super::part::{LmPart, Metadata};

/// A whole [`LmRequest`](super::request::LmRequest) is what a model receives; a patch is the
/// composable piece a field's own type adds to one — extra parts, a native tool, config, or
/// fields it wants kept out of the adapter's ordinary rendering.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LmRequestPatch {
    pub messages: Vec<LmMessage>,
    pub system_parts: Vec<LmPart>,
    pub user_parts: Vec<LmPart>,
    pub assistant_parts: Vec<LmPart>,
    pub tools: Vec<LmToolSpec>,
    pub config: Option<LmConfig>,
    pub delete_input_fields: Vec<String>,
    pub delete_output_fields: Vec<String>,
    pub metadata: Metadata,
}

impl LmRequestPatch {
    /// This patch followed by `other`, which is what makes several fields' contributions to one
    /// call combine in declaration order.
    pub fn merge(mut self, other: Self) -> Self {
        self.messages.extend(other.messages);
        self.system_parts.extend(other.system_parts);
        self.user_parts.extend(other.user_parts);
        self.assistant_parts.extend(other.assistant_parts);
        self.tools.extend(other.tools);
        self.config = merged_config(self.config, other.config);
        self.delete_input_fields.extend(other.delete_input_fields);
        self.delete_output_fields.extend(other.delete_output_fields);
        self.metadata.extend(other.metadata);
        self
    }
}

/// The later patch wins field by field, so one contributor raising the temperature does not
/// discard another's `max_tokens`.
fn merged_config(left: Option<LmConfig>, right: Option<LmConfig>) -> Option<LmConfig> {
    let (Some(left), Some(right)) = (left.clone(), right.clone()) else {
        return right.or(left);
    };
    Some(LmConfig {
        temperature: right.temperature.or(left.temperature),
        max_tokens: right.max_tokens.or(left.max_tokens),
        top_p: right.top_p.or(left.top_p),
        stop: right.stop.or(left.stop),
        n: right.n.or(left.n),
        logprobs: right.logprobs.or(left.logprobs),
        response_format: right.response_format.or(left.response_format),
        reasoning: right.reasoning.or(left.reasoning),
        tool_choice: right.tool_choice.or(left.tool_choice),
        cache: right.cache.or(left.cache),
        prompt_cache: right.prompt_cache.or(left.prompt_cache),
        extensions: left
            .extensions
            .into_iter()
            .chain(right.extensions)
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn config(pairs: [(&str, serde_json::Value); 1]) -> Option<LmConfig> {
        Some(
            LmConfig::from_kwargs(pairs.map(|(key, value)| (key.to_owned(), value)))
                .expect("builds"),
        )
    }

    #[test]
    fn merging_keeps_both_contributions_in_order() {
        let first = LmRequestPatch {
            user_parts: vec![LmPart::text("one")],
            delete_input_fields: vec!["photo".to_owned()],
            ..LmRequestPatch::default()
        };
        let second = LmRequestPatch {
            user_parts: vec![LmPart::text("two")],
            delete_input_fields: vec!["clip".to_owned()],
            ..LmRequestPatch::default()
        };

        let merged = first.merge(second);
        assert_eq!(
            merged.user_parts,
            vec![LmPart::text("one"), LmPart::text("two")]
        );
        assert_eq!(merged.delete_input_fields, ["photo", "clip"]);
    }

    /// Field by field, so one contributor's setting does not wipe another's.
    #[test]
    fn merging_configs_keeps_what_only_one_side_set() {
        let merged = LmRequestPatch {
            config: config([("max_tokens", json!(100))]),
            ..LmRequestPatch::default()
        }
        .merge(LmRequestPatch {
            config: config([("temperature", json!(0.7))]),
            ..LmRequestPatch::default()
        });

        let config = merged.config.expect("a config");
        assert_eq!(config.max_tokens, Some(100), "the earlier patch survives");
        assert_eq!(config.temperature, Some(0.7));
    }

    #[test]
    fn the_later_patch_wins_where_both_set_the_same_field() {
        let merged = LmRequestPatch {
            config: config([("temperature", json!(0.2))]),
            ..LmRequestPatch::default()
        }
        .merge(LmRequestPatch {
            config: config([("temperature", json!(0.9))]),
            ..LmRequestPatch::default()
        });
        assert_eq!(merged.config.expect("a config").temperature, Some(0.9));
    }

    #[test]
    fn a_patch_with_no_config_leaves_the_others_alone() {
        let merged = LmRequestPatch {
            config: config([("temperature", json!(0.2))]),
            ..LmRequestPatch::default()
        }
        .merge(LmRequestPatch::default());
        assert_eq!(merged.config.expect("a config").temperature, Some(0.2));
    }
}
