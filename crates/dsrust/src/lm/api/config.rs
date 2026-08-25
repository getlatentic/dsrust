//! `LMConfig` and the four nested configs it folds its flat aliases into.

use serde_json::Value;

use super::part::Metadata;

/// The three request settings whose Python type is a union, each an enum here so the wire cannot be
/// given a shape the provider rejects:
///
/// ```
/// use dsrust::lm::api::{Logprobs, RolloutId, ToolChoiceMode};
///
/// // `logprobs: bool | int` — on, or the top n per token.
/// assert_eq!(serde_json::to_value(Logprobs::Enabled(true)).unwrap(), serde_json::json!(true));
/// assert_eq!(serde_json::to_value(Logprobs::Top(5)).unwrap(), serde_json::json!(5));
///
/// // `rollout_id: int | str` — a counter or a name, and upstream accepts either.
/// assert_eq!(serde_json::to_value(RolloutId::Number(7)).unwrap(), serde_json::json!(7));
/// assert_eq!(serde_json::to_value(RolloutId::Text("a".into())).unwrap(), serde_json::json!("a"));
///
/// // `tool_choice` defaults to letting the model decide.
/// assert_eq!(ToolChoiceMode::default(), ToolChoiceMode::Auto);
/// ```
/// `bool | int`: enable logprobs, or ask for the top `n` per token.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub enum Logprobs {
    Enabled(bool),
    Top(u32),
}

/// `int | str`, and neither narrower — upstream accepts both and a caller's own identifier is
/// as likely to be a name as a counter.
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub enum RolloutId {
    Number(i64),
    Text(String),
}

#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LmReasoningConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoiceMode {
    #[default]
    Auto,
    Required,
    None,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LmToolChoice {
    #[serde(default)]
    pub mode: ToolChoiceMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allowed: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parallel: Option<bool>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LmCacheConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rollout_id: Option<RolloutId>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LmPromptCacheConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
}

/// `extra="forbid"`, with anything unrecognised routed to [`extensions`](Self::extensions)
/// instead — the opposite of `LmUsage`, which allows unknowns at the top level.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LmConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Logprobs>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<LmReasoningConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<LmToolChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache: Option<LmCacheConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_cache: Option<LmPromptCacheConfig>,
    #[serde(default, skip_serializing_if = "Metadata::is_empty")]
    pub extensions: Metadata,
}

/// The keys `_split_config_kwargs` recognises. Everything else a caller passes becomes an
/// extension rather than an error, which is what lets a provider-specific knob through without
/// this crate having to know it.
const KNOWN_KEYS: [&str; 16] = [
    "temperature",
    "max_tokens",
    "top_p",
    "stop",
    "n",
    "logprobs",
    "response_format",
    "reasoning",
    "reasoning_effort",
    "tool_choice",
    "parallel_tool_calls",
    "cache",
    "rollout_id",
    "prompt_cache",
    "prompt_cache_key",
    "extensions",
];

/// The four flat spellings that fold into a nested config, as `(flat, owner, nested)`.
const ALIASES: [(&str, &str, &str); 4] = [
    ("reasoning_effort", "reasoning", "effort"),
    ("parallel_tool_calls", "tool_choice", "parallel"),
    ("rollout_id", "cache", "rollout_id"),
    ("prompt_cache_key", "prompt_cache", "key"),
];

impl LmConfig {
    /// Upstream's `from_kwargs`: known keys stay, unknown keys become extensions, and the four
    /// flat aliases fold into the nested config that owns them.
    pub fn from_kwargs(kwargs: impl IntoIterator<Item = (String, Value)>) -> Result<Self, String> {
        let mut known = serde_json::Map::new();
        let mut extensions = Metadata::new();
        for (key, value) in kwargs {
            match key.as_str() {
                "extensions" => match value {
                    Value::Object(given) => extensions.extend(given),
                    other => return Err(format!("extensions must be an object, got {other}")),
                },
                key if KNOWN_KEYS.contains(&key) => {
                    known.insert(key.to_owned(), value);
                }
                _ => {
                    extensions.insert(key, value);
                }
            }
        }
        fold_aliases(&mut known);
        known.insert("extensions".to_owned(), Value::Object(extensions));
        serde_json::from_value(Value::Object(known)).map_err(|error| error.to_string())
    }

    pub fn rollout_id(&self) -> Option<&RolloutId> {
        self.cache.as_ref()?.rollout_id.as_ref()
    }
}

/// A flat alias only fills its nested slot when that slot is not already spoken for, matching
/// upstream's `if "x" in data and "y" not in data`.
fn fold_aliases(known: &mut serde_json::Map<String, Value>) {
    for (flat, owner, nested) in ALIASES {
        let Some(value) = known.remove(flat) else {
            continue;
        };
        let mut config = match known.remove(owner) {
            Some(Value::Object(existing)) => existing,
            _ => serde_json::Map::new(),
        };
        config.entry(nested.to_owned()).or_insert(value);
        known.insert(owner.to_owned(), Value::Object(config));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn from(pairs: [(&str, Value); 1]) -> LmConfig {
        LmConfig::from_kwargs(pairs.map(|(key, value)| (key.to_owned(), value))).expect("builds")
    }

    #[test]
    fn each_flat_alias_folds_into_the_nested_config_that_owns_it() {
        assert_eq!(
            from([("reasoning_effort", json!("high"))]).reasoning,
            Some(LmReasoningConfig {
                effort: Some("high".to_owned()),
                ..LmReasoningConfig::default()
            })
        );
        assert_eq!(
            from([("parallel_tool_calls", json!(false))])
                .tool_choice
                .expect("a tool choice")
                .parallel,
            Some(false)
        );
        assert_eq!(
            from([("rollout_id", json!(7))]).rollout_id(),
            Some(&RolloutId::Number(7))
        );
        assert_eq!(
            from([("prompt_cache_key", json!("k"))])
                .prompt_cache
                .expect("a prompt cache")
                .key,
            Some("k".to_owned())
        );
    }

    /// `rollout_id` is `int | str` upstream, so narrowing it to a counter would refuse a name a
    /// caller is entitled to use.
    #[test]
    fn a_rollout_id_is_a_number_or_a_name() {
        assert_eq!(
            from([("rollout_id", json!("attempt-two"))]).rollout_id(),
            Some(&RolloutId::Text("attempt-two".to_owned()))
        );
    }

    #[test]
    fn an_unknown_key_becomes_an_extension_rather_than_an_error() {
        let config = from([("anthropic_beta", json!("tools-2024"))]);
        assert_eq!(config.extensions["anthropic_beta"], json!("tools-2024"));
        assert_eq!(config.temperature, None);
    }

    /// The nested config wins where both spellings arrive, which is upstream's
    /// `"x" in data and "y" not in data` read the other way round.
    #[test]
    fn an_explicit_nested_value_is_not_overwritten_by_its_flat_alias() {
        let config = LmConfig::from_kwargs([
            ("reasoning".to_owned(), json!({ "effort": "low" })),
            ("reasoning_effort".to_owned(), json!("high")),
        ])
        .expect("builds");
        assert_eq!(
            config.reasoning.expect("reasoning").effort,
            Some("low".to_owned())
        );
    }

    #[test]
    fn a_flat_alias_keeps_the_rest_of_its_nested_config() {
        let config = LmConfig::from_kwargs([
            ("reasoning".to_owned(), json!({ "summary": "concise" })),
            ("reasoning_effort".to_owned(), json!("high")),
        ])
        .expect("builds");
        let reasoning = config.reasoning.expect("reasoning");
        assert_eq!(reasoning.effort, Some("high".to_owned()));
        assert_eq!(reasoning.summary, Some("concise".to_owned()));
    }

    #[test]
    fn logprobs_is_a_flag_or_a_count() {
        assert_eq!(
            from([("logprobs", json!(true))]).logprobs,
            Some(Logprobs::Enabled(true))
        );
        assert_eq!(
            from([("logprobs", json!(5))]).logprobs,
            Some(Logprobs::Top(5))
        );
    }

    /// `extra="forbid"`: a key that reaches the struct itself is an error, which is what routes
    /// unknowns through `from_kwargs` into extensions instead.
    #[test]
    fn the_struct_itself_forbids_what_it_does_not_declare() {
        assert!(
            serde_json::from_value::<LmConfig>(json!({ "temperature": 0.7, "wat": 1 })).is_err()
        );
        assert!(serde_json::from_value::<LmToolChoice>(json!({ "wat": 1 })).is_err());
    }

    #[test]
    fn tool_choice_defaults_to_auto_the_way_upstream_does() {
        let choice: LmToolChoice = serde_json::from_value(json!({})).expect("parses");
        assert_eq!(choice.mode, ToolChoiceMode::Auto);
    }
}
