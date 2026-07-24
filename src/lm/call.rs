//! How a call is sampled, and what it cost.
//!
//! The request and response themselves are the typed 3.3 values in [`api`](super::api); what
//! stays here is the sampling config a module varies per attempt and the usage every provider
//! reports, both of which predate that boundary and are read throughout the crate.

use serde_json::Value;

/// How a model should sample its reply.
///
/// These belong to one call rather than to a model: the same model answers twice and the two
/// attempts differ only here. That is what lets `BestOfN` mean anything and what lets a bootstrap
/// round after the first not repeat itself.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LmConfig {
    /// Unset leaves each provider on the default it is already sent.
    pub temperature: Option<f64>,
    pub max_tokens: Option<u32>,
    /// How many completions to ask for, upstream's `n`. They arrive as the response's outputs;
    /// only the OpenAI-shaped services take the field, so the others answer once however many are
    /// asked for.
    pub completions: Option<u32>,
    /// Varied to miss the response cache, so a second attempt is answered rather than replayed.
    ///
    /// Never sent to a provider — it is part of the cache key and nothing else, which is exactly
    /// what upstream does with it. Two requests alike but for this one number are two different
    /// cache entries and therefore two real calls, and that is the whole mechanism behind
    /// `BestOfN` and behind a bootstrap round after the first.
    pub rollout_id: Option<u64>,
}

impl LmConfig {
    /// A fresh rollout at the temperature upstream re-asks with. dspy's
    /// `lm.copy(rollout_id=n, temperature=1.0)`, which is how every one of its retry-shaped
    /// modules makes attempt two differ from attempt one.
    pub fn rollout(id: u64) -> Self {
        Self {
            temperature: Some(1.0),
            rollout_id: Some(id),
            ..Self::default()
        }
    }
}

/// What one call cost, in the two counts every provider agrees on.
///
/// Each of the three names them differently — Anthropic `input_tokens`/`output_tokens`, the
/// OpenAI-shaped services `prompt_tokens`/`completion_tokens`, ollama `prompt_eval_count`/
/// `eval_count` — so normalising them here is the whole reason a caller can compare the cost of
/// two adapters without knowing who answered.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LmUsage {
    /// dspy's own name for what the prompt cost, mirrored with [`prompt_tokens`](Self::prompt_tokens).
    pub input_tokens: Option<u32>,
    /// dspy's own name for what the reply cost, mirrored with
    /// [`completion_tokens`](Self::completion_tokens).
    pub output_tokens: Option<u32>,
    /// Derived from the two above when both are known and a provider did not state it.
    pub total_tokens: Option<u32>,
    /// The OpenAI-shaped spelling of `input_tokens`, populated alongside it rather than instead.
    pub prompt_tokens: Option<u32>,
    /// The OpenAI-shaped spelling of `output_tokens`.
    pub completion_tokens: Option<u32>,
    pub reasoning_tokens: Option<u32>,
    pub cache_read_tokens: Option<u32>,
    pub cache_write_tokens: Option<u32>,
    pub input_audio_tokens: Option<u32>,
    pub output_audio_tokens: Option<u32>,
    /// A provider's own breakdown, kept whole. Upstream allows unknown counters rather than
    /// rejecting them, so a count this crate does not model still arrives.
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub details: serde_json::Map<String, Value>,
    /// Any counter this crate does not model, kept rather than dropped.
    ///
    /// Upstream's `extra="allow"`, and deliberately the opposite of [`LmConfig`], which forbids
    /// unknowns so they are routed into `extensions` instead. A provider that starts reporting a
    /// counter nobody has modelled yet still hands it to a caller through here.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

impl LmUsage {
    /// The two counts every provider reports, under both spellings, with the total derived.
    ///
    /// The constructor rather than a literal, because the mirroring is upstream's
    /// `fill_aliases` validator and a struct literal in Rust runs nothing.
    pub fn counted(input_tokens: u32, output_tokens: u32) -> Self {
        Self {
            input_tokens: Some(input_tokens),
            output_tokens: Some(output_tokens),
            ..Self::default()
        }
        .fill_aliases()
    }

    /// Each naming convention filled from the other, and the total from both.
    ///
    /// Upstream populates *both* sets rather than normalising onto one, because "both are
    /// existing user-visible interfaces". A caller reading `prompt_tokens` finds the same number
    /// as one reading `input_tokens`.
    pub fn fill_aliases(mut self) -> Self {
        self.input_tokens = self.input_tokens.or(self.prompt_tokens);
        self.output_tokens = self.output_tokens.or(self.completion_tokens);
        self.prompt_tokens = self.prompt_tokens.or(self.input_tokens);
        self.completion_tokens = self.completion_tokens.or(self.output_tokens);
        if self.total_tokens.is_none()
            && let (Some(input), Some(output)) = (self.input_tokens, self.output_tokens)
        {
            self.total_tokens = Some(input + output);
        }
        self
    }

    /// What one call cost altogether, if enough of it is known to say.
    pub fn total(&self) -> Option<u32> {
        self.total_tokens
    }

    /// What several calls cost together, which is what one `forward` costs when a fallback or a
    /// feedback retry made more than one.
    ///
    /// A model reporting nothing stays nothing rather than becoming zero, so a scripted model in
    /// the chain cannot make a real count read as free — and one real count among several silent
    /// ones is still reported, because it is what is known rather than the whole. The same holds
    /// per counter now that each is optional: two calls of which one reported reasoning tokens
    /// report that one's.
    pub fn merge(left: Option<Self>, right: Option<Self>) -> Option<Self> {
        match (left, right) {
            (None, other) | (other, None) => other,
            (Some(left), Some(right)) => Some(
                Self {
                    input_tokens: added(left.input_tokens, right.input_tokens),
                    output_tokens: added(left.output_tokens, right.output_tokens),
                    total_tokens: added(left.total_tokens, right.total_tokens),
                    prompt_tokens: added(left.prompt_tokens, right.prompt_tokens),
                    completion_tokens: added(left.completion_tokens, right.completion_tokens),
                    reasoning_tokens: added(left.reasoning_tokens, right.reasoning_tokens),
                    cache_read_tokens: added(left.cache_read_tokens, right.cache_read_tokens),
                    cache_write_tokens: added(left.cache_write_tokens, right.cache_write_tokens),
                    input_audio_tokens: added(left.input_audio_tokens, right.input_audio_tokens),
                    output_audio_tokens: added(left.output_audio_tokens, right.output_audio_tokens),
                    details: left.details.into_iter().chain(right.details).collect(),
                    extra: left.extra.into_iter().chain(right.extra).collect(),
                }
                .fill_aliases(),
            ),
        }
    }
}

/// Two counters added, where a counter nobody reported stays unreported.
fn added(left: Option<u32>, right: Option<u32>) -> Option<u32> {
    match (left, right) {
        (None, other) | (other, None) => other,
        (Some(left), Some(right)) => Some(left + right),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// dspy populates *both* naming conventions rather than choosing one, because "both are
    /// existing user-visible interfaces". A caller reading `prompt_tokens` finds what a caller
    /// reading `input_tokens` finds.
    #[test]
    fn both_spellings_are_filled_from_whichever_arrived() {
        let from_dspy_names = LmUsage::counted(12, 30);
        assert_eq!(from_dspy_names.prompt_tokens, Some(12));
        assert_eq!(from_dspy_names.completion_tokens, Some(30));

        let from_provider_names = LmUsage {
            prompt_tokens: Some(12),
            completion_tokens: Some(30),
            ..LmUsage::default()
        }
        .fill_aliases();
        assert_eq!(from_provider_names.input_tokens, Some(12));
        assert_eq!(from_provider_names.output_tokens, Some(30));
        assert_eq!(
            from_provider_names, from_dspy_names,
            "either way in, the same out"
        );
    }

    /// Upstream allows unknown counters rather than rejecting them, so one this crate has never
    /// heard of still reaches a caller. That is `extra="allow"` — and the opposite of
    /// [`LmConfig`], which forbids unknowns so they route into `extensions` instead.
    #[test]
    fn a_counter_nobody_modelled_survives_the_round_trip() {
        let raw = serde_json::json!({
            "input_tokens": 10,
            "output_tokens": 4,
            "speculation_tokens": 3,
        });
        let usage: LmUsage = serde_json::from_value(raw).expect("unknown counters are allowed");

        assert_eq!(usage.input_tokens, Some(10));
        assert_eq!(
            usage.extra.get("speculation_tokens"),
            Some(&serde_json::json!(3)),
            "kept rather than dropped"
        );

        let written = serde_json::to_value(&usage).expect("serializes");
        assert_eq!(
            written["speculation_tokens"],
            serde_json::json!(3),
            "and goes back out at the top level, not nested"
        );
    }

    /// A count nobody reported stays unknown rather than becoming zero — which is the whole
    /// reason the counters are optional, and what a `u32` could not say.
    #[test]
    fn an_unreported_counter_stays_unknown() {
        let half = LmUsage {
            input_tokens: Some(9),
            ..LmUsage::default()
        }
        .fill_aliases();
        assert_eq!(half.output_tokens, None);
        assert_eq!(half.total_tokens, None, "a total needs both halves");
    }

    /// Merging carries every counter, so two calls of which one reported reasoning tokens report
    /// that one's.
    #[test]
    fn merging_carries_the_counters_only_one_call_reported() {
        let plain = LmUsage::counted(10, 4);
        let reasoned = LmUsage {
            reasoning_tokens: Some(7),
            ..LmUsage::counted(6, 2)
        }
        .fill_aliases();

        let both = LmUsage::merge(Some(plain), Some(reasoned)).expect("merged");
        assert_eq!(both.input_tokens, Some(16));
        assert_eq!(
            both.reasoning_tokens,
            Some(7),
            "reported by one, kept for both"
        );
        assert_eq!(both.total_tokens, Some(22));
    }

    #[test]
    fn usage_totals_the_two_counts() {
        let usage = LmUsage::counted(12, 30);
        assert_eq!(usage.total(), Some(42));
    }

    fn usage(input_tokens: u32, output_tokens: u32) -> Option<LmUsage> {
        Some(LmUsage::counted(input_tokens, output_tokens))
    }

    #[test]
    fn merging_adds_both_counts() {
        assert_eq!(
            LmUsage::merge(usage(1, 2), usage(10, 20)),
            usage(11, 22),
            "a fallback ask costs what both asks cost"
        );
    }

    /// The trap: treating a silent model as zero would turn one real count into a total that
    /// looks complete, and treating it as unknown would throw the real count away.
    #[test]
    fn merging_with_a_silent_model_keeps_what_is_known() {
        assert_eq!(LmUsage::merge(None, usage(3, 4)), usage(3, 4));
        assert_eq!(LmUsage::merge(usage(3, 4), None), usage(3, 4));
        assert_eq!(LmUsage::merge(None, None), None);
    }
}
