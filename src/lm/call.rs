//! One call to a model: what was asked, and what came back.
//!
//! dspy is normalising its LM API onto a request and a response value in place of loose keyword
//! arguments and a bare string — opt-in at 3.3, default at 3.5, the legacy shape gone by 4.0.
//! Following that rather than the shape it is removing is what lets one call differ from the one
//! before it, and it is what gives a caller somewhere to read a reply's cost from.

use serde_json::Value;

use super::{ChatTurn, OutputMode};

/// How a model should sample its reply.
///
/// These belong to one call rather than to a model: the same model answers twice and the two
/// attempts differ only here. That is what lets `BestOfN` mean anything and what lets a bootstrap
/// round after the first not repeat itself.
///
/// Two of upstream's fields are deliberately absent, both because this seam cannot carry them.
/// `n`: [`ChatModel::chat`](super::ChatModel::chat) answers with one completion, so asking for
/// several could only ever be billed and then discarded — asking for many is a change of return
/// type, not a field. `rollout_id`: upstream varies it to miss *its own* response cache and drops
/// it before the provider call, so it never reaches a wire. There is no cache here, which leaves
/// nothing for it to change; what makes a re-ask differ is `temperature`. It earns a field when a
/// cache lands and needs a key to vary.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Sampling {
    /// Unset leaves each provider on the default it is already sent.
    pub temperature: Option<f64>,
    pub max_tokens: Option<u32>,
}

/// One call: what to say, what shape to say it in, and how to sample the reply.
///
/// dspy's `LMRequest`. A request travelling as a value is also why no ambient state is needed to
/// override a model for one call — `dspy.context` scopes a ContextVar to do the same thing.
pub struct LmRequest<'a> {
    pub system: &'a str,
    pub turns: &'a [ChatTurn],
    pub mode: OutputMode<'a>,
    pub sampling: Sampling,
}

impl<'a> LmRequest<'a> {
    /// One call at the provider's own defaults.
    pub fn new(system: &'a str, turns: &'a [ChatTurn], mode: OutputMode<'a>) -> Self {
        Self {
            system,
            turns,
            mode,
            sampling: Sampling::default(),
        }
    }

    pub fn sampled(mut self, sampling: Sampling) -> Self {
        self.sampling = sampling;
        self
    }
}

/// What one call cost, in the two counts every provider agrees on.
///
/// Each of the three names them differently — Anthropic `input_tokens`/`output_tokens`, the
/// OpenAI-shaped services `prompt_tokens`/`completion_tokens`, ollama `prompt_eval_count`/
/// `eval_count` — so normalising them here is the whole reason a caller can compare the cost of
/// two adapters without knowing who answered.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

impl Usage {
    pub fn total(self) -> u32 {
        self.input_tokens + self.output_tokens
    }

    /// What several calls cost together, which is what one `forward` costs when a fallback or a
    /// feedback retry made more than one.
    ///
    /// A model reporting nothing stays nothing rather than becoming zero, so a scripted model in
    /// the chain cannot make a real count read as free — and one real count among several silent
    /// ones is still reported, because it is what is known rather than the whole.
    pub fn merge(left: Option<Self>, right: Option<Self>) -> Option<Self> {
        match (left, right) {
            (None, other) | (other, None) => other,
            (Some(left), Some(right)) => Some(Self {
                input_tokens: left.input_tokens + right.input_tokens,
                output_tokens: left.output_tokens + right.output_tokens,
            }),
        }
    }
}

/// What a model answered with.
///
/// dspy's `LMResponse`. Upstream's `outputs` has no field here: it holds the *parsed* fields, which
/// an adapter produces from `text`, and this seam sits below every adapter — a field at this level
/// could only ever be empty. [`Predict`](crate::Predict) is where text becomes fields.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LmResponse {
    /// The completion itself, which is what an adapter parses.
    pub text: String,
    /// Absent when a provider reported none, which a caller must not read as free.
    pub usage: Option<Usage>,
    /// Whether the reply was replayed rather than generated. Nothing caches replies yet, so this
    /// is false throughout — it reports a fact rather than asking a caller to supply one, which is
    /// why it is here while `rollout_id` is not.
    pub cache_hit: bool,
    /// What the provider said that this crate does not model — a stop reason, a fingerprint, a
    /// filter verdict. Kept whole rather than picked over, so reading a new one needs no release.
    pub provider_data: Option<Value>,
}

impl LmResponse {
    /// A reply carrying nothing but its text, for a model that reports no cost: the scripted ones
    /// tests install, and any provider that omits a usage block.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            ..Self::default()
        }
    }

    pub fn with_usage(mut self, usage: Option<Usage>) -> Self {
        self.usage = usage;
        self
    }

    pub fn with_provider_data(mut self, provider_data: Option<Value>) -> Self {
        self.provider_data = provider_data;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_reply_with_no_reported_cost_is_absent_rather_than_zero() {
        let scripted = LmResponse::text("the reply");
        assert_eq!(scripted.text, "the reply");
        assert_eq!(scripted.usage, None, "nothing is not the same as free");
        assert!(!scripted.cache_hit);
    }

    #[test]
    fn usage_totals_the_two_counts() {
        let usage = Usage {
            input_tokens: 12,
            output_tokens: 30,
        };
        assert_eq!(usage.total(), 42);
    }

    fn usage(input_tokens: u32, output_tokens: u32) -> Option<Usage> {
        Some(Usage {
            input_tokens,
            output_tokens,
        })
    }

    #[test]
    fn merging_adds_both_counts() {
        assert_eq!(
            Usage::merge(usage(1, 2), usage(10, 20)),
            usage(11, 22),
            "a fallback ask costs what both asks cost"
        );
    }

    /// The trap: treating a silent model as zero would turn one real count into a total that
    /// looks complete, and treating it as unknown would throw the real count away.
    #[test]
    fn merging_with_a_silent_model_keeps_what_is_known() {
        assert_eq!(Usage::merge(None, usage(3, 4)), usage(3, 4));
        assert_eq!(Usage::merge(usage(3, 4), None), usage(3, 4));
        assert_eq!(Usage::merge(None, None), None);
    }
}
