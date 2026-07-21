//! One call to a model: what was asked, and what came back.
//!
//! dspy is normalising its LM API onto a request and a response value in place of loose keyword
//! arguments and a bare string — opt-in at 3.3, default at 3.5, the legacy shape gone by 4.0.
//! Following that rather than the shape it is removing is what lets one call differ from the one
//! before it, and it is what gives a caller somewhere to read a reply's cost from.

use serde_json::{Value, json};

use super::{ChatTurn, OutputMode};

/// How a model should sample its reply.
///
/// These belong to one call rather than to a model: the same model answers twice and the two
/// attempts differ only here. That is what lets `BestOfN` mean anything and what lets a bootstrap
/// round after the first not repeat itself.
///
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Sampling {
    /// Unset leaves each provider on the default it is already sent.
    pub temperature: Option<f64>,
    pub max_tokens: Option<u32>,
    /// How many completions to ask for, upstream's `n`. They arrive in
    /// [`LmResponse::outputs`]; only the OpenAI-shaped services take the field, so the others
    /// answer once however many are asked for.
    pub completions: Option<u32>,
    /// Varied to miss the response cache, so a second attempt is answered rather than replayed.
    ///
    /// Never sent to a provider — it is part of the cache key and nothing else, which is exactly
    /// what upstream does with it. Two requests alike but for this one number are two different
    /// cache entries and therefore two real calls, and that is the whole mechanism behind
    /// `BestOfN` and behind a bootstrap round after the first.
    pub rollout_id: Option<u64>,
}

impl Sampling {
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

    /// What two identical calls share, and what [`Sampling::rollout_id`] exists to break.
    ///
    /// Everything the provider is sent is in here, because anything left out would let one call
    /// be answered with another's reply — `model` included, since the store is shared across
    /// every model in the process. `rollout_id` is in here and is *not* sent, which is the whole
    /// of what it does: it changes this string and nothing else.
    ///
    /// Credentials are deliberately absent, matching upstream's `ignored_args_for_cache_key`:
    /// rotating a key does not change what a model answers, and a key has no business in a
    /// map that outlives the call.
    /// Hashed rather than kept whole, which is what makes an entry nameable as a file and keeps
    /// a long conversation from being held twice — once as a reply and once as its own key.
    /// Upstream hashes the same way, `sha256(orjson.dumps(params, sort_keys))`.
    pub fn cache_key(&self, model: &str) -> String {
        use sha2::Digest;
        let digest = sha2::Sha256::digest(self.cache_identity(model).as_bytes());
        digest.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    /// Everything two calls must share to be the same call, as JSON.
    fn cache_identity(&self, model: &str) -> String {
        let turns: Vec<Value> = self
            .turns
            .iter()
            .map(|turn| json!({ "role": turn.role.as_str(), "content": turn.content }))
            .collect();
        json!({
            "model": model,
            "system": self.system,
            "turns": turns,
            "schema": match self.mode {
                OutputMode::Text => Value::Null,
                OutputMode::Json { schema } => schema.clone(),
            },
            "temperature": self.sampling.temperature,
            "max_tokens": self.sampling.max_tokens,
            "n": self.sampling.completions,
            "rollout_id": self.sampling.rollout_id,
        })
        .to_string()
    }
}

/// What one call cost, in the two counts every provider agrees on.
///
/// Each of the three names them differently — Anthropic `input_tokens`/`output_tokens`, the
/// OpenAI-shaped services `prompt_tokens`/`completion_tokens`, ollama `prompt_eval_count`/
/// `eval_count` — so normalising them here is the whole reason a caller can compare the cost of
/// two adapters without knowing who answered.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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

/// What a model answered with. dspy's `LMResponse`.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LmResponse {
    /// Every completion the provider returned — as many as [`Sampling::completions`] asked for,
    /// and one when it asked for nothing.
    ///
    /// `BestOfN` reads this rather than re-asking, since one request for several is one round
    /// trip where several requests are several.
    pub outputs: Vec<String>,
    /// Absent when a provider reported none, which a caller must not read as free.
    pub usage: Option<Usage>,
    /// Whether this was replayed from the cache rather than generated. A replay is not billed,
    /// so a hit carries the usage the original call reported and costs nothing again.
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
            outputs: vec![text.into()],
            ..Self::default()
        }
    }

    /// Several completions from one request, in the order the provider returned them.
    pub fn completions(outputs: impl IntoIterator<Item = String>) -> Self {
        Self {
            outputs: outputs.into_iter().collect(),
            ..Self::default()
        }
    }

    /// The completion an adapter parses: the first, which is the only one unless several were
    /// asked for. Empty when a provider answered with no choices at all, which every caller here
    /// treats as an unparseable reply rather than a panic.
    pub fn text_ref(&self) -> &str {
        self.outputs.first().map_or("", String::as_str)
    }

    /// The first completion, taken by value.
    pub fn into_text(self) -> String {
        self.outputs.into_iter().next().unwrap_or_default()
    }

    /// What this call actually cost, which is nothing when it was replayed.
    ///
    /// [`usage`](Self::usage) stays readable on a hit — it is what the answer was worth — but a
    /// replay is not billed, so anything totalling spend reads this instead. dspy draws the same
    /// line by skipping its usage tracker when `cache_hit` is set.
    pub fn spend(&self) -> Option<Usage> {
        self.usage.filter(|_| !self.cache_hit)
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
        assert_eq!(scripted.text_ref(), "the reply");
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
