//! What of an `LmConfig` a harness can carry, and what it must refuse.
//!
//! dsrust hands every call a config. `RunTuning` has a destination for part of it,
//! no destination at all for the rest, and an adapter honours only some of what it
//! has a destination for. The one outcome ruled out is answering as though a knob
//! had been applied: a judge pinned to `temperature: 0.0` that quietly samples
//! scores the same program differently on each run, and nothing in the reply says
//! why.
//!
//! The boundary of that claim is worth stating, because one refusal reads like a
//! guarantee it does not give. This is about knobs an adapter cannot honour, and
//! says nothing about how long a run takes or how many of them a caller makes. An
//! optimizer's task model is a bad use of an agent for a reason no config carries:
//! hundreds of calls at agent latency is days. That a pinned temperature happens to
//! refuse the same call is a coincidence of two hazards, not coverage of both.
//!
//! What the sampling refusal does reach is wider than a judge, and is the reason it
//! is written down in the guide as well. dspy's retry-shaped modules re-ask with
//! `Sampling::rollout` — `temperature: 1.0` and a fresh rollout id, which is how
//! attempt two differs from attempt one — so `BestOfN`, `Refine`, a
//! `BootstrapFewShot` past its first round and `InferRules` are all refused over an
//! agent, as are `SIMBA`, `COPRO` and `MIPROv2`'s proposers, which propose at a
//! named temperature. `GEPA` is not: its reflection names no sampling.
//!
//! Letting a rollout through is not the default, since the variation an agent gives
//! is not one this crate can promise and a `BestOfN` drawing three near-identical
//! candidates would look like it worked. But refusing every retry-shaped module is
//! a lot to lose over a knob whose *intent* an agent does satisfy, so the other way
//! is reachable by name: [`Temperature::FromTheAgent`]. A caller who has decided the
//! agent's own variation is the variation they want says so once, on the builder,
//! and the claim is theirs rather than the crate's.
//!
//! It does not extend to `n`. An agent answers once, and no amount of run-to-run
//! variation turns one reply into the ten completions `COPRO` asked for.

use anyhow::{Result, anyhow, bail};
use dsrust::lm::api::LmConfig;
use harness::{Features, ReasoningEffort};

/// What a temperature means to an agent that has none.
///
/// The default refuses, because a `BestOfN` reporting that it sampled at 1.0 over a
/// CLI would be claiming something it did not do. That refusal reaches every
/// retry-shaped module, which is a lot to lose, so the other way is available by
/// name — and only by name, since what it accepts cannot be checked.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Temperature {
    /// A call naming one is refused.
    #[default]
    Refused,
    /// A call naming one runs, and the agent's own run-to-run variation stands in
    /// for the sampling that was asked for.
    ///
    /// What dspy's retry-shaped modules want from `Sampling::rollout` is that
    /// attempt two differs from attempt one, and an agent does vary between runs.
    /// It varies by an amount this crate cannot promise, though: a `BestOfN` may
    /// draw three near-identical candidates and still look like it worked. Choosing
    /// this is accepting that, and it is why it is not the default.
    FromTheAgent,
}

/// The levels `RunTuning::effort` spells. dsrust's own effort is free text, so the
/// mapping is partial in one direction and total in the other.
const LEVELS: [(&str, ReasoningEffort); 4] = [
    ("minimal", ReasoningEffort::Minimal),
    ("low", ReasoningEffort::Low),
    ("medium", ReasoningEffort::Medium),
    ("high", ReasoningEffort::High),
];

/// Refuses the knobs `RunTuning` has nowhere to put.
///
/// Permanent rather than pending: a knob with no destination cannot be honoured by
/// any adapter, now or later — a different statement from one that no adapter
/// honours yet.
///
/// Only what the caller asked for reaches here. A `HarnessModel` reports no
/// defaults of its own, so a `Predict` that names nothing sends a config of all
/// `None` and passes untouched.
///
/// `cache` is the one field deliberately absent from the list. It carries dsrust's
/// own response cache and its rollout counter, neither of which is ever sent to a
/// provider — the cache sits above the model, so there is nothing here to drop.
pub(crate) fn refuse_undeliverable(config: &LmConfig, policy: Temperature) -> Result<()> {
    // Exhaustive on purpose. A field added to `LmConfig` stops this compiling until
    // someone says whether a harness can carry it, which is the one question a new
    // knob answers wrongly by default: nothing reads it, and every call reports
    // success. `registry.rs` in the harness itself holds the same line.
    let LmConfig {
        temperature,
        max_tokens,
        top_p,
        stop,
        n,
        logprobs,
        tool_choice,
        prompt_cache,
        extensions,
        // Carried elsewhere: the schema becomes the run's `output_schema`, and
        // `reasoning`'s other two fields are read by `effort_for` and the thinking
        // cap. Only its summary has nowhere to go.
        reasoning,
        response_format: _,
        // Never reaches a provider: dsrust's response cache sits above the model,
        // and the rollout counter only varies its key.
        cache: _,
    } = config;
    let asked = [
        (
            "temperature",
            temperature.is_some() && policy == Temperature::Refused,
        ),
        ("top_p", top_p.is_some()),
        ("max_tokens", max_tokens.is_some()),
        // Not reached by `Temperature::FromTheAgent`: an agent answers once, so a
        // caller asking for three completions would silently receive one. This is
        // what keeps `COPRO` refused even under the opt-in.
        ("n", n.is_some()),
        ("stop", stop.is_some()),
        ("logprobs", logprobs.is_some()),
        (
            "reasoning.summary",
            reasoning.as_ref().is_some_and(|r| r.summary.is_some()),
        ),
        ("tool_choice", tool_choice.is_some()),
        ("prompt_cache", prompt_cache.is_some()),
        ("extensions", !extensions.is_empty()),
    ];
    let named: Vec<&str> = asked
        .iter()
        .filter_map(|(name, set)| set.then_some(*name))
        .collect();
    if named.is_empty() {
        return Ok(());
    }
    bail!(
        "a coding agent cannot honour {}: the harness has no field to carry that to the model, \
         so the run would answer as though it had been applied. Ask for it from an HTTP \
         provider, or build the model without it.",
        named.join(", ")
    )
}

/// `RunTuning::effort` for this call, or the reason the request's cannot be served.
///
/// Three outcomes rather than two. The level travels; or the adapter has nowhere to
/// put it, which `Features::effort` answers; or the level is not one the harness
/// spells. The third is worth its own error — rounding `"xhigh"` down to `High`
/// answers a question nobody asked, and the caller never learns their setting was
/// approximated.
pub(crate) fn effort_for(
    config: &LmConfig,
    features: &Features,
) -> Result<Option<ReasoningEffort>> {
    let Some(asked) = config.reasoning.as_ref().and_then(|r| r.effort.as_deref()) else {
        return Ok(None);
    };
    if !features.effort {
        bail!(
            "this harness does not honour a reasoning effort, so {asked:?} would be dropped in \
             silence; drop `reasoning.effort` or run against an adapter that advertises it"
        );
    }
    LEVELS
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case(asked))
        .map(|(_, level)| Some(*level))
        .ok_or_else(|| {
            anyhow!(
                "`reasoning.effort` is {asked:?}, which the harness cannot spell: it takes \
                 minimal, low, medium or high"
            )
        })
}
