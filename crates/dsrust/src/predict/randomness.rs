//! dspy's rule that asking for several completions at a near-zero temperature sends 0.7 instead.
//!
//! `predict.py::_forward_preprocess`, three lines and a comment saying "to keep randomness":
//!
//! ```python
//! temperature = config.get("temperature") or lm.kwargs.get("temperature")
//! num_generations = config.get("n") or lm.kwargs.get("n") or lm.kwargs.get("num_generations") or 1
//! if (temperature is None or temperature <= 0.15) and num_generations > 1:
//!     config["temperature"] = 0.7
//! ```
//!
//! Neither chain means what it looks like, because `or` is Python's and **0.0 is falsy**. A caller
//! who asks for temperature 0.0 has, as far as this rule can see, asked for nothing: the value
//! falls through to the model's. So `temperature=0.0, n=2` on a model with no temperature of its
//! own resolves to 0.7 — the caller's explicit zero is overwritten — while the same call under a
//! model set to 0.9 reads 0.9, does not fire, and sends the 0.0 after all.
//!
//! Both fields resolve *through* the model, which is why this needs
//! [`ChatModel::defaults`](crate::lm::ChatModel::defaults): a `Predict` that sets neither still has
//! to know what its model would send. `num_generations` is the third arm and has no field of its
//! own here or upstream — `dspy.LM(num_generations=3)` keeps it as a plain kwarg, so it is read
//! out of [`LmConfig::extensions`](crate::lm::api::LmConfig::extensions), where an unrecognised
//! keyword lands.
//!
//! Held to `predict/completion_temperature.json`, 110 pairs of model and module settings recorded
//! by calling `_forward_preprocess` itself.

use crate::lm::Sampling;
use crate::lm::api::LmConfig;

/// What the rule writes when it fires.
const RANDOM_ENOUGH: f64 = 0.7;
/// At or below this, a temperature counts as too low to sample several completions with.
const TOO_COLD: f64 = 0.15;

/// This module's sampling, with the temperature dspy would have raised.
pub(super) fn for_completions(config: &Sampling, model: &LmConfig) -> Sampling {
    if completions(config, model) <= 1 {
        return config.clone();
    }
    match temperature(config, model) {
        Some(set) if set > TOO_COLD => config.clone(),
        _ => Sampling {
            temperature: Some(RANDOM_ENOUGH),
            ..config.clone()
        },
    }
}

/// `config.get("temperature") or lm.kwargs.get("temperature")`, zero falling through as falsy.
fn temperature(config: &Sampling, model: &LmConfig) -> Option<f64> {
    match config.temperature {
        Some(set) if set != 0.0 => Some(set),
        _ => model.temperature,
    }
}

/// The `n` chain, ending at Python's `or 1`. A zero is falsy in every arm, as a `None` is.
fn completions(config: &Sampling, model: &LmConfig) -> u32 {
    let asked = [
        config.completions,
        model.n,
        model
            .extensions
            .get("num_generations")
            .and_then(serde_json::Value::as_u64)
            .map(|count| count as u32),
    ];
    asked
        .into_iter()
        .flatten()
        .find(|&count| count != 0)
        .unwrap_or(1)
}
