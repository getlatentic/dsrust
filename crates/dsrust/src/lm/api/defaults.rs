//! dspy's `kwargs = {**self.kwargs, **kwargs}`: an LM's own settings, under one call's.
//!
//! Upstream keeps whatever was passed to `dspy.LM(...)` on the instance and merges it beneath each
//! call's overrides, so `dspy.LM("openai/gpt-4o-mini", temperature=0.5)` applies to every ask
//! through it without a caller repeating themselves. Field by field rather than wholesale, because
//! a call that sets only `max_tokens` must keep the LM's temperature rather than clearing it.

use super::config::LmConfig;

/// Fill anything this call left unset from the LM's own settings.
///
/// One direction only: what the call states wins, and what it says nothing about is inherited. That
/// is `{**lm.kwargs, **call_kwargs}` read the other way round, and it is the same result.
pub fn beneath(call: &mut LmConfig, lm: &LmConfig) {
    fill(&mut call.temperature, &lm.temperature);
    fill(&mut call.max_tokens, &lm.max_tokens);
    fill(&mut call.top_p, &lm.top_p);
    fill(&mut call.stop, &lm.stop);
    fill(&mut call.n, &lm.n);
    fill(&mut call.logprobs, &lm.logprobs);
    fill(&mut call.response_format, &lm.response_format);
    fill(&mut call.reasoning, &lm.reasoning);
    fill(&mut call.tool_choice, &lm.tool_choice);
    fill(&mut call.cache, &lm.cache);
    fill(&mut call.prompt_cache, &lm.prompt_cache);
    // Extensions are provider knobs keyed by name, so the two maps merge rather than replace and
    // the call's entry wins per key — which is what a dict update does to a nested dict it shares.
    for (key, value) in &lm.extensions {
        call.extensions
            .entry(key.clone())
            .or_insert_with(|| value.clone());
    }
}

fn fill<T: Clone>(call: &mut Option<T>, lm: &Option<T>) {
    if call.is_none() {
        *call = lm.clone();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// What the call states wins. Anything else would make an LM's default impossible to override.
    #[test]
    fn one_calls_own_setting_wins() {
        let mut call = LmConfig {
            temperature: Some(0.9),
            ..LmConfig::default()
        };
        let lm = LmConfig {
            temperature: Some(0.1),
            max_tokens: Some(512),
            ..LmConfig::default()
        };
        beneath(&mut call, &lm);
        assert_eq!(call.temperature, Some(0.9), "the call's");
        assert_eq!(
            call.max_tokens,
            Some(512),
            "and the LM's, where the call said nothing"
        );
    }

    /// A call setting one field must not clear the others. This is the difference between merging
    /// field by field and taking whichever config is non-default.
    #[test]
    fn setting_one_field_does_not_clear_the_rest() {
        let mut call = LmConfig {
            max_tokens: Some(64),
            ..LmConfig::default()
        };
        let lm = LmConfig {
            temperature: Some(0.3),
            top_p: Some(0.8),
            stop: Some(vec!["END".to_owned()]),
            ..LmConfig::default()
        };
        beneath(&mut call, &lm);
        assert_eq!(call.max_tokens, Some(64));
        assert_eq!(call.temperature, Some(0.3));
        assert_eq!(call.top_p, Some(0.8));
        assert_eq!(call.stop.as_deref(), Some(["END".to_owned()].as_slice()));
    }

    /// Provider knobs merge per key, as a dict update does, so an LM-wide knob survives a call that
    /// sets a different one.
    #[test]
    fn provider_knobs_merge_per_key() {
        let mut call = LmConfig::default();
        call.extensions.insert("seed".to_owned(), json!(7));
        let mut lm = LmConfig::default();
        lm.extensions.insert("seed".to_owned(), json!(1));
        lm.extensions
            .insert("service_tier".to_owned(), json!("flex"));

        beneath(&mut call, &lm);
        assert_eq!(call.extensions["seed"], json!(7), "the call's");
        assert_eq!(
            call.extensions["service_tier"],
            json!("flex"),
            "and the LM's other knob"
        );
    }
}
