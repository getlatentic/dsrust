//! dspy `_validate_openai_reasoning_temperature`: an OpenAI reasoning model asked to reason takes
//! the default temperature and nothing else.
//!
//! Upstream refuses the call locally rather than letting the provider refuse it, and names both
//! features it could not honour together. Without this the same body goes out and comes back a
//! 400 — the same outcome, one round trip later, described in the provider's words rather than in
//! terms of the two settings that conflict.
//!
//! Called from both request builders, because upstream calls it from both `common_config_kwargs`
//! and `responses_config_kwargs` — the endpoint is the only thing that differs, and it travels into
//! the issue line.
//!
//! Always enforced, which is the pin. Main has since added an `enforce_reasoning_temperature=True`
//! parameter to the Responses path only, so a caller there can opt out; the chat path still always
//! enforces. Worth revisiting when the pin moves — not before, since the pin is what every gate
//! here is measured against.

use crate::lm::api;
use crate::lm::error::{LmErrorKind, LmFailure};

use super::super::token_limit::is_openai_reasoning_model;

/// Refuse the pairing dspy refuses: a reasoning model, a reasoning effort that is not `none`, and a
/// temperature that is neither unset nor `1`.
///
/// `endpoint` is dspy's, and reaches the caller in the issue line: `chat` or `responses`.
pub(super) fn checked(
    config: &api::LmConfig,
    model: &str,
    endpoint: &str,
) -> Result<(), LmFailure> {
    if !is_openai_reasoning_model(model) {
        return Ok(());
    }
    let effort = config
        .reasoning
        .as_ref()
        .and_then(|reasoning| reasoning.effort.as_deref());
    match effort {
        None | Some("none") => return Ok(()),
        Some(_) => {}
    }
    // Upstream's `config.temperature in {None, 1}`: unset is the default and 1.0 *is* the default,
    // so neither conflicts with reasoning.
    match config.temperature {
        None => return Ok(()),
        Some(temperature) if temperature == 1.0 => return Ok(()),
        Some(_) => {}
    }
    let effort = effort.unwrap_or_default();
    let temperature = config.temperature.unwrap_or_default();
    Err(LmFailure {
        kind: LmErrorKind::UnsupportedFeature,
        message: "OpenAI reasoning models only support the default temperature when reasoning \
                  effort is active. Use temperature=None or temperature=1, or set \
                  reasoning_effort='none'."
            .to_owned(),
        model: Some(model.to_owned()),
        provider: Some("openai".to_owned()),
        features: vec!["temperature".to_owned(), "reasoning".to_owned()],
        issues: vec![format!(
            "{endpoint} request used reasoning effort {effort:?} with temperature={temperature:?}."
        )],
        ..LmFailure::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(effort: Option<&str>, temperature: Option<f64>) -> api::LmConfig {
        api::LmConfig {
            temperature,
            reasoning: effort.map(|effort| api::LmReasoningConfig {
                effort: Some(effort.to_owned()),
                ..api::LmReasoningConfig::default()
            }),
            ..api::LmConfig::default()
        }
    }

    /// The one pairing upstream refuses.
    #[test]
    fn a_reasoning_model_reasoning_at_a_chosen_temperature_is_refused() {
        let failed =
            checked(&config(Some("high"), Some(0.7)), "openai/o3", "chat").expect_err("refused");
        assert_eq!(failed.kind, LmErrorKind::UnsupportedFeature);
        assert_eq!(failed.features, ["temperature", "reasoning"]);
        assert_eq!(
            failed.issues,
            [r#"chat request used reasoning effort "high" with temperature=0.7."#]
        );
    }

    /// Every arm that lets the call through, each for its own reason: a model that does not reason,
    /// reasoning switched off, and the two temperatures that are the default.
    #[test]
    fn everything_else_goes_through() {
        let allowed = [
            (config(Some("high"), Some(0.7)), "openai/gpt-4o-mini"),
            (config(None, Some(0.7)), "openai/o3"),
            (config(Some("none"), Some(0.7)), "openai/o3"),
            (config(Some("high"), None), "openai/o3"),
            (config(Some("high"), Some(1.0)), "openai/o3"),
        ];
        for (config, model) in allowed {
            assert!(
                checked(&config, model, "chat").is_ok(),
                "expected {model} to be allowed"
            );
        }
    }

    /// The endpoint reaches the issue line, which is how a reader knows which wire refused.
    #[test]
    fn the_endpoint_is_named_in_the_issue() {
        let failed = checked(&config(Some("low"), Some(0.2)), "openai/gpt-5", "responses")
            .expect_err("refused");
        assert!(
            failed.issues[0].starts_with("responses request used"),
            "got {:?}",
            failed.issues
        );
    }
}
