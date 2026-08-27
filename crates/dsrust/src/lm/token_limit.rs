//! Which field carries the generation cap on an OpenAI-shaped request.
//!
//! OpenAI's reasoning families reject `max_tokens` outright and require `max_completion_tokens` in
//! its place. Every other OpenAI model, and every other service speaking the same wire format,
//! takes `max_tokens`.
//!
//! Which names those families cover is [`reasoning_model::on_the_wire`], dspy's own predicate for
//! this decision. The rule here is only the second half: *whether the endpoint applies it at all*,
//! which litellm settles by provider — its `ProviderConfigManager` hands the call to a
//! reasoning-model config only when the provider is OpenAI itself.
//!
//! [`reasoning_model::on_the_wire`]: super::reasoning_model::on_the_wire

use super::reasoning_model::on_the_wire;

/// The JSON key a generation cap travels under.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenLimitField {
    /// What every OpenAI-shaped service accepts, and what OpenAI's chat models take.
    MaxTokens,
    /// What OpenAI's reasoning families require instead.
    MaxCompletionTokens,
}

impl TokenLimitField {
    /// The key itself, as it appears in the request body.
    pub fn wire_name(self) -> &'static str {
        match self {
            TokenLimitField::MaxTokens => "max_tokens",
            TokenLimitField::MaxCompletionTokens => "max_completion_tokens",
        }
    }
}

/// How an endpoint chooses between the two fields.
///
/// The rule belongs to the endpoint rather than to the model, because one model name means
/// different things at different hosts: OpenRouter serves `openai/gpt-5` on the `max_tokens`
/// envelope it has always accepted, while OpenAI itself refuses that same request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenLimitRule {
    /// `max_tokens` for every model. OpenRouter, Groq, Together, vLLM and LM Studio all
    /// take it, and none of them treat OpenAI's reasoning-model names as special.
    AlwaysMaxTokens,
    /// OpenAI's own rule: `max_completion_tokens` for the reasoning families, `max_tokens`
    /// for the rest.
    ByOpenAiModelFamily,
}

impl TokenLimitRule {
    /// The field `model` travels under. This is the whole of the decision, so it is worth
    /// asserting on its own rather than only through an assembled request.
    pub fn field_for(self, model: &str) -> TokenLimitField {
        match self {
            TokenLimitRule::AlwaysMaxTokens => TokenLimitField::MaxTokens,
            TokenLimitRule::ByOpenAiModelFamily if on_the_wire(model) => {
                TokenLimitField::MaxCompletionTokens
            }
            TokenLimitRule::ByOpenAiModelFamily => TokenLimitField::MaxTokens,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Which names are reasoning models is `reasoning_model::on_the_wire`, held name by name
    /// against dspy in `lm_api/reasoning_models.json`. What this file decides is whether the
    /// endpoint asks that question at all.
    #[test]
    fn the_openai_rule_asks_the_family_and_the_always_rule_does_not() {
        for model in ["o3", "gpt-5", "openai/o1-mini"] {
            assert_eq!(
                TokenLimitRule::ByOpenAiModelFamily.field_for(model),
                TokenLimitField::MaxCompletionTokens,
                "{model} is a reasoning model at OpenAI"
            );
            assert_eq!(
                TokenLimitRule::AlwaysMaxTokens.field_for(model),
                TokenLimitField::MaxTokens,
                "{model} is unchanged away from OpenAI"
            );
        }
    }

    /// What keeps an OpenRouter-hosted `openai/gpt-5` on the envelope OpenRouter accepts: the two
    /// rules answer differently for the same name, which is why the rule belongs to the endpoint.
    #[test]
    fn the_two_rules_disagree_about_a_reasoning_model() {
        let model = "openai/gpt-5";
        assert_ne!(
            TokenLimitRule::ByOpenAiModelFamily.field_for(model),
            TokenLimitRule::AlwaysMaxTokens.field_for(model)
        );
    }

    #[test]
    fn a_chat_model_takes_max_tokens_under_either_rule() {
        for rule in [
            TokenLimitRule::ByOpenAiModelFamily,
            TokenLimitRule::AlwaysMaxTokens,
        ] {
            assert_eq!(rule.field_for("gpt-4o-mini"), TokenLimitField::MaxTokens);
        }
    }

    #[test]
    fn each_field_names_the_key_it_travels_under() {
        assert_eq!(TokenLimitField::MaxTokens.wire_name(), "max_tokens");
        assert_eq!(
            TokenLimitField::MaxCompletionTokens.wire_name(),
            "max_completion_tokens"
        );
    }
}
