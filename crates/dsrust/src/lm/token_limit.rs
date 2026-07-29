//! Which field carries the generation cap on an OpenAI-shaped request.
//!
//! OpenAI's reasoning families reject `max_tokens` outright and require
//! `max_completion_tokens` in its place. Every other OpenAI model, and every other service
//! speaking the same wire format, takes `max_tokens`. litellm settles this the same way:
//! its `ProviderConfigManager` hands the call to a reasoning-model config only when the
//! provider is OpenAI itself, and that config renames the field by model family.
//!
//! The families are matched on the name alone. litellm also requires the name to appear in
//! its bundled model registry, which costs it `o1-mini` and `o1-preview` — both reasoning
//! models that do reject `max_tokens` — for as long as that registry lags. Matching on the
//! shape of the name keeps those two right without vendoring a copy of the registry.

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
            TokenLimitRule::ByOpenAiModelFamily if is_reasoning_model(model) => {
                TokenLimitField::MaxCompletionTokens
            }
            TokenLimitRule::ByOpenAiModelFamily => TokenLimitField::MaxTokens,
        }
    }
}

/// Whether `model` names one of OpenAI's reasoning families.
///
/// The name is all that is available at request time, and it is what OpenAI's own families
/// are told apart by. A vendor-prefixed id — `openai/o3` — names the same model as the bare
/// one, so only the last segment is examined.
fn is_reasoning_model(model: &str) -> bool {
    let name = model
        .rsplit_once('/')
        .map_or(model, |(_, last)| last)
        .to_ascii_lowercase();
    is_o_series(&name) || is_gpt_5_reasoning(&name)
}

/// `o1`, `o3-mini`, `o4-mini-2025-04-16`: an `o`, a version number, then an optional
/// suffix. Requiring the digit keeps `omni-moderation` out, and requiring the boundary
/// after it keeps the `o200k` tokenizer names out.
fn is_o_series(name: &str) -> bool {
    let Some(rest) = name.strip_prefix('o') else {
        return false;
    };
    let tail = rest.trim_start_matches(|c: char| c.is_ascii_digit());
    tail.len() < rest.len() && (tail.is_empty() || tail.starts_with('-'))
}

/// Every `gpt-5` variant reasons except the `gpt-5-chat` line, which is the family's plain
/// chat model. A minor version between the two — `gpt-5.1-chat` — reasons again, so the
/// carve-out is anchored at the start of the name rather than matched anywhere in it.
fn is_gpt_5_reasoning(name: &str) -> bool {
    name.contains("gpt-5") && !name.starts_with("gpt-5-chat")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn field(model: &str) -> TokenLimitField {
        TokenLimitRule::ByOpenAiModelFamily.field_for(model)
    }

    #[test]
    fn the_reasoning_families_take_max_completion_tokens() {
        for model in [
            "o1",
            "o1-mini",
            "o1-preview",
            "o1-pro",
            "o3",
            "o3-mini",
            "o3-mini-2025-01-31",
            "o4-mini",
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-nano",
            "gpt-5-codex",
            "gpt-5.1",
        ] {
            assert_eq!(
                field(model),
                TokenLimitField::MaxCompletionTokens,
                "{model} is a reasoning model"
            );
        }
    }

    #[test]
    fn the_chat_models_keep_max_tokens() {
        for model in [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4.1",
            "gpt-3.5-turbo",
            "chatgpt-4o-latest",
            "llama-3.3-70b",
            "omni-moderation-latest",
            "o200k-base",
        ] {
            assert_eq!(
                field(model),
                TokenLimitField::MaxTokens,
                "{model} is not a reasoning model"
            );
        }
    }

    /// The one carve-out inside the gpt-5 family, and the one a plain substring test gets
    /// wrong in both directions.
    #[test]
    fn the_gpt_5_chat_line_is_a_chat_model_but_a_versioned_chat_name_still_reasons() {
        assert_eq!(field("gpt-5-chat"), TokenLimitField::MaxTokens);
        assert_eq!(field("gpt-5-chat-latest"), TokenLimitField::MaxTokens);
        assert_eq!(field("gpt-5.1-chat"), TokenLimitField::MaxCompletionTokens);
    }

    #[test]
    fn a_vendor_prefix_names_the_same_model() {
        assert_eq!(field("openai/o3"), TokenLimitField::MaxCompletionTokens);
        assert_eq!(field("openai/gpt-5"), TokenLimitField::MaxCompletionTokens);
        assert_eq!(field("openai/gpt-5-chat"), TokenLimitField::MaxTokens);
        assert_eq!(field("openai/gpt-4o-mini"), TokenLimitField::MaxTokens);
    }

    /// What keeps an OpenRouter-hosted `openai/gpt-5` on the envelope OpenRouter accepts.
    #[test]
    fn the_always_rule_never_looks_at_the_model_name() {
        for model in ["o3", "gpt-5", "openai/gpt-5", "gpt-4o-mini"] {
            assert_eq!(
                TokenLimitRule::AlwaysMaxTokens.field_for(model),
                TokenLimitField::MaxTokens,
                "{model} is unchanged away from OpenAI"
            );
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
