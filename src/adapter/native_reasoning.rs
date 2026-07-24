//! Asking a reasoning model to think in its own channel, rather than in a rendered field.
//!
//! dspy's `Reasoning.adapt_to_native_lm_feature`: when the model exposes extended thinking and the
//! signature declares a `Reasoning` output, the field leaves the render and the request carries a
//! `reasoning_effort` instead — the model reasons natively and its thinking comes back on its own
//! channel, not as a `[[ ## reasoning ## ]]` block the prompt asked for.

use crate::lm::Capabilities;
use crate::signature::{FieldKind, Signature};

/// dspy's `reasoning_effort` as one call carries it.
///
/// `Unset` is a call that says nothing, so native reasoning turns on wherever the model and the
/// signature allow it; `Off` is a caller's explicit `None`, disabling it; `Level` names the budget.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum ReasoningEffort {
    #[default]
    Unset,
    Off,
    Level(String),
}

/// What native reasoning changes about a request: the effort it carries, and the signature to
/// render once the `Reasoning` output has left it.
#[derive(Debug, Clone, PartialEq)]
pub struct NativeReasoning {
    pub effort: String,
    pub signature: Signature,
}

/// The first `Reasoning` output field's name, if the signature declares one. dspy adapts each
/// native-response type in turn; `Reasoning` is the one this crate models.
pub fn reasoning_output_field(signature: &Signature) -> Option<&str> {
    signature
        .outputs
        .iter()
        .find(|field| matches!(field.kind, FieldKind::Reasoning))
        .map(|field| field.name.as_str())
}

/// What the request should carry for native reasoning, or `None` to render the field as usual.
///
/// dspy `adapt_to_native_lm_feature`: the effort is the caller's when set, otherwise `"low"` — the
/// default that turns native reasoning on for a signature that asked for it. A caller's explicit
/// `Off`, or a model that cannot reason, leaves the field in the render and sends no effort.
pub fn plan(
    signature: &Signature,
    capabilities: Capabilities,
    effort: &ReasoningEffort,
) -> Option<NativeReasoning> {
    let field = reasoning_output_field(signature)?;
    let effort = match effort {
        ReasoningEffort::Off => return None,
        ReasoningEffort::Level(level) => level.clone(),
        ReasoningEffort::Unset => "low".to_owned(),
    };
    if !capabilities.reasoning {
        return None;
    }
    let field = field.to_owned();
    Some(NativeReasoning { effort, signature: signature.delete(&field) })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signature::{OutField, Signature};

    fn reasoning_signature() -> Signature {
        let mut signature = Signature::single_input(
            "Answer.",
            vec![OutField { name: "answer".into(), ..Default::default() }],
        );
        signature.outputs.insert(
            0,
            OutField { name: "reasoning".into(), kind: FieldKind::Reasoning, ..Default::default() },
        );
        signature
    }

    fn able() -> Capabilities {
        Capabilities { reasoning: true, ..Default::default() }
    }

    /// A reasoning model with a `Reasoning` field: the field leaves the render and the request
    /// carries the default `"low"` effort.
    #[test]
    fn a_reasoning_field_moves_onto_the_request_at_the_default_effort() {
        let planned =
            plan(&reasoning_signature(), able(), &ReasoningEffort::Unset).expect("native reasoning");
        assert_eq!(planned.effort, "low");
        let names: Vec<&str> = planned.signature.outputs.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(names, ["answer"], "the reasoning field is gone from the render");
    }

    /// A named level is carried as stated.
    #[test]
    fn a_named_effort_is_carried_as_given() {
        let planned = plan(&reasoning_signature(), able(), &ReasoningEffort::Level("high".into()))
            .expect("native reasoning");
        assert_eq!(planned.effort, "high");
    }

    /// A caller's explicit `Off` leaves the field in the render — dspy's `reasoning_effort=None`.
    #[test]
    fn an_off_effort_leaves_the_field_rendered() {
        assert_eq!(plan(&reasoning_signature(), able(), &ReasoningEffort::Off), None);
    }

    /// A model that cannot reason renders the field however the effort is set.
    #[test]
    fn a_model_that_cannot_reason_renders_the_field() {
        assert_eq!(
            plan(&reasoning_signature(), Capabilities::default(), &ReasoningEffort::Unset),
            None
        );
    }

    /// A signature with no reasoning field is left alone.
    #[test]
    fn a_signature_with_no_reasoning_field_is_left_alone() {
        let plain = Signature::single_input("Answer.", vec![OutField::default()]);
        assert_eq!(plan(&plain, able(), &ReasoningEffort::Unset), None);
    }
}
