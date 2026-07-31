//! Letting Anthropic cite in its own channel, rather than asking the prompt for a `citations` field.
//!
//! dspy's `Citations.adapt_to_native_lm_feature` is one line and it is not a formatting preference:
//!
//! ```python
//! if lm.model.startswith("anthropic/"):
//!     return signature.delete(field_name)
//! return signature
//! ```
//!
//! So for an Anthropic model the field comes *out* of the rendered signature entirely — the prompt
//! never mentions it — and `Citations.parse_lm_response` fills it afterwards from the citations the
//! provider attached to its own text blocks. On every other provider the field renders as usual and
//! the model is asked for it in prose.
//!
//! Both halves matter and only together: deleting the field without reading the channel answers
//! nothing, and reading the channel without deleting the field asks twice.

use crate::signature::{FieldKind, Signature};

/// What native citations change about a request: the field that left the render, and the signature
/// to render once it has.
#[derive(Debug, Clone, PartialEq)]
pub struct NativeCitations {
    /// The output field the provider's own channel will fill.
    pub field: String,
    pub signature: Signature,
}

/// The first `Citations` output field's name, if the signature declares one.
///
/// A `Citations` output is a structured field whose annotation names the type — this crate has no
/// `FieldKind::Citations`, because upstream has no special *rendering* for it either; what makes it
/// special is only this native path.
pub fn citations_output_field(signature: &Signature) -> Option<&str> {
    signature
        .outputs
        .iter()
        .find(|field| match &field.kind {
            FieldKind::Json(json) => json.annotation == "Citations",
            _ => false,
        })
        .map(|field| field.name.as_str())
}

/// The signature to render, once a `Citations` output has left it — or `None` to render it as usual.
///
/// `usable` is dspy's `lm.model.startswith("anthropic/")`, asked of the model rather than computed
/// here: which providers answer with citations is the model's own fact, the same way
/// [`native_reasoning_usable`](crate::lm::ChatModel::native_reasoning_usable) is.
pub fn plan(signature: &Signature, usable: bool) -> Option<NativeCitations> {
    let field = citations_output_field(signature)?.to_owned();
    if !usable {
        return None;
    }
    Some(NativeCitations {
        signature: signature.delete(&field),
        field,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signature::{JsonType, OutField, Signature};

    fn signature_with(kind: FieldKind) -> Signature {
        Signature::single_input(
            "Answer.",
            vec![
                OutField {
                    name: "answer".into(),
                    ..Default::default()
                },
                OutField {
                    name: "citations".into(),
                    kind,
                    ..Default::default()
                },
            ],
        )
    }

    /// An Anthropic model takes the field out of the render, which is the byte that moves.
    #[test]
    fn an_anthropic_model_removes_the_field_from_the_render() {
        let signature = signature_with(FieldKind::Json(JsonType::plain("Citations")));
        let planned = plan(&signature, true).expect("a Citations output on Anthropic");

        assert_eq!(planned.field, "citations");
        assert!(
            planned
                .signature
                .outputs
                .iter()
                .all(|f| f.name != "citations"),
            "the field should have left the rendered signature"
        );
        assert_eq!(planned.signature.outputs.len(), 1);
    }

    /// Every other provider renders it, because upstream's carve-out is anthropic-only.
    #[test]
    fn any_other_provider_renders_the_field() {
        let signature = signature_with(FieldKind::Json(JsonType::plain("Citations")));
        assert_eq!(plan(&signature, false), None);
    }

    /// A signature with no Citations output is untouched whatever the provider — the plan is keyed
    /// on the field, not on the model.
    #[test]
    fn a_signature_without_citations_is_untouched() {
        let signature = signature_with(FieldKind::Json(JsonType::plain("dict[str, Any]")));
        assert_eq!(plan(&signature, true), None);
        assert_eq!(plan(&signature, false), None);
    }
}
