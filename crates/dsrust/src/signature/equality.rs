//! dspy's `Signature.equals`, which is not structural equality and is asked a real question by
//! GEPA: which trace entries belong to the component being reflected on.

use std::collections::BTreeMap;

use super::{Signature, infer_prefix};

impl Signature {
    /// dspy's `Signature.equals`: the same instructions, the same field names, and the same
    /// `json_schema_extra` on each — which is the side, the prefix and the description.
    ///
    /// **The annotation is deliberately not compared.** Upstream walks `json_schema_extra`, which
    /// carries `__dspy_field_type`, `prefix` and `desc` and not the type, so two signatures whose
    /// fields are typed differently still count as equal. Derived `PartialEq` compares the kind,
    /// the closed set and the constraints as well, which is *stricter* — this is what GEPA's trace
    /// matching asks for, and asking the stricter question there pools fewer instances than dspy
    /// pools.
    ///
    /// The prefix compares resolved, since dspy stores the inferred one rather than a `None`.
    pub fn equals(&self, other: &Signature) -> bool {
        if self.instructions != other.instructions {
            return false;
        }
        let extras = |signature: &Signature| -> BTreeMap<String, (bool, String, String)> {
            let inputs = signature.inputs.iter().map(|field| {
                (
                    field.name.clone(),
                    (
                        true,
                        resolved_prefix(&field.name, &field.prefix),
                        field.desc.clone(),
                    ),
                )
            });
            let outputs = signature.outputs.iter().map(|field| {
                (
                    field.name.clone(),
                    (
                        false,
                        resolved_prefix(&field.name, &field.prefix),
                        field.desc.clone(),
                    ),
                )
            });
            inputs.chain(outputs).collect()
        };
        extras(self) == extras(other)
    }
}

/// The prefix a field compares as: whatever was set, or the one dspy infers from the name — which
/// is what upstream stores, so a `None` here and an inferred prefix there are the same field.
fn resolved_prefix(name: &str, prefix: &Option<String>) -> String {
    prefix.clone().unwrap_or_else(|| infer_prefix(name))
}
