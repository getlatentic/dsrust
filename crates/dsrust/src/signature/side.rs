//! Which half of a signature a field belongs to, and the edit that changes one.
//!
//! `Side` is dspy's `__dspy_field_type`, which upstream reads off the FieldInfo extra because
//! `InputField` and `OutputField` build the same pydantic object. `FieldEdit` is the mutation half
//! of `with_updated_fields`, public because optimizers hold it.

use super::{FieldKind, InField, OutField};

/// A field being added, carrying which side of the signature it belongs to.
///
/// dspy reads that off the field's own `__dspy_field_type`, because a Python `InputField` and an
/// `OutputField` are the same class with a marker. Rust has two types, so the side is the type —
/// Which side a field is added to, when a signature grows one.
///
/// dspy takes `(name, field, type_)` as three loose arguments that have to agree — an `InputField`
/// passed with an output's name is a runtime error there. Here the side *is* the type, so
/// [`Signature::append`](crate::Signature::append) cannot be given a mismatched trio:
///
/// ```
/// use dsrust::signature::{InField, Side};
///
/// let signature: dsrust::Signature = "question -> answer".parse().expect("parses");
/// let grown = signature.append(Side::Input(InField {
///     name: "context".to_owned(),
///     desc: "Passages to answer from.".to_owned(),
///     ..Default::default()
/// }));
/// assert_eq!(grown.inputs.len(), 2, "appended last among its own side");
/// assert_eq!(grown.outputs.len(), 1, "the other side is untouched");
/// ```
/// this is what lets one `insert` take either without a runtime look-up.
#[derive(Debug, Clone, PartialEq)]
pub enum Side {
    Input(InField),
    Output(OutField),
}

impl Side {
    /// What upstream's error message calls this side.
    pub(super) fn side_name(&self) -> &'static str {
        match self {
            Side::Input(_) => "input",
            Side::Output(_) => "output",
        }
    }
}

impl From<InField> for Side {
    fn from(field: InField) -> Self {
        Side::Input(field)
    }
}

impl From<OutField> for Side {
    fn from(field: OutField) -> Self {
        Side::Output(field)
    }
}

/// One field of a signature, handed to a caller editing it in place.
#[derive(Debug)]
pub enum FieldEdit<'a> {
    Input(&'a mut InField),
    Output(&'a mut OutField),
}

impl FieldEdit<'_> {
    /// The field's description, whichever side it is on.
    pub fn set_desc(&mut self, desc: impl Into<String>) {
        match self {
            FieldEdit::Input(field) => field.desc = desc.into(),
            FieldEdit::Output(field) => field.desc = desc.into(),
        }
    }

    /// The field's kind, whichever side it is on — upstream's `type_` argument.
    pub fn set_kind(&mut self, kind: FieldKind) {
        match self {
            FieldEdit::Input(field) => field.kind = kind,
            FieldEdit::Output(field) => field.kind = kind,
        }
    }
}
