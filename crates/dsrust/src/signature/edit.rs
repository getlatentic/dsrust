//! dspy `Signature`'s editing API: the same fields under a different shape.
//!
//! Upstream builds a signature and then edits it, and programs that decide their fields per
//! request do nothing else — `delete` a field, `append` another, restate the objective. These are
//! the calls that make that possible, kept together because they share one rule: each answers with
//! a new signature and leaves the one it was given alone.

use super::{FieldEdit, Side, Signature, instructions};

impl Signature {
    /// dspy `Signature.delete`: this signature without the named field, from whichever side
    /// holds it. A name that is not there leaves the signature unchanged, as upstream has it —
    /// deleting a field an adapter only sometimes adds should not depend on whether it did.
    pub fn delete(&self, name: &str) -> Self {
        let mut without = self.clone();
        without.inputs.retain(|field| field.name != name);
        without.outputs.retain(|field| field.name != name);
        without
    }

    /// dspy `Signature.insert`: this signature with a field added at `index` of its own side.
    ///
    /// Upstream takes one `field` and reads which side it belongs to off its
    /// `__dspy_field_type`; a Rust field already *is* one side or the other, so
    /// [`Side`] carries that and there is nothing to look up.
    ///
    /// A negative index counts from the end *past* the last field, not before it: `-1` appends
    /// and `-2` inserts before the last. Upstream adds `len + 1` rather than Python's usual `len`,
    /// which is what makes one-past-the-end reachable from either direction.
    pub fn insert(&self, index: isize, field: Side) -> Result<Self, String> {
        let mut edited = self.clone();
        let length = match &field {
            Side::Input(_) => edited.inputs.len(),
            Side::Output(_) => edited.outputs.len(),
        } as isize;
        let at = match index < 0 {
            true => index + length + 1,
            false => index,
        };
        if at < 0 || at > length {
            // Upstream builds this message *after* adjusting, so a rejected negative index is
            // reported as what it became rather than as what was passed.
            return Err(format!(
                "Invalid index to insert: {at}, index must be in the range of [{}, {length}] for \
                 {} fields, but received: {at}.",
                length - 1,
                field.side_name(),
            ));
        }
        match field {
            Side::Input(input) => edited.inputs.insert(at as usize, input),
            Side::Output(output) => edited.outputs.insert(at as usize, output),
        }
        Ok(edited)
    }

    /// dspy `Signature.prepend`: the field first among its own side.
    pub fn prepend(&self, field: Side) -> Self {
        self.insert(0, field).expect("index 0 is always in range")
    }

    /// dspy `Signature.append`: the field last among its own side.
    pub fn append(&self, field: Side) -> Self {
        let end = match &field {
            Side::Input(_) => self.inputs.len(),
            Side::Output(_) => self.outputs.len(),
        } as isize;
        self.insert(end, field).expect("the end is always in range")
    }

    /// dspy `Signature.with_updated_fields`: one field's description or kind changed, the rest of
    /// the signature untouched.
    ///
    /// Upstream takes arbitrary `**kwargs` into the field's `json_schema_extra`, which is a
    /// Python dict; a Rust field is a struct, so the caller is handed the field to edit.
    ///
    /// A name on neither side is an error, not a no-op — upstream indexes `fields_copy[name]` and
    /// raises `KeyError`. This is the opposite of [`delete`](Self::delete), where upstream is
    /// deliberately forgiving, and the difference is worth keeping: deleting a field an adapter
    /// only sometimes adds is reasonable, while editing one that was never there is a typo.
    pub fn with_updated_fields(
        &self,
        name: &str,
        edit: impl FnOnce(&mut FieldEdit<'_>),
    ) -> Result<Self, String> {
        let mut edited = self.clone();
        if let Some(input) = edited.inputs.iter_mut().find(|field| field.name == name) {
            edit(&mut FieldEdit::Input(input));
        } else if let Some(output) = edited.outputs.iter_mut().find(|field| field.name == name) {
            edit(&mut FieldEdit::Output(output));
        } else {
            return Err(format!("{name:?}"));
        }
        Ok(edited)
    }

    /// dspy `Signature.with_instructions`: the same fields under a different objective. What an
    /// optimizer produces — every proposal it scores is this call.
    ///
    /// The text is normalised the way upstream normalises it: dspy keeps instructions in
    /// `__doc__` and reads them back through `inspect.cleandoc`, so a docstring's indent and its
    /// blank first and last lines never reach a prompt, and empty text states the default
    /// objective rather than leaving a signature with none.
    pub fn with_instructions(&self, instructions: impl Into<String>) -> Self {
        let inputs: Vec<&str> = self.inputs.iter().map(|f| f.name.as_str()).collect();
        let outputs: Vec<&str> = self.outputs.iter().map(|f| f.name.as_str()).collect();
        Self {
            instructions: instructions::stated(&instructions.into(), &inputs, &outputs),
            ..self.clone()
        }
    }

    /// dspy `Signature.append_instructions`: the existing instructions, then these, joined by a
    /// blank line.
    pub fn append_instructions(&self, instructions: impl Into<String>) -> Self {
        self.with_instructions(format!("{}\n\n{}", self.instructions, instructions.into()))
    }
}
