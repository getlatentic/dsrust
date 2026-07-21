//! Advice carried from one attempt into the next.
//!
//! `Refine` runs a module several times, and between attempts asks a model what each predictor
//! should do differently. That answer reaches the predictor as one more input field — upstream
//! appends `hint_` to the signature and fills it from its per-module advice, so the module that
//! went wrong is the one told about it.
//!
//! Appended per call rather than written into the signature. The advice lasts exactly one attempt,
//! and a module carrying none renders byte-identically to one that never had any.

use serde_json::Value;

use crate::adapter::Input;
use crate::signature::{InField, Signature};

/// Upstream's name for the field. The trailing underscore is its own, and keeps it from colliding
/// with a field a caller declared.
const FIELD: &str = "hint_";

/// The line the field describes itself with, which reaches the prompt verbatim.
const DESC: &str = "A hint to the module from an earlier run";

/// `signature` with the hint field appended, or unchanged when there is no hint.
pub(super) fn signature_with(signature: &Signature, hint: Option<&str>) -> Signature {
    let mut asked = signature.clone();
    if hint.is_some() {
        asked.inputs.push(InField {
            name: FIELD.to_owned(),
            desc: DESC.to_owned(),
            ..Default::default()
        });
    }
    asked
}

/// `inputs` with the hint's value beside the field that describes it.
pub(super) fn inputs_with<'a>(inputs: &[Input<'a>], hint: Option<&str>) -> Vec<Input<'a>> {
    let mut hinted = inputs.to_vec();
    if let Some(hint) = hint {
        hinted.push(Input::new(FIELD, Value::String(hint.to_owned())));
    }
    hinted
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signature::OutField;

    fn signature() -> Signature {
        Signature {
            instructions: "Answer.".to_owned(),
            inputs: vec![InField {
                name: "question".to_owned(),
                ..Default::default()
            }],
            outputs: vec![OutField {
                name: "answer".to_owned(),
                ..Default::default()
            }],
        }
    }

    #[test]
    fn no_hint_leaves_the_signature_and_inputs_exactly_as_they_were() {
        let signature = signature();
        assert_eq!(signature_with(&signature, None).inputs.len(), 1);
        let inputs = [Input::new("question", Value::String("q".to_owned()))];
        assert_eq!(inputs_with(&inputs, None), inputs);
    }

    #[test]
    fn a_hint_appends_the_field_after_the_declared_ones() {
        let asked = signature_with(&signature(), Some("try warmer"));
        let names: Vec<&str> = asked.inputs.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(names, ["question", FIELD], "appended, not inserted");
        assert_eq!(asked.inputs[1].desc, DESC);
    }

    #[test]
    fn a_hint_travels_as_the_value_of_that_field() {
        let inputs = [Input::new("question", Value::String("q".to_owned()))];
        let hinted = inputs_with(&inputs, Some("try warmer"));
        assert_eq!(hinted.len(), 2);
        assert_eq!(hinted[1].name, FIELD);
        assert_eq!(hinted[1].value, Value::String("try warmer".to_owned()));
    }
}
