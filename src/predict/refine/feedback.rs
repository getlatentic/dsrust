//! dspy `OfferFeedback`: what each module should do differently next time.
//!
//! Between two attempts `Refine` asks a model to assign blame across the modules that ran and
//! prescribe advice for each. The advice comes back keyed by module name, and reaches the next
//! attempt as that predictor's `hint_` input — so the module that went wrong is the one told
//! about it.

use serde_json::Value;

use crate::signature::{FieldKind, InField, JsonType, OutField, Signature};

/// Verbatim from `OfferFeedback.__doc__` as dspy normalises it, since it reaches the prompt.
const INSTRUCTIONS: &str = "In the discussion, assign blame to each module that contributed to the final reward being below the threshold, if
any. Then, prescribe concrete advice of how the module should act on its future input when we retry the process, if
it were to receive the same or similar inputs. If a module is not to blame, the advice should be N/A.
The module will not see its own history, so it needs to rely on entirely concrete and actionable advice from you
to avoid the same mistake on the same or similar inputs.";

const ADVICE_DESC: &str = "For each module, describe very concretely, in this order: the specific scenarios in which it has made mistakes in the past and what each mistake was, followed by what it should do differently in that kind ofscenario in the future. If the module is not to blame, write N/A.";

/// What a module with no advice of its own is told.
pub(super) const NO_ADVICE: &str = "N/A";

fn input(name: &str, desc: &str, kind: FieldKind) -> InField {
    InField {
        name: name.to_owned(),
        desc: desc.to_owned(),
        kind,
        ..Default::default()
    }
}

/// The nine inputs and two outputs, in dspy's declaration order — which is the order they render.
pub fn signature() -> Signature {
    Signature {
        instructions: INSTRUCTIONS.to_owned(),
        inputs: vec![
            input(
                "program_code",
                "The code of the program that we are analyzing",
                FieldKind::Str,
            ),
            input(
                "modules_defn",
                "The definition of each module in the program, including its I/O",
                FieldKind::Str,
            ),
            input(
                "program_inputs",
                "The inputs to the program that we are analyzing",
                FieldKind::Str,
            ),
            input(
                "program_trajectory",
                "The trajectory of the program's execution, showing each module's I/O",
                FieldKind::Str,
            ),
            input(
                "program_outputs",
                "The outputs of the program that we are analyzing",
                FieldKind::Str,
            ),
            input(
                "reward_code",
                "The code of the reward function that we are analyzing",
                FieldKind::Str,
            ),
            input(
                "target_threshold",
                "The target threshold for the reward function",
                FieldKind::Float,
            ),
            input(
                "reward_value",
                "The reward value assigned to the program's outputs",
                FieldKind::Float,
            ),
            input(
                "module_names",
                "The names of the modules in the program, for which we seek advice",
                FieldKind::Json(JsonType::plain("list[str]")),
            ),
        ],
        outputs: vec![
            OutField {
                name: "discussion".to_owned(),
                desc: "Discussing blame of where each module went wrong, if it did".to_owned(),
                kind: FieldKind::Str,
                ..Default::default()
            },
            OutField {
                name: "advice".to_owned(),
                desc: ADVICE_DESC.to_owned(),
                kind: FieldKind::Json(JsonType::plain("dict[str, str]")),
                // A structured output field carries its schema onto the prompt line, which is
                // what steers the reply into a map rather than prose. An input never does.
                schema: Some(
                    serde_json::json!({ "type": "object", "additionalProperties": { "type": "string" } }),
                ),
                ..Default::default()
            },
        ],
    }
}

/// The advice a reply carries, keyed by module name.
///
/// A reply that parsed but named no module is not an error: every predictor then falls back to
/// [`NO_ADVICE`], which is what upstream's `advice.get(name, "N/A")` does per lookup.
pub(super) fn advice_of(outputs: &serde_json::Map<String, Value>) -> serde_json::Map<String, Value> {
    match outputs.get("advice") {
        Some(Value::Object(advice)) => advice.clone(),
        _ => serde_json::Map::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_nine_inputs_and_two_outputs_are_in_dspys_order() {
        let signature = signature();
        let inputs: Vec<&str> = signature.inputs.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(
            inputs,
            [
                "program_code",
                "modules_defn",
                "program_inputs",
                "program_trajectory",
                "program_outputs",
                "reward_code",
                "target_threshold",
                "reward_value",
                "module_names",
            ]
        );
        let outputs: Vec<&str> = signature.outputs.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(outputs, ["discussion", "advice"]);
    }

    /// The two numeric inputs are floats and the two collections carry their Python spelling,
    /// because both reach the prompt on the numbered field line.
    #[test]
    fn the_declared_types_are_the_ones_dspy_prints() {
        let signature = signature();
        assert_eq!(signature.inputs[6].kind, FieldKind::Float);
        assert_eq!(
            signature.inputs[8].kind,
            FieldKind::Json(JsonType::plain("list[str]"))
        );
        assert_eq!(
            signature.outputs[1].kind,
            FieldKind::Json(JsonType::plain("dict[str, str]"))
        );
    }

    #[test]
    fn a_reply_naming_no_module_leaves_every_predictor_on_the_fallback() {
        let outputs = serde_json::Map::new();
        assert!(advice_of(&outputs).is_empty());
    }

    #[test]
    fn advice_is_read_back_keyed_by_module_name() {
        let outputs = serde_json::json!({ "advice": { "predict": "be terser" } });
        let advice = advice_of(outputs.as_object().expect("an object"));
        assert_eq!(advice["predict"], serde_json::json!("be terser"));
    }
}
