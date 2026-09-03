//! SIMBA's `OfferFeedback` — which is not `refine.py`'s, despite the shared name.
//!
//! Both ask a model to advise a program's modules and both are called `OfferFeedback`. Refine's
//! assigns blame against a **threshold** over one trajectory; this one contrasts a **worse
//! trajectory against a better one** and asks what would have made the worse behave like the
//! better. Thirteen input fields against nine, and an instruction block of its own.
//!
//! Held to `optimize/simba_offer_feedback.json`, which is the signature rendered through dspy's own
//! `ChatAdapter` — so the field order, every description and the schema note on `module_advice` are
//! compared as prompt bytes rather than as a list.

use crate::signature::{FieldKind, InField, JsonType, OutField, Signature};

const INSTRUCTIONS: &str = "You will be given two trajectories of an LLM-driven program's execution. Your goal is to help the program's modules\nbuild up experience on how to maximize the reward value assigned to the program's outputs if it were to receive\nsimilar inputs in the future.\n\nThe module won't see its own history. It will rely on your advice balancing being concrete and being generalizable.\n\nIn your advice:\n- Avoid boilerplate. Offer advice that would change the module's behavior for the better in the future.\n- Ensure that advice offered to a module M is specific to that M's specific sub-task, not the overall program.\n- Rely on contrasting the behavior of the worse trajectory against the better trajectory in making recommendations.\n- Ensure each unique module name appears exactly once as a key in the advice dictionary.";

const ADVICE_DESC: &str = "For each module, describe very concretely: If the module receives ${description of input or patterns therein}, then it should ${description of content, behavior, or strategies to adopt and/or others to avoid}. Basically, your advice be such that if the module has access to your tip, it would be much more likely to act like the successful trajectory rather than the lower-scoring trajectory.";

const TRAJECTORY_DESC: &str =
    "The trajectory of the program's execution, showing each module's I/O";
const OUTPUTS_DESC: &str = "The outputs of the program that we are analyzing";
const REWARD_DESC: &str = "The reward value assigned to the program's outputs";
const REWARD_INFO_DESC: &str =
    "Additional information that might be helpful to understanding the assigned reward value.";

fn input(name: &str, desc: &str, kind: FieldKind) -> InField {
    InField {
        name: name.to_owned(),
        desc: desc.to_owned(),
        kind,
        ..Default::default()
    }
}

/// dspy `simba_utils.OfferFeedback`.
///
/// The `worse_` block comes before the `better_` one, which is the order the model reads them in
/// and therefore prompt bytes rather than a preference.
pub fn offer_feedback() -> Signature {
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
                "oracle_metadata",
                "Any (hidden) metadata about the training set instance we're analyzing",
                FieldKind::Str,
            ),
            input("worse_program_trajectory", TRAJECTORY_DESC, FieldKind::Str),
            input("worse_program_outputs", OUTPUTS_DESC, FieldKind::Str),
            input("worse_reward_value", REWARD_DESC, FieldKind::Float),
            input("worse_reward_info", REWARD_INFO_DESC, FieldKind::Str),
            input("better_program_trajectory", TRAJECTORY_DESC, FieldKind::Str),
            input("better_program_outputs", OUTPUTS_DESC, FieldKind::Str),
            input("better_reward_value", REWARD_DESC, FieldKind::Float),
            input("better_reward_info", REWARD_INFO_DESC, FieldKind::Str),
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
                name: "module_advice".to_owned(),
                desc: ADVICE_DESC.to_owned(),
                kind: FieldKind::Json(JsonType::plain("dict[str, str]")),
                // As refine's: a structured output carries its schema onto the prompt line, which
                // is what steers the reply into a map rather than prose.
                schema: Some(
                    serde_json::json!({ "type": "object", "additionalProperties": { "type": "string" } }),
                ),
                ..Default::default()
            },
        ],
    }
}
