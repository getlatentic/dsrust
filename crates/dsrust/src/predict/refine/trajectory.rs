//! The nine inputs `Refine` hands `OfferFeedback`: one attempt's run, described for advice.
//!
//! dspy assembles these in `Refine.forward`. The program's code, the modules' definition and the
//! reward's code travel as the strings they already are; every other value is JSON-dumped at a
//! two-space indent — upstream's `orjson.dumps(recursive_mask(v), option=OPT_INDENT_2)`. The
//! feedback half has no upstream oracle, so this is built to the algorithm and tested against
//! its own assertions rather than a golden.

use serde_json::{Value, json};

use crate::example::{Example, Prediction};
use crate::module::TraceStep;

/// The advice call's inputs, ready for the advisor [`Predict`](crate::Predict) to render.
///
/// Every value is a string: the three code fields verbatim, the rest as indented JSON — which is
/// what upstream sends, since it dumps each non-string value before the predictor ever sees it.
#[allow(clippy::too_many_arguments)]
pub(super) fn advise_inputs(
    program_code: &str,
    modules_defn: &str,
    program_inputs: &Example,
    trace: &[TraceStep],
    program_outputs: &Prediction,
    reward_code: &str,
    threshold: Option<f64>,
    reward_value: f64,
    module_names: &[String],
) -> Example {
    let names: Vec<Value> = module_names.iter().map(|name| json!(name)).collect();
    Example::new([
        ("program_code", json!(program_code)),
        ("modules_defn", json!(modules_defn)),
        ("program_inputs", dumped(&object_of(program_inputs))),
        ("program_trajectory", dumped(&trajectory(trace))),
        (
            "program_outputs",
            dumped(&object_of(&program_outputs.example)),
        ),
        ("reward_code", json!(reward_code)),
        ("target_threshold", dumped(&json!(threshold))),
        ("reward_value", dumped(&json!(reward_value))),
        ("module_names", dumped(&Value::Array(names))),
    ])
    .with_inputs([
        "program_code",
        "modules_defn",
        "program_inputs",
        "program_trajectory",
        "program_outputs",
        "reward_code",
        "target_threshold",
        "reward_value",
        "module_names",
    ])
}

/// One step per predictor that ran, in the order it ran — dspy's
/// `{"module_name", "inputs", "outputs"}` list.
fn trajectory(trace: &[TraceStep]) -> Value {
    let steps: Vec<Value> = trace
        .iter()
        // An unparsed step is omitted, which is what upstream's trace holds: a call whose parse
        // failed raises out of `Refine`'s forward and records nothing.
        .filter_map(|step| {
            let outputs = step.outputs.answered()?;
            Some(json!({
                "module_name": step.predictor,
                "inputs": object_of(&step.inputs),
                "outputs": object_of(outputs),
            }))
        })
        .collect();
    Value::Array(steps)
}

/// An example as the JSON object of its fields, which is how dspy dumps `dict(example)`.
fn object_of(example: &Example) -> Value {
    Value::Object(
        example
            .fields()
            .map(|(name, value)| (name.to_owned(), value.clone()))
            .collect(),
    )
}

/// A value as the string a predictor renders it from, indented two spaces — `orjson`'s
/// `OPT_INDENT_2`, which `serde_json`'s pretty writer matches.
fn dumped(value: &Value) -> Value {
    json!(serde_json::to_string_pretty(value).unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::example;

    fn program_inputs() -> Example {
        example! { question: "Why is the sky blue?" }.with_inputs(["question"])
    }

    fn outputs() -> Prediction {
        Prediction::new(example! { answer: "Rayleigh scattering." }, "raw")
    }

    fn trace() -> Vec<TraceStep> {
        vec![TraceStep {
            predictor: "predict".to_owned(),
            inputs: example! { question: "Why is the sky blue?" },
            outputs: crate::StepOutputs::Answered(example! { answer: "Rayleigh scattering." }),
            signature: crate::Signature::single_input("Answer.", Vec::new()),
        }]
    }

    fn inputs() -> Example {
        advise_inputs(
            "class Program: ...",
            "the modules",
            &program_inputs(),
            &trace(),
            &outputs(),
            "def reward(...): ...",
            Some(1.0),
            0.5,
            &["predict".to_owned()],
        )
    }

    #[test]
    fn every_one_of_offer_feedbacks_inputs_is_present_and_a_string() {
        let inputs = inputs();
        for field in [
            "program_code",
            "modules_defn",
            "program_inputs",
            "program_trajectory",
            "program_outputs",
            "reward_code",
            "target_threshold",
            "reward_value",
            "module_names",
        ] {
            let value = inputs
                .get(field)
                .unwrap_or_else(|| panic!("{field} is missing"));
            assert!(value.is_string(), "{field} reaches the advisor as a string");
        }
    }

    /// The three code fields are the strings they were handed; nothing dumps them again.
    #[test]
    fn the_code_fields_travel_verbatim() {
        let inputs = inputs();
        assert_eq!(
            inputs.get("program_code").unwrap(),
            &json!("class Program: ...")
        );
        assert_eq!(
            inputs.get("reward_code").unwrap(),
            &json!("def reward(...): ...")
        );
    }

    /// A float reaches the prompt as its own digits, not quoted twice — dumping `1.0` is `1.0`.
    #[test]
    fn the_threshold_and_reward_are_dumped_as_bare_numbers() {
        let inputs = inputs();
        assert_eq!(inputs.get("target_threshold").unwrap(), &json!("1.0"));
        assert_eq!(inputs.get("reward_value").unwrap(), &json!("0.5"));
    }

    /// A missing threshold dumps as `null`, since that is what upstream's `self.threshold` of
    /// `None` becomes.
    #[test]
    fn an_absent_threshold_dumps_as_null() {
        let inputs = advise_inputs(
            "",
            "",
            &program_inputs(),
            &trace(),
            &outputs(),
            "",
            None,
            0.5,
            &[],
        );
        assert_eq!(inputs.get("target_threshold").unwrap(), &json!("null"));
    }

    #[test]
    fn the_trajectory_is_a_step_per_predictor_that_ran() {
        let inputs = inputs();
        let rendered = inputs.get("program_trajectory").unwrap().as_str().unwrap();
        let parsed: Value = serde_json::from_str(rendered).expect("valid json");

        assert_eq!(parsed.as_array().expect("an array").len(), 1);
        assert_eq!(parsed[0]["module_name"], json!("predict"));
        assert_eq!(
            parsed[0]["inputs"]["question"],
            json!("Why is the sky blue?")
        );
        assert_eq!(
            parsed[0]["outputs"]["answer"],
            json!("Rayleigh scattering.")
        );
    }

    /// The names arrive as a JSON array, which the `list[str]` field renders straight.
    #[test]
    fn the_module_names_are_a_json_array() {
        let inputs = advise_inputs(
            "",
            "",
            &program_inputs(),
            &trace(),
            &outputs(),
            "",
            Some(1.0),
            0.5,
            &["first".to_owned(), "second".to_owned()],
        );
        let rendered = inputs.get("module_names").unwrap().as_str().unwrap();
        let parsed: Value = serde_json::from_str(rendered).expect("valid json");
        assert_eq!(parsed, json!(["first", "second"]));
    }
}
