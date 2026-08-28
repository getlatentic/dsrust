//! The two things SIMBA does with a bucket: show the program an example, or tell it a rule.
//!
//! Both are gated on the batch's own percentiles rather than on absolute scores, and both *decline*
//! rather than fail when the gate closes — a bucket whose best run is not better than the batch's
//! 10th percentile has nothing to teach, and dspy's strategies return `False` there.

use anyhow::Result;

use serde_json::Value;

use super::search::{Bucket, Run, Strategy};
use crate::example::Example;
use crate::module::Module;
use crate::predict::Predict;
use crate::predict::refine::describe;

/// Run one strategy against a bucket, answering whether it changed the program.
pub(super) async fn apply(
    strategy: &Strategy,
    bucket: &Bucket,
    student: &mut dyn Module,
    gates: (f64, f64),
    demo_input_field_maxlen: usize,
    advisor: &Predict,
) -> Result<bool> {
    match strategy {
        Strategy::AppendADemo => Ok(append_a_demo(
            bucket,
            student,
            gates.0,
            demo_input_field_maxlen,
        )),
        Strategy::AppendARule => append_a_rule(bucket, student, gates, advisor).await,
    }
}

/// dspy `append_a_demo`: the best run's steps become one demo per predictor.
///
/// **The last step wins per predictor**, not the first: upstream writes into `name2demo[name]` as
/// it walks the trace, so a predictor called twice contributes only its final call. And the demo
/// carries `augmented=true`, which is the mark that tells a bootstrapped demo from a labelled one
/// everywhere else in the crate.
fn append_a_demo(bucket: &Bucket, student: &mut dyn Module, batch_10p: f64, maxlen: usize) -> bool {
    let Some(best) = bucket.runs.first() else {
        return false;
    };
    // Not `<`: a best run *at* the 10th percentile is declined too, which matters when a batch's
    // scores are mostly equal and the percentile lands on the maximum.
    if best.score <= batch_10p {
        return false;
    }
    let mut per_predictor: Vec<(String, Example)> = Vec::new();
    for step in &best.trace {
        let mut fields: Vec<(String, serde_json::Value)> =
            vec![("augmented".to_owned(), serde_json::Value::Bool(true))];
        for (name, value) in step.inputs.fields() {
            fields.push((name.to_owned(), truncated(value, maxlen)));
        }
        for (name, value) in step.outputs.fields() {
            fields.push((name.to_owned(), value.clone()));
        }
        let demo = Example::new(fields);
        match per_predictor
            .iter_mut()
            .find(|(name, _)| *name == step.predictor)
        {
            Some(slot) => slot.1 = demo,
            None => per_predictor.push((step.predictor.clone(), demo)),
        }
    }
    let mut added = false;
    for predictor in student.named_predictors() {
        if let Some((_, demo)) = per_predictor
            .iter()
            .find(|(name, _)| *name == predictor.name)
        {
            predictor.demos.push(demo.clone());
            added = true;
        }
    }
    added
}

/// dspy's truncation of a long input field, with its exact marker.
///
/// The cut is on the *rendered* string and the marker carries two tabs, both of which reach the
/// prompt.
fn truncated(value: &serde_json::Value, maxlen: usize) -> serde_json::Value {
    let rendered = match value {
        serde_json::Value::String(text) => text.clone(),
        other => other.to_string(),
    };
    if maxlen == 0 || rendered.chars().count() <= maxlen {
        return value.clone();
    }
    let kept: String = rendered.chars().take(maxlen).collect();
    serde_json::Value::String(format!("{kept}\n\t\t... <TRUNCATED FOR BREVITY>"))
}

/// dspy `append_a_rule`: ask a model what the better run did that the worse one did not, and append
/// its advice to each named predictor's instructions.
///
/// Gated at *both* ends — a best run at or below the 10th percentile has nothing to teach, and a
/// worst run at or above the 90th has nothing to warn about.
async fn append_a_rule(
    bucket: &Bucket,
    student: &mut dyn Module,
    gates: (f64, f64),
    advisor: &Predict,
) -> Result<bool> {
    let (batch_10p, batch_90p) = gates;
    let (Some(best), Some(worst)) = (bucket.runs.first(), bucket.runs.last()) else {
        return Ok(false);
    };
    if best.score <= batch_10p || worst.score >= batch_90p {
        return Ok(false);
    }
    let names: Vec<String> = student
        .named_predictors()
        .iter()
        .map(|predictor| predictor.name.clone())
        .collect();
    let described = describe::modules(&student.named_predictors());
    let asked = ask(best, worst, &names, &described);

    let advice = advisor.forward(asked).await?;
    let Some(Value::Object(per_module)) = advice.get("module_advice").cloned() else {
        // A reply the adapter accepted but which carries no map: dspy would raise on the lookup,
        // and a candidate that cannot be advised is one the search moves past.
        return Ok(false);
    };
    let mut applied = false;
    for predictor in student.named_predictors() {
        // dspy's `if name in advice`, so a module the model did not name keeps its instructions.
        if let Some(Value::String(advice)) = per_module.get(&predictor.name) {
            // Appended after a blank line, not substituted: the original instruction stays and the
            // rule is read as an addition to it.
            predictor.signature.instructions =
                format!("{}\n\n{advice}", predictor.signature.instructions);
            applied = true;
        }
    }
    Ok(applied)
}

/// The thirteen inputs, built from the two runs.
///
/// Every non-string value is two-space JSON — dspy's `orjson.dumps(..., OPT_INDENT_2)` — so a
/// trajectory reaches the model as indented text rather than as one line.
fn ask(best: &Run, worst: &Run, names: &[String], described: &str) -> Example {
    let (better, worse) = blanked(best, worst);
    Example::new([
        // Rust has no runtime source to read, where dspy calls `inspect.getsource` on the
        // program's class. The signature description below carries what the ask needs.
        ("program_code", Value::from("")),
        ("modules_defn", Value::from(described)),
        ("program_inputs", indented(&fields_of(&best.example))),
        ("oracle_metadata", indented(&labels_of(&best.example))),
        ("worse_program_trajectory", indented(&worse.trajectory)),
        ("worse_program_outputs", indented(&worse.outputs)),
        ("worse_reward_value", worse.score.clone()),
        (
            "worse_reward_info",
            indented(&Value::Object(Default::default())),
        ),
        ("better_program_trajectory", indented(&better.trajectory)),
        ("better_program_outputs", indented(&better.outputs)),
        ("better_reward_value", better.score.clone()),
        (
            "better_reward_info",
            indented(&Value::Object(Default::default())),
        ),
        ("module_names", indented(&Value::from(names))),
    ])
}

/// One side of the contrast, after dspy's tie handling.
struct Side {
    trajectory: Value,
    outputs: Value,
    score: Value,
}

/// dspy's tie arm: when the better run did **not** beat the worse, the better one is blanked.
///
/// Its trace empties, its outputs become `{"N/A": "Prediction not available"}`, and its score
/// becomes the *string* `"N/A"` — in a field the signature declares `float`, which is upstream's
/// and reaches the model as those three characters.
///
/// dspy has a second arm here that blanks the *worse* run instead, and it is **unreachable**: past
/// the gate `bad < p90`, and that arm needs `good <= bad` and `good > p90`, so
/// `p90 < good <= bad < p90`. A sweep of the four scores finds no case, recorded in
/// `optimize/simba_rule.json` as `worse_blanking_arm_reachable: false`.
fn blanked(best: &Run, worst: &Run) -> (Side, Side) {
    let worse = Side {
        trajectory: trajectory_of(worst),
        outputs: outputs_of(worst),
        score: Value::from(worst.score),
    };
    if best.score <= worst.score {
        return (
            Side {
                trajectory: Value::Array(Vec::new()),
                outputs: serde_json::json!({ "N/A": "Prediction not available" }),
                score: Value::from("N/A"),
            },
            worse,
        );
    }
    (
        Side {
            trajectory: trajectory_of(best),
            outputs: outputs_of(best),
            score: Value::from(best.score),
        },
        worse,
    )
}

/// dspy's trajectory shape: one object per recorded step, in the order they ran.
fn trajectory_of(run: &Run) -> Value {
    Value::Array(
        run.trace
            .iter()
            .map(|step| {
                serde_json::json!({
                    "module_name": step.predictor,
                    "inputs": fields_of(&step.inputs),
                    "outputs": fields_of(&step.outputs),
                })
            })
            .collect(),
    )
}

fn outputs_of(run: &Run) -> Value {
    match &run.prediction {
        Some(prediction) => fields_of(&prediction.example),
        None => Value::Object(Default::default()),
    }
}

fn fields_of(example: &Example) -> Value {
    Value::Object(
        example
            .fields()
            .map(|(name, value)| (name.to_owned(), value.clone()))
            .collect(),
    )
}

/// dspy's `example.labels()`: the fields that are not declared inputs.
fn labels_of(example: &Example) -> Value {
    Value::Object(
        example
            .fields()
            .filter(|(name, _)| !example.is_input(name))
            .map(|(name, value)| (name.to_owned(), value.clone()))
            .collect(),
    )
}

/// `orjson.dumps(value, option=OPT_INDENT_2)`, which is `serde_json`'s pretty printer exactly —
/// two spaces, `": "` between key and value, and no trailing newline. Checked rather than assumed.
fn indented(value: &Value) -> Value {
    Value::from(serde_json::to_string_pretty(value).unwrap_or_default())
}
