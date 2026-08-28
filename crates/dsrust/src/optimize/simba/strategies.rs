//! The two things SIMBA does with a bucket: show the program an example, or tell it a rule.
//!
//! Both are gated on the batch's own percentiles rather than on absolute scores, and both *decline*
//! rather than fail when the gate closes — a bucket whose best run is not better than the batch's
//! 10th percentile has nothing to teach, and dspy's strategies return `False` there.

use anyhow::Result;

use super::search::{Bucket, Strategy};
use crate::example::Example;
use crate::module::Module;

/// Run one strategy against a bucket, answering whether it changed the program.
pub(super) async fn apply(
    strategy: &Strategy,
    bucket: &Bucket,
    student: &mut dyn Module,
    gates: (f64, f64),
    demo_input_field_maxlen: usize,
) -> Result<bool> {
    match strategy {
        Strategy::AppendADemo => Ok(append_a_demo(
            bucket,
            student,
            gates.0,
            demo_input_field_maxlen,
        )),
        Strategy::AppendARule => append_a_rule(bucket, student, gates).await,
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
) -> Result<bool> {
    let (batch_10p, batch_90p) = gates;
    let (Some(best), Some(worst)) = (bucket.runs.first(), bucket.runs.last()) else {
        return Ok(false);
    };
    if best.score <= batch_10p || worst.score >= batch_90p {
        return Ok(false);
    }
    let _ = student;
    // The ask itself is `feedback::offer_feedback`, whose thirteen inputs are built from these two
    // runs. Not wired to a model here yet — see the module doc.
    Ok(false)
}
