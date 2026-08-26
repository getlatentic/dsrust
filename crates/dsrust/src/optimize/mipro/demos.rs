//! dspy `create_n_fewshot_demo_sets` (`teleprompt/utils.py`): MIPROv2's Step 1. Build
//! `num_candidate_sets` demo sets per predictor by mixing strategies — a zero-shot set (no demos),
//! a labels-only set, an unshuffled bootstrap, then shuffled bootstraps whose size is drawn per set.
//!
//! The generator is the shared CPython RNG; only the shuffled sets draw from it (a shuffle then a
//! size), so Step 1 advances the RNG exactly that much before Step 2's proposal reads it. Bootstrap
//! runs the program, so the sets are a function of the model too — deterministic under a scripted one.

use anyhow::Result;

use super::super::rng::Rng;
use super::super::{BootstrapFewShot, LabeledFewShot};
use crate::example::{Example, Prediction};
use crate::module::Module;

/// dspy `min_num_samples`: the smallest a drawn bootstrap size may be.
const MIN_NUM_SAMPLES: u64 = 1;

/// Build the demo sets, indexed `[predictor][set]`. Each set is produced by clearing the program's
/// demos (the in-place stand-in for dspy's `reset_copy`), running the set's strategy, and reading
/// the demos back. The program is left demo-free.
pub(super) async fn create_demo_sets<S, M>(
    student: &mut S,
    num_candidate_sets: usize,
    trainset: &[Example],
    max_labeled: usize,
    max_bootstrapped: usize,
    metric: &M,
    metric_threshold: Option<f64>,
    rng: &mut Rng,
) -> Result<Vec<Vec<Vec<Example>>>>
where
    S: Module + ?Sized,
    M: Fn(&Example, &Prediction) -> f64 + Send + Sync,
{
    let predictors = student.named_predictors().len();
    let mut sets: Vec<Vec<Vec<Example>>> = vec![Vec::new(); predictors];

    // dspy's `range(-3, num_candidate_sets - 3)`: three special sets, then the rest shuffled.
    for seed in -3..(num_candidate_sets as i64 - 3) {
        clear_demos(student);
        match seed {
            -3 => {} // zero-shot: the cleared program is the set.
            -2 if max_labeled > 0 => {
                LabeledFewShot {
                    k: max_labeled,
                    sample: true,
                    seed: 0,
                }
                .compile(student, trainset);
            }
            -1 => {
                bootstrap(max_bootstrapped, max_labeled, metric, metric_threshold)
                    .compile(student, trainset)
                    .await?;
            }
            _ => {
                // dspy shuffles a fresh copy of the trainset and draws a size, both off the shared RNG.
                let mut shuffled = trainset.to_vec();
                rng.shuffle(&mut shuffled);
                let size = rng.randint(MIN_NUM_SAMPLES, max_bootstrapped as u64) as usize;
                bootstrap(size, max_labeled, metric, metric_threshold)
                    .compile(student, &shuffled)
                    .await?;
            }
        }
        for (index, predictor) in student.named_predictors().iter().enumerate() {
            sets[index].push(predictor.demos.clone());
        }
    }

    clear_demos(student);
    Ok(sets)
}

/// A bootstrap teleprompter borrowing the metric, so each set builds its own without the metric
/// needing to be cloned. dspy constructs a fresh `BootstrapFewShot` per set the same way.
fn bootstrap<M>(
    max_bootstrapped: usize,
    max_labeled: usize,
    metric: &M,
    metric_threshold: Option<f64>,
) -> BootstrapFewShot<&M> {
    BootstrapFewShot {
        max_bootstrapped_demos: max_bootstrapped,
        max_labeled_demos: max_labeled,
        metric_threshold,
        ..BootstrapFewShot::new(metric)
    }
}

fn clear_demos<S: Module + ?Sized>(student: &mut S) {
    for predictor in student.named_predictors() {
        predictor.demos.clear();
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use serde_json::Value;

    use super::super::super::goldens;
    use super::*;
    use crate::evaluate::exact_match;
    use crate::predict::Predict;
    use crate::{DummyLM, example};

    /// The builder passes the metric threshold through — deleted, every bootstrap accepted every
    /// trace and the threshold a caller set did nothing.
    #[test]
    fn the_builder_keeps_the_metric_threshold() {
        let metric = |_: &crate::Example, _: &crate::Prediction| 1.0;
        let built = bootstrap(4, 2, &metric, Some(0.75));
        assert_eq!(built.metric_threshold, Some(0.75));
        assert_eq!(built.max_bootstrapped_demos, 4);
        assert_eq!(built.max_labeled_demos, 2);
    }

    /// `clear_demos` actually clears — replaced by `()`, every fewshot round started from the
    /// previous round's demos instead of from none.
    #[test]
    fn clear_demos_empties_every_predictor() {
        let mut student =
            crate::Predict::from_signature("question -> answer".parse().expect("parses"));
        student
            .demos
            .push(crate::example! { question: "q", answer: "a" });
        clear_demos(&mut student);
        assert!(student.demos.is_empty());
    }

    fn fixture() -> Value {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/conformance/optimize/demo_sets.json");
        let text = std::fs::read_to_string(&path).expect("the demo-sets golden is committed");
        serde_json::from_str(&text).expect("the golden parses")
    }

    fn trainset(fixture: &Value) -> Vec<Example> {
        fixture["trainset"]
            .as_array()
            .expect("trainset")
            .iter()
            .map(|row| {
                example! { question: row["question"].as_str().unwrap().to_owned(), answer: row["answer"].as_str().unwrap().to_owned() }
                    .with_inputs(["question"])
            })
            .collect()
    }

    /// The keyed model dspy bootstrapped against: each question answers with the table's answer.
    fn model(fixture: &Value) -> Arc<DummyLM> {
        let pairs =
            fixture["answers"]
                .as_object()
                .expect("answers")
                .iter()
                .map(|(question, answer)| {
                    (
                        question.clone(),
                        example! { answer: answer.as_str().unwrap().to_owned() },
                    )
                });
        Arc::new(DummyLM::keyed(pairs))
    }

    /// A demo set, every field of every demo.
    ///
    /// This compared the answers alone, which identify the examples but say nothing about how
    /// they got into the set. `augmented` is the difference between a demo the teacher earned and
    /// one drawn from the trainset, and it is the key [`super::super::grounded`] gathers on — so a
    /// set that lost it still held the right examples and still grounded no proposal.
    fn built(set: &[Example]) -> Vec<BTreeMap<String, String>> {
        set.iter().map(goldens::fields).collect()
    }

    fn expected(set: &Value) -> Vec<BTreeMap<String, String>> {
        set.as_array()
            .expect("a set")
            .iter()
            .map(goldens::recorded_fields)
            .collect()
    }

    #[tokio::test]
    async fn builds_the_demo_sets_dspy_builds() {
        let fixture = fixture();
        let train = trainset(&fixture);
        for case in fixture["cases"].as_array().expect("cases") {
            let mut student = Predict::parse("question -> answer")
                .expect("parses")
                .set_lm(model(&fixture));
            let mut rng = Rng::seeded(case["seed"].as_u64().expect("seed"));
            let sets = create_demo_sets(
                &mut student,
                case["num_sets"].as_u64().expect("num_sets") as usize,
                &train,
                case["max_labeled"].as_u64().expect("max_labeled") as usize,
                case["max_bootstrapped"].as_u64().expect("max_bootstrapped") as usize,
                &exact_match,
                None,
                &mut rng,
            )
            .await
            .expect("builds the sets");

            let ours: Vec<_> = sets[0].iter().map(|set| built(set)).collect();
            let want: Vec<_> = case["sets"]
                .as_array()
                .expect("sets")
                .iter()
                .map(expected)
                .collect();
            assert_eq!(ours, want, "case {case}");
        }
    }
}
