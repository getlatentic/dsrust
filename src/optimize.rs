//! Compilers: they read a program and write back a better one.
//!
//! This is the layer DSPy is named for. A signature says what the task is, a module says how to
//! ask, and an optimizer decides what the prompt should actually contain — by choosing demos,
//! or rewriting instructions — measured against a metric rather than guessed at.
//!
//! Every optimizer here works through [`Module::named_predictors`], the same seam dspy's do:
//! walk the program, read each predictor, write improved demos back. That is why `Predict`
//! implementing `Module` mattered — without it there is nothing for a compiler to reach into.

use anyhow::Result;

use crate::example::{Example, Prediction};
use crate::module::Module;

/// Deterministic sampling without pulling in a random-number crate.
///
/// dspy seeds `random.Random(0)` so a compile is reproducible; the same requirement here is
/// better served by a fixed permutation than by matching Python's Mersenne Twister, which no
/// Rust crate reproduces byte for byte anyway. Reproducible, not identical — and said out loud
/// because a caller comparing the two would otherwise expect the same picks.
fn sample(trainset: &[Example], k: usize, seed: u64) -> Vec<Example> {
    let take = k.min(trainset.len());
    let mut indices: Vec<usize> = (0..trainset.len()).collect();
    // A simple xorshift walk over the indices: stable across runs and platforms.
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    for position in (1..indices.len()).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let swap = (state % (position as u64 + 1)) as usize;
        indices.swap(position, swap);
    }
    indices
        .into_iter()
        .take(take)
        .map(|index| trainset[index].clone())
        .collect()
}

/// Fill every predictor's demos straight from the trainset.
///
/// dspy's `LabeledFewShot`: no model calls, no scoring — it simply shows the program examples
/// that were already labelled. Weak as an optimizer, but it is the honest baseline every other
/// one is measured against, and it exercises the whole compile seam.
pub struct LabeledFewShot {
    /// How many demos each predictor receives. dspy defaults to 16.
    pub k: usize,
    /// Take a deterministic sample of the trainset rather than its first `k`.
    pub sample: bool,
    pub seed: u64,
}

impl Default for LabeledFewShot {
    fn default() -> Self {
        Self {
            k: 16,
            sample: true,
            seed: 0,
        }
    }
}

impl LabeledFewShot {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            ..Self::default()
        }
    }

    /// Write demos into every predictor of `student`. An empty trainset leaves it untouched,
    /// matching dspy, because a compile that silently erased demos would be worse than a
    /// compile that did nothing.
    pub fn compile<M: Module>(&self, student: &mut M, trainset: &[Example]) {
        if trainset.is_empty() {
            return;
        }
        let chosen = match self.sample {
            true => sample(trainset, self.k, self.seed),
            false => trainset.iter().take(self.k).cloned().collect(),
        };
        for predictor in student.named_predictors() {
            *predictor.demos = chosen.clone();
        }
    }
}

/// Keep only the demos the program can actually solve.
///
/// dspy's `BootstrapFewShot` in its essential form: run the student over the trainset, score
/// each attempt with the metric, and keep the traces that pass as demos. The difference from
/// [`LabeledFewShot`] is the difference between showing a program any labelled example and
/// showing it examples of its own successful behaviour.
pub struct BootstrapFewShot<M> {
    pub metric: M,
    /// Stop after this many successful demos. dspy's `max_bootstrapped_demos` is 4.
    pub max_demos: usize,
    /// A trace counts as a demo at or above this score.
    pub threshold: f64,
}

impl<M> BootstrapFewShot<M>
where
    M: Fn(&Example, &Prediction) -> f64,
{
    pub fn new(metric: M) -> Self {
        Self {
            metric,
            max_demos: 4,
            threshold: 1.0,
        }
    }

    /// Run the student over the trainset and keep the traces that score well enough.
    ///
    /// The student is asked with whatever demos it already has, which is how dspy bootstraps
    /// from a teacher: the traces reflect the program as it stands, not an idealised one.
    pub async fn compile<P: Module>(&self, student: &mut P, trainset: &[Example]) -> Result<usize> {
        let mut kept: Vec<Example> = Vec::new();
        for example in trainset {
            if kept.len() >= self.max_demos {
                break;
            }
            let inputs = example.inputs()?;
            let Ok(prediction) = student.forward(inputs.clone()).await else {
                // A failed call is not a usable demo, and it is not a reason to abandon the
                // rest of the trainset either.
                continue;
            };
            if (self.metric)(example, &prediction) < self.threshold {
                continue;
            }
            // The demo carries the inputs that were asked and the outputs that worked, which
            // is what the adapter renders as a solved turn pair.
            let mut demo = inputs;
            for (name, value) in prediction.example.fields() {
                demo.set(name, value.clone());
            }
            kept.push(demo);
        }
        let count = kept.len();
        for predictor in student.named_predictors() {
            *predictor.demos = kept.clone();
        }
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluate::exact_match;
    use crate::example;
    use crate::module::NamedPredictor;
    use crate::signature::Signature;
    use serde_json::json;

    /// A student that answers from a table, so a compile can be checked without a provider.
    struct Student {
        signature: Signature,
        demos: Vec<Example>,
        correct: bool,
    }

    impl Student {
        fn new(correct: bool) -> Self {
            Self {
                signature: Signature::single_input("Answer.", Vec::new()),
                demos: Vec::new(),
                correct,
            }
        }
    }

    impl Module for Student {
        fn forward<'a>(
            &'a self,
            inputs: Example,
        ) -> std::pin::Pin<Box<dyn Future<Output = Result<Prediction>> + Send + 'a>> {
            Box::pin(async move {
                let question = inputs
                    .get("question")
                    .and_then(|value| value.as_str())
                    .unwrap_or_default();
                let answer = match (self.correct, question.contains("France")) {
                    (true, true) => "Paris",
                    (true, false) => "Berlin",
                    (false, _) => "no idea",
                };
                Ok(Prediction::new(
                    Example::new([("answer", json!(answer))]),
                    "raw",
                ))
            })
        }

        fn named_predictors(&mut self) -> Vec<NamedPredictor<'_>> {
            vec![NamedPredictor {
                name: "self".to_owned(),
                signature: &mut self.signature,
                demos: &mut self.demos,
            }]
        }
    }

    fn trainset() -> Vec<Example> {
        vec![
            example! { question: "capital of France?", answer: "Paris" }.with_inputs(["question"]),
            example! { question: "capital of Germany?", answer: "Berlin" }
                .with_inputs(["question"]),
        ]
    }

    #[test]
    fn labeled_few_shot_writes_demos_into_the_program() {
        let mut student = Student::new(true);
        LabeledFewShot::new(2).compile(&mut student, &trainset());
        assert_eq!(student.demos.len(), 2);
    }

    #[test]
    fn k_caps_the_demo_count() {
        let mut student = Student::new(true);
        LabeledFewShot::new(1).compile(&mut student, &trainset());
        assert_eq!(student.demos.len(), 1);
    }

    #[test]
    fn an_empty_trainset_leaves_the_program_alone() {
        // Erasing demos would be worse than doing nothing, so a no-op compile stays a no-op.
        let mut student = Student::new(true);
        student.demos = vec![example! { question: "kept", answer: "kept" }];
        LabeledFewShot::new(4).compile(&mut student, &[]);
        assert_eq!(student.demos.len(), 1);
    }

    #[test]
    fn sampling_is_reproducible_across_runs() {
        let first = sample(&trainset(), 2, 0);
        let second = sample(&trainset(), 2, 0);
        assert_eq!(first, second);
    }

    #[tokio::test]
    async fn bootstrap_keeps_only_traces_the_metric_accepts() {
        let mut student = Student::new(true);
        let kept = BootstrapFewShot::new(exact_match)
            .compile(&mut student, &trainset())
            .await
            .expect("compile succeeds");
        assert_eq!(kept, 2);
        assert_eq!(student.demos.len(), 2);
        // A bootstrapped demo carries the inputs asked and the outputs that worked.
        let demo = &student.demos[0];
        assert_eq!(demo.get("question").unwrap(), &json!("capital of France?"));
        assert_eq!(demo.get("answer").unwrap(), &json!("Paris"));
    }

    #[tokio::test]
    async fn a_program_that_never_succeeds_bootstraps_nothing() {
        // The honest outcome: no demos, rather than demos of wrong behaviour.
        let mut student = Student::new(false);
        let kept = BootstrapFewShot::new(exact_match)
            .compile(&mut student, &trainset())
            .await
            .expect("compile succeeds");
        assert_eq!(kept, 0);
        assert!(student.demos.is_empty());
    }

    #[tokio::test]
    async fn max_demos_stops_the_walk_early() {
        let mut student = Student::new(true);
        let mut bootstrap = BootstrapFewShot::new(exact_match);
        bootstrap.max_demos = 1;
        let kept = bootstrap
            .compile(&mut student, &trainset())
            .await
            .expect("compile succeeds");
        assert_eq!(kept, 1);
    }
}
