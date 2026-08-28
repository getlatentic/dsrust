//! dspy `LabeledFewShot` (`teleprompt/vanilla.py`): show the program examples that were
//! already labelled.

use crate::example::Example;
use crate::module::Module;

use super::Optimizer;
use super::rng::Rng;

/// Fill every predictor's demos straight from the trainset.
///
/// No model calls, no scoring — it simply shows the program examples somebody already labelled.
/// Weak as an optimizer, but it is the honest baseline every other one is measured against, and
/// [`BootstrapFewShot`](super::BootstrapFewShot) runs it over the teacher before asking the
/// teacher to solve anything, so the teacher works few-shot rather than cold.
pub struct LabeledFewShot {
    /// How many demos each predictor receives. dspy defaults to 16.
    pub k: usize,
    /// Take a deterministic sample of the trainset rather than its first `k`.
    pub sample: bool,
    /// dspy hardcodes `random.Random(0)`; nothing upstream depends on the seed being zero, so
    /// it is a knob here.
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

    /// Write demos into every predictor of `student`.
    ///
    /// dspy compiles into `student.reset_copy()` and hands that back, so whatever demos a
    /// predictor already held are gone — including on an empty trainset, where dspy returns the
    /// reset copy before writing anything at all. Compiling in place, that same decision reads
    /// as clearing the demos rather than leaving the program untouched.
    pub fn compile<M: Module + ?Sized>(&self, student: &mut M, trainset: &[Example]) {
        let mut rng = Rng::seeded(self.seed);
        let k = self.k.min(trainset.len());
        for predictor in student.named_predictors() {
            *predictor.demos = match (trainset.is_empty(), self.sample) {
                (true, _) => Vec::new(),
                // dspy draws from one generator for the whole walk, so the second predictor
                // gets a sample of its own rather than a copy of the first's.
                (false, true) => rng.sample(trainset, k),
                (false, false) => trainset[..k].to_vec(),
            };
        }
    }
}

impl Optimizer for LabeledFewShot {
    async fn compile<'a>(
        &'a self,
        student: &'a mut dyn Module,
        teacher: Option<&'a mut dyn Module>,
        trainset: &'a [Example],
        valset: Option<&'a [Example]>,
    ) -> anyhow::Result<()> {
        if valset.is_some() {
            return Err(anyhow::anyhow!(
                "LabeledFewShot samples demos from the trainset and never scores a candidate, \
                 so it has no valset to score on"
            ));
        }
        if teacher.is_some() {
            return Err(anyhow::anyhow!(
                "LabeledFewShot draws demos from the trainset and has no teacher to learn from"
            ));
        }
        LabeledFewShot::compile(self, student, trainset);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::example;
    use crate::optimize::scripted::{Answers, Pair, Solver, answers, trainset};

    #[test]
    fn labeled_few_shot_writes_demos_into_the_program() {
        let mut student = Solver::new(Answers::Correctly);
        LabeledFewShot::new(2).compile(&mut student, &trainset());
        assert_eq!(student.demos.len(), 2);
    }

    #[test]
    fn k_caps_the_demo_count() {
        let mut student = Solver::new(Answers::Correctly);
        LabeledFewShot::new(1).compile(&mut student, &trainset());
        assert_eq!(student.demos.len(), 1);
    }

    #[test]
    fn a_trainset_shorter_than_k_is_taken_whole() {
        let mut student = Solver::new(Answers::Correctly);
        LabeledFewShot::new(50).compile(&mut student, &trainset());
        assert_eq!(student.demos.len(), trainset().len());
    }

    /// dspy compiles into a reset copy, so an empty trainset clears the demos rather than
    /// leaving them: the program a caller gets back is always one this compile decided about.
    #[test]
    fn an_empty_trainset_clears_the_demos() {
        let mut student = Solver::new(Answers::Correctly);
        student.demos = vec![example! { question: "stale", answer: "stale" }];
        LabeledFewShot::new(4).compile(&mut student, &[]);
        assert!(student.demos.is_empty());
    }

    #[test]
    fn compiling_twice_replaces_rather_than_appends() {
        let mut student = Solver::new(Answers::Correctly);
        LabeledFewShot::new(3).compile(&mut student, &trainset());
        LabeledFewShot::new(3).compile(&mut student, &trainset());
        assert_eq!(student.demos.len(), 3);
    }

    /// dspy creates one generator per compile and calls `sample` once per predictor, so the
    /// generator has advanced by the time the second predictor is filled.
    #[test]
    fn each_predictor_draws_its_own_sample() {
        let mut student = Pair::new();
        LabeledFewShot::new(3).compile(&mut student, &trainset());
        assert_eq!(student.first_demos.len(), 3);
        assert_ne!(
            answers(&student.first_demos),
            answers(&student.second_demos)
        );
    }

    /// The unsampled path is a take, not a draw: dspy slices `trainset[:k]` in order.
    #[test]
    fn the_unsampled_path_takes_the_front_of_the_trainset_in_order() {
        let mut student = Solver::new(Answers::Correctly);
        let labeled = LabeledFewShot {
            sample: false,
            ..LabeledFewShot::new(2)
        };
        labeled.compile(&mut student, &trainset());
        assert_eq!(answers(&student.demos), ["Paris", "Berlin"]);
    }

    #[test]
    fn sampling_reorders_rather_than_taking_the_front() {
        let mut student = Solver::new(Answers::Correctly);
        LabeledFewShot::new(trainset().len()).compile(&mut student, &trainset());
        assert_ne!(answers(&student.demos), answers(&trainset()));
    }

    /// The answers of a demo list, which identify the trainset examples drawn since every answer is
    /// distinct — the order is the draw order, which is what a sample has and a take does not.
    fn drawn(demos: &[Example]) -> Vec<String> {
        demos
            .iter()
            .map(|demo| {
                demo.get("answer")
                    .and_then(|value| value.as_str())
                    .unwrap_or_default()
                    .to_owned()
            })
            .collect()
    }

    fn expected(case: &serde_json::Value, predictor: usize) -> Vec<String> {
        case["demos"][predictor]
            .as_array()
            .expect("a predictor's demos")
            .iter()
            .map(|demo| demo["answer"].as_str().expect("an answer").to_owned())
            .collect()
    }

    /// LabeledFewShot draws the demos dspy draws — the whole point of holding `random.sample` to
    /// CPython, seen end to end. From `tests/conformance/optimize/labeled_few_shot.json`, generated
    /// by running dspy's own `LabeledFewShot`. A two-predictor case pins that the second predictor
    /// draws from the advanced generator, not a copy of the first's sample.
    #[test]
    fn draws_the_demos_dspy_draws() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/conformance/optimize/labeled_few_shot.json");
        let text = std::fs::read_to_string(&path).expect("the labeled golden is committed");
        let fixture: serde_json::Value = serde_json::from_str(&text).expect("the golden parses");
        let cases = fixture["cases"].as_array().expect("cases");
        assert!(!cases.is_empty(), "the golden records no cases");

        for case in cases {
            let labeled = LabeledFewShot {
                k: case["k"].as_u64().expect("k") as usize,
                sample: case["sample"].as_bool().expect("sample"),
                seed: 0,
            };
            match case["predictors"].as_u64().expect("predictors") {
                1 => {
                    let mut student = Solver::new(Answers::Correctly);
                    labeled.compile(&mut student, &trainset());
                    assert_eq!(drawn(&student.demos), expected(case, 0), "case {case}");
                }
                _ => {
                    let mut student = Pair::new();
                    labeled.compile(&mut student, &trainset());
                    assert_eq!(
                        drawn(&student.first_demos),
                        expected(case, 0),
                        "first, case {case}"
                    );
                    assert_eq!(
                        drawn(&student.second_demos),
                        expected(case, 1),
                        "second, case {case}"
                    );
                }
            }
        }
    }
}
