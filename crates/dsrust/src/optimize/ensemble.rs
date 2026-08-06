//! dspy `teleprompt/ensemble.py`: several programs answering together.
//!
//! The one teleprompter that does not optimize anything. Every other takes a student and improves
//! it; this takes a *list* of programs and hands back one that runs them all and reduces their
//! answers — which is why its `compile` has a different shape from the [`Optimizer`](super::Optimizer) trait's, here
//! as upstream.
//!
//! The reduction is the caller's: `dspy.majority` is the one its docstring names, and
//! [`majority`](crate::predict::majority) is that function.

use std::sync::Mutex;

use anyhow::Result;
use pyrng::Random;

use crate::example::{Example, Prediction};
use crate::module::{Module, NamedPredictor, TraceStep};

/// dspy `Ensemble`: build a program that asks several and reduces what they say.
pub struct Ensemble<R> {
    reduce: Option<R>,
    size: Option<usize>,
    seed: u64,
}

impl<R> Ensemble<R>
where
    R: Fn(&[Prediction]) -> Result<Prediction> + Send + Sync,
{
    /// Every program answers, and `reduce` decides what that comes to.
    pub fn new(reduce: R) -> Self {
        Self {
            reduce: Some(reduce),
            size: None,
            seed: 0,
        }
    }

    /// A subset of `size` programs answers each call, drawn without replacement.
    ///
    /// dspy draws with the global `random`, so a run is reproducible only against a seeded
    /// interpreter. The draw is per *call*, not per compile — the same ensemble asked twice may
    /// ask different members, which is the point of sampling one.
    pub fn size(mut self, size: usize) -> Self {
        self.size = Some(size);
        self
    }

    /// The seed the per-call draw starts from. dspy has no equivalent — it reaches for the
    /// process-wide `random` — and a caller who wants the same subset twice has nowhere to say so.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// dspy `Ensemble.compile`: the ensembled program.
    ///
    /// Takes the programs rather than a student, which is why this is not [`Optimizer`](super::Optimizer). Upstream's
    /// signature differs from every other teleprompter's in exactly the same way.
    pub fn compile(self, programs: Vec<Box<dyn Module>>) -> Ensembled<R> {
        Ensembled {
            programs,
            reduce: self.reduce,
            size: self.size,
            rng: Mutex::new(Random::seeded(self.seed)),
        }
    }
}

/// dspy's `EnsembledProgram`: what [`Ensemble::compile`] hands back.
pub struct Ensembled<R> {
    programs: Vec<Box<dyn Module>>,
    reduce: Option<R>,
    size: Option<usize>,
    /// Behind a lock because [`Module::forward`] takes `&self` and a draw advances the stream —
    /// the same reason a `Tool` holding a subprocess does.
    rng: Mutex<Random>,
}

impl<R> Ensembled<R>
where
    R: Fn(&[Prediction]) -> Result<Prediction> + Send + Sync,
{
    /// How many programs are in the ensemble, whatever a call may sample from them.
    pub fn len(&self) -> usize {
        self.programs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.programs.is_empty()
    }

    /// Which programs answer this call: all of them, or a draw of `size` without replacement.
    fn asked(&self) -> Vec<usize> {
        let all: Vec<usize> = (0..self.programs.len()).collect();
        match self.size {
            None => all,
            Some(size) => self.rng.lock().expect("the draw lock").sample(&all, size),
        }
    }

    async fn run(&self, inputs: Example, trace: &mut Vec<TraceStep>) -> Result<Prediction> {
        let mut answers = Vec::new();
        for index in self.asked() {
            answers.push(
                self.programs[index]
                    .forward_traced(inputs.clone(), trace)
                    .await?,
            );
        }
        match &self.reduce {
            Some(reduce) => reduce(&answers),
            // dspy returns the raw list when there is no reduction. A `Prediction` is one answer,
            // so the list travels as a field rather than as a shape this crate does not have.
            None => Ok(Prediction::new(
                Example::new([(
                    "outputs",
                    serde_json::Value::Array(
                        answers
                            .iter()
                            .map(|answer| {
                                answer
                                    .example
                                    .fields()
                                    .map(|(name, value)| (name.to_owned(), value.clone()))
                                    .collect::<serde_json::Map<_, _>>()
                                    .into()
                            })
                            .collect(),
                    ),
                )]),
                String::new(),
            )),
        }
    }
}

impl<R> Module for Ensembled<R>
where
    R: Fn(&[Prediction]) -> Result<Prediction> + Send + Sync,
{
    fn forward<'a>(
        &'a self,
        inputs: Example,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<Prediction>> + Send + 'a>> {
        Box::pin(async move {
            let mut discarded = Vec::new();
            self.run(inputs, &mut discarded).await
        })
    }

    fn forward_traced<'a>(
        &'a self,
        inputs: Example,
        trace: &'a mut Vec<TraceStep>,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<Prediction>> + Send + 'a>> {
        Box::pin(async move { self.run(inputs, trace).await })
    }

    /// Every member's predictors, each under its own index — so an optimizer walking an ensemble
    /// reaches all of them and two members' identically-named predictors stay apart.
    fn named_predictors(&mut self) -> Vec<NamedPredictor<'_>> {
        let mut predictors = Vec::new();
        for (index, program) in self.programs.iter_mut().enumerate() {
            for mut predictor in program.named_predictors() {
                predictor.name = format!("programs[{index}].{}", predictor.name);
                predictors.push(predictor);
            }
        }
        predictors
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The accessors answer for the programs actually held, and an optimizer walking the ensemble
    /// reaches every member's predictors under distinct indexed names.
    #[test]
    fn the_ensemble_reports_its_members_and_walks_all_of_them() {
        let reduce = |predictions: &[crate::Prediction]| {
            Ok(predictions.first().expect("one prediction").clone())
        };
        let programs: Vec<Box<dyn crate::Module>> = (0..2)
            .map(|_| {
                Box::new(crate::Predict::from_signature(
                    "question -> answer".parse().expect("parses"),
                )) as Box<dyn crate::Module>
            })
            .collect();
        let mut ensembled = Ensemble::new(reduce).compile(programs);
        assert_eq!(ensembled.len(), 2);
        assert!(!ensembled.is_empty());

        let named = crate::Module::named_predictors(&mut ensembled);
        let names: Vec<String> = named.iter().map(|p| p.name.clone()).collect();
        assert_eq!(names.len(), 2, "every member is walked: {names:?}");
        assert!(names[0].starts_with("programs[0]."), "{names:?}");
        assert!(names[1].starts_with("programs[1]."), "{names:?}");

        let empty = Ensemble::new(reduce).compile(Vec::new());
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
    }
}
