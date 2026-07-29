//! dspy `teleprompt/ensemble.py`: several programs answering together.
//!
//! The one teleprompter that does not optimize anything. Every other takes a student and improves
//! it; this takes a *list* of programs and hands back one that runs them all and reduces their
//! answers — which is why its `compile` has a different shape from the [`Optimizer`] trait's, here
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
        Self { reduce: Some(reduce), size: None, seed: 0 }
    }

    /// A subset of `size` programs answers each call, drawn without replacement.
    ///
    /// dspy draws with the global `random`, so a run is reproducible only against a seeded
    /// interpreter. The draw is per *call*, not per compile — the same ensemble asked twice may
    /// ask different members, which is the point of sampling one.
    pub fn with_size(mut self, size: usize) -> Self {
        self.size = Some(size);
        self
    }

    /// The seed the per-call draw starts from. dspy has no equivalent — it reaches for the
    /// process-wide `random` — and a caller who wants the same subset twice has nowhere to say so.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// dspy `Ensemble.compile`: the ensembled program.
    ///
    /// Takes the programs rather than a student, which is why this is not [`Optimizer`]. Upstream's
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
            answers.push(self.programs[index].forward_traced(inputs.clone(), trace).await?);
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
    use crate::example;
    use serde_json::json;

    /// A program that answers with what it was built with, so which members ran is readable.
    struct Fixed(&'static str);

    impl Module for Fixed {
        fn forward<'a>(
            &'a self,
            _inputs: Example,
        ) -> std::pin::Pin<Box<dyn Future<Output = Result<Prediction>> + Send + 'a>> {
            Box::pin(std::future::ready(Ok(Prediction::new(
                Example::new([("answer", json!(self.0))]),
                String::new(),
            ))))
        }
    }

    fn programs(names: &[&'static str]) -> Vec<Box<dyn Module>> {
        names.iter().map(|name| Box::new(Fixed(name)) as Box<dyn Module>).collect()
    }

    fn first_answer(answers: &[Prediction]) -> Result<Prediction> {
        Ok(answers.first().cloned().expect("at least one answer"))
    }

    /// Every member answers and the reduction decides, which is the whole module.
    #[tokio::test]
    async fn every_program_answers_and_the_reduction_decides() {
        let seen = std::sync::Arc::new(Mutex::new(Vec::new()));
        let recorded = seen.clone();
        let ensembled = Ensemble::new(move |answers: &[Prediction]| {
            recorded.lock().expect("seen").extend(
                answers.iter().map(|a| a.get("answer").cloned().unwrap_or(json!(null))),
            );
            first_answer(answers)
        })
        .compile(programs(&["a", "b", "c"]));

        let answered = ensembled.forward(example! { q: "x" }).await.expect("answers");
        assert_eq!(answered.get("answer"), Some(&json!("a")), "the reduction chose");
        assert_eq!(*seen.lock().expect("seen"), vec![json!("a"), json!("b"), json!("c")]);
    }

    /// With no reduction dspy hands back the raw list; a `Prediction` is one answer, so the list
    /// travels as a field.
    #[tokio::test]
    async fn no_reduction_hands_back_every_answer() {
        let ensembled: Ensembled<fn(&[Prediction]) -> Result<Prediction>> =
            Ensemble { reduce: None, size: None, seed: 0 }.compile(programs(&["a", "b"]));
        let answered = ensembled.forward(example! { q: "x" }).await.expect("answers");
        let outputs = answered.get("outputs").expect("the list").as_array().expect("an array");
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0]["answer"], json!("a"));
    }

    /// A size draws that many members, without replacement, and a fresh draw per call.
    #[tokio::test]
    async fn a_size_draws_that_many_members_per_call() {
        let seen = std::sync::Arc::new(Mutex::new(Vec::new()));
        let recorded = seen.clone();
        let ensembled = Ensemble::new(move |answers: &[Prediction]| {
            recorded
                .lock()
                .expect("seen")
                .push(answers.iter().filter_map(|a| a.get("answer").cloned()).collect::<Vec<_>>());
            first_answer(answers)
        })
        .with_size(2)
        .compile(programs(&["a", "b", "c", "d"]));

        for _ in 0..3 {
            ensembled.forward(example! { q: "x" }).await.expect("answers");
        }
        let draws = seen.lock().expect("seen").clone();
        for draw in &draws {
            assert_eq!(draw.len(), 2, "two members answered");
            assert_ne!(draw[0], draw[1], "and without replacement");
        }
        assert!(draws.len() == 3, "a draw per call");
    }

    /// The same seed draws the same members, which is the part dspy leaves to a process-wide RNG.
    #[tokio::test]
    async fn the_same_seed_draws_the_same_members() {
        let drawn = |seed: u64| {
            let ensembled = Ensemble::new(first_answer as fn(&[Prediction]) -> Result<Prediction>)
                .with_size(2)
                .with_seed(seed)
                .compile(programs(&["a", "b", "c", "d"]));
            (0..3).map(|_| ensembled.asked()).collect::<Vec<_>>()
        };
        assert_eq!(drawn(7), drawn(7));
        assert_ne!(drawn(7), drawn(8), "and a different seed draws differently");
    }
}
