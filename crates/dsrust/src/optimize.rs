//! Compilers: they read a program and write back a better one.
//!
//! This is the layer DSPy is named for. A signature says what the task is, a module says how to
//! ask, and an optimizer decides what the prompt should actually contain — by choosing demos,
//! or rewriting instructions — measured against a metric rather than guessed at.
//!
//! Every optimizer here works through [`Module::named_predictors`](crate::Module), the same seam
//! dspy's do: walk the program, read each predictor, write improved demos back. That is why
//! `Predict` implementing `Module` mattered — without it there is nothing for a compiler to
//! reach into.
//!
//! The split mirrors dspy's own: `vanilla.py` holds the labelled baseline, `bootstrap.py` holds
//! the optimizer that runs a program to earn its demos and imports the baseline to prime its
//! teacher.

mod better_together;
mod bootstrap;
mod copro;
mod earned;
mod ensemble;
mod gepa;
mod infer_rules;
pub mod knn_fewshot;
mod labeled;
mod mipro;
mod optuna;
mod random_search;
mod rng;
pub mod simba;

#[cfg(test)]
mod conformance;
#[cfg(test)]
mod goldens;
#[cfg(test)]
pub(crate) mod scripted;

/// What an optimizer's scoring passes are bounded by — dspy's `num_threads` and `max_errors`, which
/// every teleprompter takes and hands to the `Evaluate` it builds.
///
/// One type rather than two fields per optimizer: four of them build an `Evaluate`, and a fifth
/// would otherwise inherit neither setting by simply not knowing to. [`apply`](Self::apply) is the
/// only way an optimizer wires it, so there is one place to read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Scoring {
    /// Rows in flight per pass. dspy's `num_threads`, `None` for one at a time as upstream defaults.
    pub num_threads: Option<usize>,
    /// Failed rows a pass tolerates before giving up. dspy's `max_errors`, default 10.
    pub max_errors: usize,
}

impl Default for Scoring {
    fn default() -> Self {
        Self {
            num_threads: None,
            max_errors: crate::evaluate::DEFAULT_MAX_ERRORS,
        }
    }
}

impl Scoring {
    /// These settings on one scoring pass.
    pub fn apply<P, M, F>(
        self,
        evaluate: crate::evaluate::Evaluate<P, M>,
    ) -> crate::evaluate::Evaluate<P, M>
    where
        P: Fn(crate::Example) -> F,
        F: std::future::Future<Output = anyhow::Result<crate::Prediction>>,
        M: crate::evaluate::Metric,
    {
        let evaluate = evaluate.max_errors(self.max_errors);
        match self.num_threads {
            Some(threads) => evaluate.num_threads(threads),
            None => evaluate,
        }
    }
}

pub use better_together::{BetterTogether, StepResult};
pub use bootstrap::BootstrapFewShot;
pub use copro::{COPRO, CoproStats, DepthScores};
pub use ensemble::{Ensemble, Ensembled};
// `Reflective` travels with the two: a caller implementing `InstructionProposer` cannot read the
// dataset it is handed without naming the type its entries are, and reaching for it would mean
// depending on the engine crate directly.
pub use gepa::{
    Candidate, Event, Feedback, GEPA, GepaOutcome, InstructionProposer, MetricContext,
    MultiModalInstructionProposer, Progress, Reflective, ReflectiveDataset, Reported,
};
pub use infer_rules::{InferRules, RuleCandidate};
pub use labeled::LabeledFewShot;
// `Trial` travels beside it: `MIPROv2::compile_traced` answers with `Vec<Trial>`, and a type a
// caller cannot name is one they cannot put in a signature, a field or a `let`.
pub use mipro::{Auto, MIPROv2, Trial};
pub use optuna::{BootstrapFewShotWithOptuna, OptunaTrial};
pub use random_search::{Attempt, BootstrapRandomSearch};

use std::pin::Pin;

use anyhow::Result;

use crate::example::Example;
use crate::module::Module;

/// dspy's `Teleprompter`: the trait an optimizer implements, and the one to implement to add
/// your own.
///
/// Upstream's subclasses each vary `compile`'s signature through `**kwargs`, which nothing here
/// can do, so this is the part they share and per-optimizer configuration lives on the struct.
/// Each optimizer also keeps an inherent `compile` of its own natural shape, taking a concrete
/// student and answering with whatever it learned; the two differ in arity, so a call resolves
/// to one or the other at compile time rather than silently.
///
/// A program is compiled in place. dspy compiles into `student.reset_copy()` and hands back the
/// copy, which is the same decision written where ownership is explicit.
pub trait Optimizer {
    /// A teacher produces the demos the student keeps, which is how a strong model teaches a
    /// cheap one. An optimizer with no use for one refuses it rather than ignoring it, the way
    /// upstream's would refuse the keyword — and the same goes for the valset.
    ///
    /// `valset` is what candidates are scored on. What `None` means is each optimizer's own
    /// business, because upstream's differ: GEPA's is `valset or trainset`, scoring on the whole
    /// trainset, while MIPROv2's `_set_and_validate_datasets` keeps the first 20% to bootstrap from
    /// and scores on the last 80%. An optimizer that never scores at all — `LabeledFewShot`,
    /// `BootstrapFewShot` — refuses one, since upstream's `compile` has no such keyword to pass.
    fn compile<'a>(
        &'a self,
        student: &'a mut dyn Module,
        teacher: Option<&'a mut dyn Module>,
        trainset: &'a [Example],
        valset: Option<&'a [Example]>,
    ) -> impl Future<Output = Result<()>> + Send + 'a;
}

/// The object-safe form of [`Optimizer`], which every `Optimizer` gets through the blanket impl
/// below.
///
/// A meta-optimizer needs it: dspy's `BetterTogether` holds other optimizers and runs them in
/// sequence, and holding one in Rust means `Box<dyn DynOptimizer>`.
pub trait DynOptimizer: Send + Sync {
    fn compile_dyn<'a>(
        &'a self,
        student: &'a mut dyn Module,
        teacher: Option<&'a mut dyn Module>,
        trainset: &'a [Example],
        valset: Option<&'a [Example]>,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;
}

impl<T: Optimizer + Send + Sync> DynOptimizer for T {
    fn compile_dyn<'a>(
        &'a self,
        student: &'a mut dyn Module,
        teacher: Option<&'a mut dyn Module>,
        trainset: &'a [Example],
        valset: Option<&'a [Example]>,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(self.compile(student, teacher, trainset, valset))
    }
}

#[cfg(test)]
mod tests {
    use super::scripted::{Answers, Solver, answers, trainset};
    use super::*;
    use crate::evaluate::exact_match;

    /// An optimizer written from outside the crate's own types, against nothing but the trait.
    struct KeepTheFirst;

    impl Optimizer for KeepTheFirst {
        async fn compile<'a>(
            &'a self,
            student: &'a mut dyn Module,
            _teacher: Option<&'a mut dyn Module>,
            trainset: &'a [Example],
            _valset: Option<&'a [Example]>,
        ) -> Result<()> {
            for predictor in student.named_predictors() {
                *predictor.demos = trainset.iter().take(1).cloned().collect();
            }
            Ok(())
        }
    }

    #[tokio::test]
    async fn an_optimizer_defined_by_a_caller_compiles_a_program() {
        let examples = trainset();
        let mut student = Solver::new(Answers::Correctly);
        KeepTheFirst
            .compile(&mut student, None, &examples, None)
            .await
            .expect("a caller's optimizer compiles");
        assert_eq!(answers(&student.demos), ["Paris"]);
    }

    /// dspy's `BetterTogether` holds other optimizers and runs them in sequence. Storing one
    /// behind a pointer is what makes that possible, and it is why [`DynOptimizer`] exists.
    #[tokio::test]
    async fn optimizers_of_different_types_run_from_behind_one_pointer() {
        let examples = trainset();
        let sequence: Vec<Box<dyn DynOptimizer>> = vec![
            Box::new(KeepTheFirst),
            Box::new(LabeledFewShot::new(2)),
            Box::new(BootstrapFewShot::new(exact_match)),
        ];

        let mut student = Solver::new(Answers::Correctly);
        for optimizer in &sequence {
            optimizer
                .compile_dyn(&mut student, None, &examples, None)
                .await
                .expect("each optimizer in the sequence compiles");
        }
        // The last one to run decides what the program keeps, so the answer is the one
        // `bootstrap_few_shot.json` records upstream making at these budgets.
        assert_eq!(
            answers(&student.demos),
            [
                "Paris",
                "Berlin",
                "riddle 3!",
                "riddle 0!",
                "riddle 2!",
                "riddle 1!"
            ]
        );
    }

    #[tokio::test]
    async fn an_optimizer_with_no_use_for_a_teacher_refuses_one() {
        let examples = trainset();
        let mut student = Solver::new(Answers::Correctly);
        let mut teacher = Solver::new(Answers::Correctly);
        let refused = Optimizer::compile(
            &LabeledFewShot::new(2),
            &mut student,
            Some(&mut teacher),
            &examples,
            None,
        )
        .await;
        assert!(
            refused.unwrap_err().to_string().contains("no teacher"),
            "a teacher should be refused rather than dropped"
        );
    }

    #[tokio::test]
    async fn a_teacher_reaches_the_optimizer_that_takes_one() {
        let examples = trainset();
        let mut student = Solver::new(Answers::Correctly);
        let mut teacher = Solver::new(Answers::Correctly);
        Optimizer::compile(
            &BootstrapFewShot {
                max_labeled_demos: 0,
                ..BootstrapFewShot::new(exact_match)
            },
            &mut student,
            Some(&mut teacher),
            &examples,
            None,
        )
        .await
        .expect("a teacher compiles the student");

        assert_eq!(answers(&student.demos), ["Paris", "Berlin"]);
        assert!(teacher.demos.is_empty(), "the student took the result");
    }

    /// An optimizer that never scores a candidate refuses a valset rather than ignoring one, the way
    /// it already refuses a teacher it cannot learn from. Upstream's `compile` has no such keyword,
    /// so passing one is a `TypeError` there and silently dropping it here would be worse: a caller
    /// would believe their validation set was being used.
    #[tokio::test]
    async fn an_optimizer_that_never_scores_refuses_a_valset() {
        let examples = trainset();
        let mut student = Solver::new(Answers::Correctly);
        let refused = Optimizer::compile(
            &LabeledFewShot::new(2),
            &mut student,
            None,
            &examples,
            Some(&examples),
        )
        .await;
        assert!(
            refused
                .unwrap_err()
                .to_string()
                .contains("no valset to score on"),
            "LabeledFewShot should refuse a valset it would not read"
        );
    }
}
