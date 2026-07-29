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
mod gepa;
mod mipro;
mod copro;
mod earned;
mod ensemble;
mod random_search;
mod labeled;
mod rng;

#[cfg(test)]
mod conformance;
#[cfg(test)]
pub(crate) mod scripted;

pub use better_together::{BetterTogether, Candidate};
pub use bootstrap::BootstrapFewShot;
pub use copro::COPRO;
pub use ensemble::{Ensemble, Ensembled};
pub use random_search::{Attempt, BootstrapRandomSearch};
pub use gepa::{Feedback, GEPA, GepaOutcome};
pub use mipro::MIPROv2;
pub use labeled::LabeledFewShot;

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
    /// upstream's would refuse the keyword.
    fn compile<'a>(
        &'a self,
        student: &'a mut dyn Module,
        teacher: Option<&'a mut dyn Module>,
        trainset: &'a [Example],
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
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;
}

impl<T: Optimizer + Send + Sync> DynOptimizer for T {
    fn compile_dyn<'a>(
        &'a self,
        student: &'a mut dyn Module,
        teacher: Option<&'a mut dyn Module>,
        trainset: &'a [Example],
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(self.compile(student, teacher, trainset))
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
        fn compile<'a>(
            &'a self,
            student: &'a mut dyn Module,
            _teacher: Option<&'a mut dyn Module>,
            trainset: &'a [Example],
        ) -> impl Future<Output = Result<()>> + Send + 'a {
            async move {
                for predictor in student.named_predictors() {
                    *predictor.demos = trainset.iter().take(1).cloned().collect();
                }
                Ok(())
            }
        }
    }

    #[tokio::test]
    async fn an_optimizer_defined_by_a_caller_compiles_a_program() {
        let examples = trainset();
        let mut student = Solver::new(Answers::Correctly);
        KeepTheFirst
            .compile(&mut student, None, &examples)
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
                .compile_dyn(&mut student, None, &examples)
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
        )
        .await
        .expect("a teacher compiles the student");

        assert_eq!(answers(&student.demos), ["Paris", "Berlin"]);
        assert!(teacher.demos.is_empty(), "the student took the result");
    }
}
