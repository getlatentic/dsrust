//! dspy `teleprompt/knn_fewshot.py::KNNFewShot`: few-shot demos chosen at call time.
//!
//! Upstream compiles a student whose `forward` finds the k training examples nearest the call's
//! inputs, runs `BootstrapFewShot` over just those, and answers with the program that produced.
//! The same here: the compiled program builds a fresh student each call — upstream's
//! `reset_copy`, which a Rust module has no `deepcopy` for, so the caller hands in how one is
//! built — bootstraps it on the nearest examples, and forwards through it.

use std::sync::Arc;

use anyhow::Result;

use super::Optimizer;
use super::bootstrap::BootstrapFewShot;
use crate::evaluate::Metric;
use crate::example::{Example, Prediction};
use crate::lm::embedding::Embedder;
use crate::module::{Module, NamedPredictor};
use crate::predict::knn::Knn;

pub struct KnnFewShot<M> {
    knn: Arc<Knn>,
    bootstrap: BootstrapFewShot<M>,
}

impl<M> KnnFewShot<M>
where
    M: Metric + Clone + Send + Sync + 'static,
{
    /// `KNNFewShot(k, trainset, vectorizer, **few_shot_bootstrap_args)`: the bootstrap's own
    /// arguments arrive as a `BootstrapFewShot` built the way upstream's keyword arguments build one.
    pub async fn build(
        k: usize,
        trainset: Vec<Example>,
        vectorizer: Arc<Embedder>,
        bootstrap: BootstrapFewShot<M>,
    ) -> Result<Self> {
        Ok(Self {
            knn: Arc::new(Knn::build(k, trainset, vectorizer).await?),
            bootstrap,
        })
    }

    pub fn knn(&self) -> &Knn {
        &self.knn
    }

    /// `compile(student, teacher=...)`: the program that bootstraps on the nearest examples at
    /// each call. `student` and `teacher` build a fresh module each time they are called, the way
    /// upstream's `reset_copy` produces one.
    pub fn compile<S: Module>(
        &self,
        student: impl Fn() -> S + Send + Sync + 'static,
        teacher: Option<impl Fn() -> S + Send + Sync + 'static>,
    ) -> KnnFewShotProgram<S, M> {
        KnnFewShotProgram {
            student: Box::new(student),
            teacher: teacher.map(|build| Box::new(build) as Box<dyn Fn() -> S + Send + Sync>),
            knn: Arc::clone(&self.knn),
            bootstrap: self.bootstrap.clone(),
        }
    }
}

/// The compiled program: `forward_pass` in upstream's `compile`.
pub struct KnnFewShotProgram<S, M> {
    student: Box<dyn Fn() -> S + Send + Sync>,
    teacher: Option<Box<dyn Fn() -> S + Send + Sync>>,
    knn: Arc<Knn>,
    bootstrap: BootstrapFewShot<M>,
}

impl<S, M> Module for KnnFewShotProgram<S, M>
where
    S: Module,
    M: Metric + Clone + Send + Sync + 'static,
{
    fn forward<'a>(
        &'a self,
        inputs: Example,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Prediction>> + Send + 'a>> {
        Box::pin(async move {
            let nearest = self.knn.call(&inputs).await?;
            let mut compiled = (self.student)();
            let mut teacher = self.teacher.as_ref().map(|build| build());
            let teacher: Option<&mut dyn Module> = teacher.as_mut().map(|t| t as &mut dyn Module);
            Optimizer::compile(&self.bootstrap, &mut compiled, teacher, &nearest, None).await?;
            compiled.forward(inputs).await
        })
    }

    /// A fresh student is built for every call, so there is no predictor of this program's own to
    /// hand an optimizer — the demos it would write are chosen at call time.
    fn named_predictors(&mut self) -> Vec<NamedPredictor<'_>> {
        Vec::new()
    }
}
