//! One answering shape for every module a task was named on.
//!
//! dspy is uniform without trying: every module returns a `Prediction`, `Prediction.__getattr__`
//! reads its store, and so `out.answer` is the same line whichever module produced it. Rust has no
//! dynamic attribute, so the field has to exist — which it does for `Predict<Task>`, whose type
//! carries the task, and does not for an agent, whose outputs are whatever `submit` carried.
//!
//! This is what carries the task for the modules that cannot: the task is named where the module
//! is built, and the answer arrives as the task's own outputs struct. So `call!` answers with a
//! struct for every spelling that names a task, which is what a reader expects after seeing one.

use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;

use anyhow::Result;

use crate::Example;
use crate::module::{Ask, Module};
use crate::signature::SignatureSpec;

/// A module asked through the task it was built from.
///
/// Built by the task arm of a module macro — `ReActV2!(QA, tools)` — rather than named directly.
pub struct Typed<S: SignatureSpec, M> {
    module: M,
    spec: PhantomData<S>,
}

impl<S: SignatureSpec, M> Typed<S, M> {
    pub fn new(module: M) -> Self {
        Self {
            module,
            spec: PhantomData,
        }
    }

    /// The module underneath, for the builders it carries and for the `Prediction` an agent's
    /// trajectory lives in — which a typed answer has nowhere to put.
    pub fn into_module(self) -> M {
        self.module
    }

    /// A builder applied to the module underneath, keeping the task: agents configure by consuming
    /// `self` and answering with `Self`, and there is no way to forward a method this does not
    /// know the name of.
    ///
    /// ```
    /// # use dsrust::{ReActV2, Signature, Tool, tool};
    /// # #[derive(Signature)]
    /// # /// Answer it.
    /// # struct QA {
    /// #     #[input] question: String,
    /// #     #[output] answer: String,
    /// # }
    /// # let tools: Vec<Box<dyn Tool>> = Vec::new();
    /// let agent = ReActV2!(QA, tools).map(|agent| agent.max_iters(8));
    /// ```
    pub fn map(self, build: impl FnOnce(M) -> M) -> Self {
        Self::new(build(self.module))
    }
}

/// The module underneath, for reading: `agent.signature` and `agent.max_iters` are the module's
/// own, and naming the task should not put them out of reach. A builder consumes `self` and so is
/// not reachable this way — that is what [`Typed::map`] is for.
impl<S: SignatureSpec, M> std::ops::Deref for Typed<S, M> {
    type Target = M;

    fn deref(&self) -> &M {
        &self.module
    }
}

/// Still a module, so an optimizer and an evaluator take it as they took what it wraps. Naming a
/// task changes how the answer is *read*, not what the thing is: `compile` rewrites the demos on
/// the module underneath, and `forward` answers with the `Prediction` those tools work in.
impl<S, M> Module for Typed<S, M>
where
    S: SignatureSpec + Send + Sync,
    M: Module,
{
    fn forward<'a>(
        &'a self,
        inputs: Example,
    ) -> Pin<Box<dyn Future<Output = Result<crate::Prediction>> + Send + 'a>> {
        self.module.forward(inputs)
    }

    fn named_predictors(&mut self) -> Vec<crate::module::NamedPredictor<'_>> {
        self.module.named_predictors()
    }
}

impl<S, M> Ask for Typed<S, M>
where
    S: SignatureSpec + Send + Sync,
    M: Module + Send + Sync,
{
    type Answer = S::Outputs;

    fn ask<'a>(
        &'a self,
        inputs: Example,
    ) -> Pin<Box<dyn Future<Output = Result<S::Outputs>> + Send + 'a>> {
        Box::pin(async move { self.module.forward(inputs).await?.typed::<S::Outputs>() })
    }
}
