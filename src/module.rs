//! What every module has in common, so a caller can write their own.
//!
//! dspy's `Module` is a base class: `Predict`, `ChainOfThought` and `ReAct` all subclass it,
//! and users write their own the same way. Two things depend on that shape. An evaluator
//! takes any module, not a specific one. And an optimizer walks a program to find the
//! predictors inside it, reads their demos and instructions, and writes back improved ones —
//! `named_predictors` in Python is exactly that walk.
//!
//! The Rust equivalent is a trait. It stays object-safe so a program can hold
//! `Box<dyn Module>`, which is what a composed program needs.

use anyhow::Result;

use crate::example::{Example, Prediction};
use crate::lm::LmConfig;
use crate::signature::Signature;

/// One predictor inside a program: its signature and its demos, borrowed for inspection or
/// mutation. An optimizer's whole job is reading these and writing back better ones.
pub struct NamedPredictor<'a> {
    pub name: String,
    pub signature: &'a mut Signature,
    pub demos: &'a mut Vec<Example>,
    /// How this predictor asks for its reply to be sampled.
    ///
    /// Unlike the other two this is not something an optimizer *learns* — it is set for the
    /// duration of a round and then set back. It rides the same walk because reaching every
    /// predictor is the same problem, and dspy solves it the same way: `set_lm` on a program
    /// assigns to all of them.
    pub config: &'a mut LmConfig,
    /// Advice for this predictor from an earlier attempt, shown to it on the next one.
    ///
    /// `Refine` writes it and `Predict` renders it as one more input field. Per predictor rather
    /// than per program because that is what upstream's advice is: `OfferFeedback` answers with a
    /// map keyed by module name, so the module that went wrong is the one told about it.
    pub hint: &'a mut Option<String>,
}

/// One predictor call: which predictor ran, what it was asked, and what it answered.
///
/// dspy's `settings.trace` step, a `(predictor, inputs, outputs)` triple appended to a
/// thread-local that an optimizer reads afterwards. Here the trace is passed rather than
/// ambient, and a predictor is identified by the name [`Module::named_predictors`] gives it
/// rather than by object identity, so the two walks agree by construction.
#[derive(Clone)]
pub struct TraceStep {
    pub predictor: String,
    pub inputs: Example,
    pub outputs: Example,
}

/// What a program of your own has to say: how to run it.
///
/// [`Module`] is what everything else takes, and it is boxed and object-safe so a composed
/// program can hold `Box<dyn Module>`. That shape is right for a caller and wrong for an author,
/// who would be writing `Pin<Box<dyn Future>>` by hand for no reason. Implement this instead and
/// `#[derive(Module)]` writes the rest: the walk an optimizer needs, the trace a compile needs,
/// and the boxing.
///
/// ```ignore
/// #[derive(Module)]
/// struct Outline {
///     plan: Predict,
///     write: Predict,
/// }
///
/// impl Forward for Outline {
///     async fn forward(&self, inputs: Example) -> Result<Prediction> {
///         let angle = self.plan.forward(inputs).await?;
///         self.write
///             .forward(dsrs::input! { angle: angle.get("angle").cloned().unwrap_or_default() })
///             .await
///     }
/// }
/// ```
pub trait Forward: Send + Sync {
    fn forward(&self, inputs: Example) -> impl Future<Output = Result<Prediction>> + Send;
}

/// A callable program. Implement it to add a module of your own.
pub trait Module: Send + Sync {
    /// Run the program over one example's inputs.
    ///
    /// Boxed rather than `async fn` so the trait stays object-safe: a composed program holds
    /// its children as `Box<dyn Module>`, and an evaluator takes any module at all.
    fn forward<'a>(
        &'a self,
        inputs: Example,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<Prediction>> + Send + 'a>>;

    /// Every predictor this program contains, named. dspy's `named_predictors`, and the seam
    /// an optimizer works through: it reads the demos and instructions here and writes back
    /// improved ones. A leaf module returns itself; a composed one returns its children.
    fn named_predictors(&mut self) -> Vec<NamedPredictor<'_>> {
        Vec::new()
    }

    /// Ask every predictor in this program for its reply to be sampled this way.
    ///
    /// dspy's `set_lm`, which assigns one model to a whole program so `lm.copy(rollout_id=n,
    /// temperature=1.0)` reaches every call an attempt makes. LmConfig travels on a request here
    /// rather than on a model, so this sets config instead — the effect is the one upstream
    /// relies on: a second attempt at a program differs from the first everywhere, not only at
    /// whichever predictor a caller remembered.
    fn set_config(&mut self, config: LmConfig) {
        for predictor in self.named_predictors() {
            *predictor.config = config.clone();
        }
    }

    /// Run the program, recording which predictor saw what.
    ///
    /// An optimizer needs this to give each predictor demos its own calls earned. A composed
    /// module passes the trace down and relabels what its children recorded, the same way
    /// [`Self::named_predictors`] relabels theirs, so a step's name always matches a predictor's.
    ///
    /// Recording nothing is allowed and means the program cannot be attributed, so every
    /// predictor in it receives the same program-level demo. That costs nothing for a program
    /// with one predictor, where the two are the same list.
    fn forward_traced<'a>(
        &'a self,
        inputs: Example,
        trace: &'a mut Vec<TraceStep>,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<Prediction>> + Send + 'a>> {
        let _ = trace;
        self.forward(inputs)
    }
}

/// Rename every step `record` added, which is how a composed module claims its children's calls
/// as its own predictor.
pub fn relabel(trace: &mut [TraceStep], from: usize, name: &str) {
    for step in &mut trace[from..] {
        step.predictor = name.to_owned();
    }
}

/// Ask a module, naming each input where its value goes.
///
/// ```
/// # async fn wrapper(haiku: dsrs::Predict) -> anyhow::Result<()> {
/// let result = dsrs::call!(haiku, subject = "computer science", tone = "wry").await?;
/// # Ok(()) }
/// ```
///
/// Rust has neither named arguments nor a mapping literal, so the two are written here instead:
/// the field name sits where the value does, which is what `subject=` does in Python. Evaluates
/// to the call's future, so the caller writes `.await?` and sees where the model is reached.
///
/// Asks through [`Ask`], so what comes back is whatever the module promised. A module of your
/// own joins in with one line — `dsrs::asks_with_a_prediction!(YourModule);` — which is the same
/// line the modules here use.
#[macro_export]
macro_rules! call {
    ($module:expr, $($field:ident = $value:expr),* $(,)?) => {
        $crate::Ask::ask(
            &$module,
            $crate::input! { $($field: $value),* },
        )
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::example;
    use serde_json::json;

    /// A module written outside the crate's own types, to prove the trait is implementable.
    struct Echo {
        signature: Signature,
        demos: Vec<Example>,
        config: LmConfig,
        hint: Option<String>,
    }

    impl Module for Echo {
        fn forward<'a>(
            &'a self,
            inputs: Example,
        ) -> std::pin::Pin<Box<dyn Future<Output = Result<Prediction>> + Send + 'a>> {
            Box::pin(async move {
                let echoed = inputs.get("request").cloned().unwrap_or(json!(""));
                Ok(Prediction::new(
                    Example::new([("answer", echoed)]),
                    "echoed",
                ))
            })
        }

        fn named_predictors(&mut self) -> Vec<NamedPredictor<'_>> {
            vec![NamedPredictor {
                name: "self".to_owned(),
                signature: &mut self.signature,
                demos: &mut self.demos,
                config: &mut self.config,
                hint: &mut self.hint,
            }]
        }
    }

    fn echo() -> Echo {
        Echo {
            signature: Signature::single_input("Echo the request.", Vec::new()),
            demos: Vec::new(),
            config: LmConfig::default(),
            hint: None,
        }
    }

    #[tokio::test]
    async fn a_module_defined_by_a_caller_runs() {
        let module = echo();
        let prediction = module
            .forward(example! { request: "hello" }.with_inputs(["request"]))
            .await
            .expect("echo succeeds");
        assert_eq!(prediction.get("answer"), Some(&json!("hello")));
    }

    #[tokio::test]
    async fn any_module_can_be_held_behind_a_box() {
        // An evaluator and a composed program both need this; a non-object-safe trait would
        // have made both impossible.
        let modules: Vec<Box<dyn Module>> = vec![Box::new(echo())];
        for module in &modules {
            let prediction = module
                .forward(example! { request: "held" }.with_inputs(["request"]))
                .await
                .expect("boxed module runs");
            assert_eq!(prediction.get("answer"), Some(&json!("held")));
        }
    }

    #[test]
    fn an_optimizer_can_write_demos_back_through_the_walk() {
        let mut module = echo();
        for predictor in module.named_predictors() {
            predictor.demos.push(example! { request: "a", answer: "a" });
            predictor.signature.instructions = "Echo it exactly.".to_owned();
        }
        assert_eq!(module.demos.len(), 1);
        assert_eq!(module.signature.instructions, "Echo it exactly.");
    }
}

/// What asking one module answers with.
///
/// dspy has one call spelling across both ways of declaring a task, because every module there
/// answers with the same dynamic `Prediction`. Rust need not give that up to match: the spelling
/// is shared and the answer stays whatever the module promised, so a derived task still hands
/// back its own outputs struct and `result.answer` still means the field.
pub trait Ask {
    type Answer;

    fn ask<'a>(
        &'a self,
        inputs: Example,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<Self::Answer>> + Send + 'a>>;
}

/// Answer with the fields the module parsed, which is what a task declared by its field names
/// has to give.
///
/// Written per module rather than blanket over [`Module`], because a derived task is a `Module`
/// too and answers with something better than a `Prediction`. One blanket impl would make that
/// unreachable.
#[macro_export]
macro_rules! asks_with_a_prediction {
    ($module:ty) => {
        impl $crate::Ask for $module {
            type Answer = $crate::Prediction;

            fn ask<'a>(
                &'a self,
                inputs: $crate::Example,
            ) -> ::std::pin::Pin<
                ::std::boxed::Box<
                    dyn ::std::future::Future<Output = ::anyhow::Result<$crate::Prediction>>
                        + Send
                        + 'a,
                >,
            > {
                $crate::Module::forward(self, inputs)
            }
        }
    };
}
