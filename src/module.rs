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
use crate::signature::Signature;

/// One predictor inside a program: its signature and its demos, borrowed for inspection or
/// mutation. An optimizer's whole job is reading these and writing back better ones.
pub struct NamedPredictor<'a> {
    pub name: String,
    pub signature: &'a mut Signature,
    pub demos: &'a mut Vec<Example>,
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
            }]
        }
    }

    fn echo() -> Echo {
        Echo {
            signature: Signature::single_input("Echo the request.", Vec::new()),
            demos: Vec::new(),
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
