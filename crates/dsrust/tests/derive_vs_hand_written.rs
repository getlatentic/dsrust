//! What `#[derive(Module)]` writes that a hand-written `impl Module` does not.
//!
//! Both are legal, and the guide says so. They are not equivalent: the derive writes the
//! observability point and the optimizer's walk as well as the boxing, and a hand-written impl gets
//! the trait's defaults for both — `named_predictors` returns nothing, so an optimizer rewrites
//! nothing and reports success. This pins the difference so the guide can state it.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use dsrust::callback::{CallId, Callback, configure_callbacks};
use dsrust::lm::DynChatModel;
use dsrust::{
    DummyLM, Example, Forward, Module, Predict, Prediction, Signature, example, signature,
};

#[derive(Signature)]
/// Answer the question.
struct QA {
    #[input]
    question: String,
    #[output]
    answer: String,
}

fn predictor() -> Predict {
    use signature::SignatureSpec;
    Predict::from_signature(QA::signature()).set_lm(Arc::new(DummyLM::new([
        example! { answer: "Paris" },
        example! { answer: "Paris" },
    ])) as Arc<dyn DynChatModel>)
}

#[derive(Module)]
struct Derived {
    plan: Predict,
    write: Predict,
}

impl Forward for Derived {
    async fn forward(&self, inputs: Example) -> Result<Prediction> {
        let _ = self.plan.forward(inputs.clone()).await?;
        self.write.forward(inputs).await
    }
}

struct HandWritten {
    plan: Predict,
    write: Predict,
}

impl Module for HandWritten {
    fn forward<'a>(
        &'a self,
        inputs: Example,
    ) -> Pin<Box<dyn Future<Output = Result<Prediction>> + Send + 'a>> {
        Box::pin(async move {
            let _ = self.plan.forward(inputs.clone()).await?;
            self.write.forward(inputs).await
        })
    }
}

#[derive(Default)]
struct Recorder {
    modules: Mutex<Vec<String>>,
}

impl Callback for Recorder {
    fn on_module_start(&self, _call: &CallId, module: &str, _inputs: &Example) {
        self.modules
            .lock()
            .expect("not poisoned")
            .push(module.to_owned());
    }
}

#[tokio::test]
async fn the_derive_writes_the_point_and_the_walk() {
    let recorder = Arc::new(Recorder::default());
    configure_callbacks([recorder.clone() as Arc<dyn Callback>]);

    let mut derived = Derived {
        plan: predictor(),
        write: predictor(),
    };
    Module::forward(
        &derived,
        Example::new([("question", serde_json::json!("q"))]),
    )
    .await
    .expect("it runs");
    let mut hand = HandWritten {
        plan: predictor(),
        write: predictor(),
    };
    hand.forward(Example::new([("question", serde_json::json!("q"))]))
        .await
        .expect("it runs");
    configure_callbacks([]);

    // The derive opens the module point; the hand-written one never announces itself, so a trace
    // shows its two `Predict` children with nothing above them.
    let seen = recorder.modules.lock().expect("not poisoned").clone();
    assert_eq!(seen.iter().filter(|name| *name == "Derived").count(), 1);
    assert_eq!(seen.iter().filter(|name| *name == "HandWritten").count(), 0);
    assert_eq!(seen.iter().filter(|name| *name == "Predict").count(), 4);

    // And the walk an optimizer works through. This is the one that fails quietly: a hand-written
    // module takes the trait's empty default, so `compile` rewrites nothing and reports success.
    assert_eq!(derived.named_predictors().len(), 2);
    assert_eq!(hand.named_predictors().len(), 0);
}
