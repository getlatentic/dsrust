//! A bootstrapped demo belongs to the predictor that earned it, or the program is refused.
//!
//! `Module::forward_traced` records nothing by default, so a module written the way the guide
//! writes one used to be optimizable and unattributable at once: `BootstrapFewShot` filed its demos
//! under the whole program and handed the same list to every predictor, teaching each step the
//! *program's* fields, which its signature may not even have.
//!
//! A predictor now records its own call the way `Predict.__call__` does upstream, so a composed
//! module traces itself and `forward_traced` is for saying something the calls do not — a step no
//! predictor made, or a name other than the predictor's own. What is left unattributable is a
//! program whose predictors never run at all, and that is refused rather than taught nothing.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use anyhow::Result;
use dsrust::lm::DynChatModel;
use dsrust::module::{TraceStep, relabel};
use dsrust::signature::SignatureSpec;
use dsrust::{
    BootstrapFewShot, DummyLM, Example, Module, Predict, Prediction, Signature, exact_match,
    example,
};

#[derive(Signature)]
/// Draft a note from the question.
struct Draft {
    #[input]
    question: String,
    #[output]
    note: String,
}

#[derive(Signature)]
/// Polish the note into an answer.
struct Polish {
    #[input]
    note: String,
    #[output]
    answer: String,
}

/// Two predictors with different fields, written the way the guide writes one.
#[derive(dsrust::Module)]
struct Untraced {
    drafting: Predict,
    polishing: Predict,
}

impl dsrust::Forward for Untraced {
    async fn forward(&self, inputs: Example) -> Result<Prediction> {
        let drafted = self.drafting.forward(inputs).await?;
        let mut next = Example::default();
        next.set("note", drafted.get("note").cloned().unwrap_or_default());
        self.polishing.forward(next).await
    }
}

/// The same pipeline, passing the trace down and naming each child's steps after the field it
/// lives in — which is what `named_predictors` calls them, so the two agree.
struct Traced {
    drafting: Predict,
    polishing: Predict,
}

impl Module for Traced {
    fn forward<'a>(
        &'a self,
        inputs: Example,
    ) -> Pin<Box<dyn Future<Output = Result<Prediction>> + Send + 'a>> {
        Box::pin(async move {
            let drafted = self.drafting.forward(inputs).await?;
            let mut next = Example::default();
            next.set("note", drafted.get("note").cloned().unwrap_or_default());
            self.polishing.forward(next).await
        })
    }

    fn named_predictors(&mut self) -> Vec<dsrust::NamedPredictor<'_>> {
        let mut all = Vec::new();
        for (name, predictor) in [
            ("drafting", &mut self.drafting),
            ("polishing", &mut self.polishing),
        ] {
            for mut inner in predictor.named_predictors() {
                inner.name = name.to_owned();
                all.push(inner);
            }
        }
        all
    }

    fn forward_traced<'a>(
        &'a self,
        inputs: Example,
        trace: &'a mut Vec<TraceStep>,
    ) -> Pin<Box<dyn Future<Output = Result<Prediction>> + Send + 'a>> {
        Box::pin(async move {
            let mark = trace.len();
            let drafted = self.drafting.forward_traced(inputs, trace).await?;
            relabel(trace, mark, "drafting");

            let mut next = Example::default();
            next.set("note", drafted.get("note").cloned().unwrap_or_default());
            let mark = trace.len();
            let answered = self.polishing.forward_traced(next, trace).await?;
            relabel(trace, mark, "polishing");
            Ok(answered)
        })
    }
}

fn scripted(field: &'static str, value: &'static str) -> Arc<dyn DynChatModel> {
    Arc::new(DummyLM::new(std::iter::repeat_n(
        Example::new([(field, serde_json::json!(value))]),
        8,
    ))) as Arc<dyn DynChatModel>
}

fn trainset() -> Vec<Example> {
    vec![
        example! { question: "capital of France?", answer: "Paris" }.with_inputs(["question"]),
        example! { question: "capital of Japan?", answer: "Paris" }.with_inputs(["question"]),
    ]
}

/// A pipeline that implements no `forward_traced` is attributed anyway.
///
/// It used to compile, report success, and leave `drafting` — whose signature is
/// `question -> note` — holding demos of `question` and `answer`, never having seen a `note`.
#[tokio::test]
async fn a_pipeline_that_traces_nothing_itself_is_still_attributed() {
    let mut program = Untraced {
        drafting: Predict::from_signature(Draft::signature()).set_lm(scripted("note", "a note")),
        polishing: Predict::from_signature(Polish::signature()).set_lm(scripted("answer", "Paris")),
    };
    BootstrapFewShot::new(exact_match)
        .compile(&mut program, &trainset())
        .await
        .expect("a module that records nothing of its own is traced by its predictors");
    each_predictor_earned_its_own_fields(&mut program);
}

/// A program whose predictors never run has demos belonging to nobody, and is refused.
///
/// The one shape ambient recording cannot rescue: `named_predictors` reports two and neither is
/// asked anything, so the whole-program demo is all there is and it is not either predictor's.
#[tokio::test]
async fn a_program_whose_predictors_never_run_is_refused() {
    struct Bypassing {
        drafting: Predict,
        polishing: Predict,
    }
    impl Module for Bypassing {
        fn forward<'a>(
            &'a self,
            _inputs: Example,
        ) -> Pin<Box<dyn Future<Output = Result<Prediction>> + Send + 'a>> {
            // Answers without asking either predictor, the way a module wrapping a call of its own
            // would.
            Box::pin(async move { Ok(Prediction::new(example! { answer: "Paris" }, "")) })
        }
        fn named_predictors(&mut self) -> Vec<dsrust::NamedPredictor<'_>> {
            let mut all = Vec::new();
            for (name, predictor) in [
                ("drafting", &mut self.drafting),
                ("polishing", &mut self.polishing),
            ] {
                for mut inner in predictor.named_predictors() {
                    inner.name = name.to_owned();
                    all.push(inner);
                }
            }
            all
        }
    }

    let mut program = Bypassing {
        drafting: Predict::from_signature(Draft::signature()),
        polishing: Predict::from_signature(Polish::signature()),
    };
    let refused = BootstrapFewShot::new(exact_match)
        .compile(&mut program, &trainset())
        .await
        .expect_err("a program that cannot attribute a demo is refused");
    let said = refused.to_string();
    assert!(said.contains("forward_traced"), "it names the seam: {said}");
    assert!(said.contains("2 predictors"), "it says how many: {said}");
}

/// Every augmented demo a predictor holds carries that predictor's own fields.
fn each_predictor_earned_its_own_fields<M: Module + ?Sized>(program: &mut M) {
    for predictor in program.named_predictors() {
        let wanted: Vec<String> = predictor
            .signature
            .inputs
            .iter()
            .map(|field| field.name.clone())
            .chain(
                predictor
                    .signature
                    .outputs
                    .iter()
                    .map(|field| field.name.clone()),
            )
            .collect();
        let augmented: Vec<&Example> = predictor
            .demos
            .iter()
            .filter(|demo| demo.get("augmented").is_some())
            .collect();
        assert!(
            !augmented.is_empty(),
            "{} earned no demos of its own",
            predictor.name
        );
        for demo in augmented {
            for field in &wanted {
                assert!(
                    demo.get(field).is_some(),
                    "{} is taught a demo with no `{field}` in it",
                    predictor.name,
                );
            }
        }
    }
}

/// The same pipeline, traced: each predictor is taught demos carrying its own fields.
///
/// The positive half. Refusing the broken shape says nothing about whether the right one works,
/// and it is the right one that has to keep working.
#[tokio::test]
async fn a_traced_pipeline_teaches_each_predictor_its_own_fields() {
    let mut program = Traced {
        drafting: Predict::from_signature(Draft::signature()).set_lm(scripted("note", "a note")),
        polishing: Predict::from_signature(Polish::signature()).set_lm(scripted("answer", "Paris")),
    };
    BootstrapFewShot::new(exact_match)
        .compile(&mut program, &trainset())
        .await
        .expect("a traced pipeline compiles");

    each_predictor_earned_its_own_fields(&mut program);
}

/// A single predictor that records no trace keeps the shortcut it always had.
///
/// The fallback is exact there — one predictor's demos and the program's are the same list — and
/// refusing it would break every plain `Predict` a caller optimizes.
#[tokio::test]
async fn one_predictor_needs_no_trace() {
    let mut program =
        Predict::from_signature(Draft::signature()).set_lm(scripted("note", "a note"));
    BootstrapFewShot::new(exact_match)
        .compile(&mut program, &trainset())
        .await
        .expect("one predictor compiles without a trace");
}

/// Nothing to attribute is not the same as nothing traced: a run where no example solved has no
/// program-level demos either, so there is nothing to refuse.
#[tokio::test]
async fn a_pipeline_that_solved_nothing_is_not_refused() {
    let mut program = Untraced {
        drafting: Predict::from_signature(Draft::signature()).set_lm(scripted("note", "a note")),
        polishing: Predict::from_signature(Polish::signature())
            .set_lm(scripted("answer", "not the answer")),
    };
    BootstrapFewShot::new(exact_match)
        .compile(&mut program, &trainset())
        .await
        .expect("a program that earned nothing has nothing to misattribute");
}

/// Naming and running are separate calls because they need different borrows.
///
/// `predictor_names` walks the program and needs `&mut self`; a run holds `&self`, and GEPA runs a
/// whole batch concurrently. So an optimizer takes the names once and hands the same
/// [`PredictorNames`] to every run, which is what upstream does with `predictor2name`.
#[tokio::test]
async fn names_are_taken_once_and_used_by_many_runs() {
    let mut program = Untraced {
        drafting: Predict::from_signature(Draft::signature()).set_lm(scripted("note", "a note")),
        polishing: Predict::from_signature(Polish::signature()).set_lm(scripted("answer", "Paris")),
    };
    let names: dsrust::module::PredictorNames = program.predictor_names();

    // Two runs against one set of names, from a shared borrow — the shape GEPA needs.
    let asked = || Example::new([("question", serde_json::json!("q"))]);
    let held = &program;
    let (first, second) = tokio::join!(
        held.traced_with(&names, asked()),
        held.traced_with(&names, asked())
    );
    for (answered, trace) in [first, second] {
        answered.expect("it runs");
        let named: Vec<&str> = trace.iter().map(|step| step.predictor.as_str()).collect();
        assert_eq!(named, ["drafting", "polishing"], "in call order, by name");
    }

    // And the trace comes back from a run that failed, which is the one a reflection reads.
    let mut empty = Untraced {
        drafting: Predict::from_signature(Draft::signature()),
        polishing: Predict::from_signature(Polish::signature()),
    };
    let (answered, trace) = empty.traced(asked()).await;
    assert!(answered.is_err(), "no model is configured for it");
    assert!(trace.is_empty(), "nothing ran, so nothing was recorded");
}
