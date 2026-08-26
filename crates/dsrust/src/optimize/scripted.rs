//! Programs a compile can be run against without a provider: they answer from a table and
//! record what each call saw, so a test can assert on the decisions an optimizer made rather
//! than on a model's mood.

use std::pin::Pin;
use std::sync::Mutex;

use anyhow::{Result, anyhow};
use serde_json::{Value, json};

use crate::example;
use crate::example::{Example, Prediction};
use crate::lm::Sampling;
use crate::module::{Module, NamedPredictor, TraceStep};
use crate::signature::Signature;

/// What a program was looking at when it was asked one question.
#[derive(Clone)]
pub(crate) struct Call {
    pub(crate) question: String,
    /// The demos the predictor held at the moment of the call. dspy hides the example being
    /// solved for exactly this window and puts it back afterwards, so a test that only reads
    /// the demos after a compile cannot see the difference.
    pub(crate) demos: Vec<Example>,
    /// What the predictor was asked to sample with at that moment, which is how a test sees a
    /// bootstrap round after the first arriving as a fresh rollout.
    pub(crate) config: Sampling,
    /// What the predictor was told to do differently, which is how a `Refine` test sees one
    /// attempt's advice arriving as the next attempt's hint.
    pub(crate) hint: Option<String>,
}

/// How a scripted program answers.
pub(crate) enum Answers {
    /// From the capital table, so `exact_match` scores it 1.0.
    Correctly,
    /// The same wrong answer every time.
    Wrongly,
    /// Wrong until the same question has been asked this many times, then right — the shape
    /// `max_rounds` exists for.
    RightOnRound(usize),
    /// Right on exactly this attempt and wrong on every other, so the best answer is not the
    /// last one. Without that a "keep the best" rule and a "keep the last" rule agree, and a
    /// test cannot tell which one is implemented.
    RightOnlyOnRound(usize),
    /// Answers (wrongly) for this many attempts and then fails every time after.
    ///
    /// A run where everything fails cannot tell an index-against-budget rule from a plain failure
    /// count — both give out on the same attempt. Successes early are what separate them, since
    /// they advance the index without spending the budget.
    FailingAfter(usize),
    /// Every call fails, which is what the error budget counts.
    Failing,
}

/// One predictor, one rule for answering it.
pub(crate) struct Solver {
    /// This double's pinned model, which a saved program records. Never set — it is here
    /// because a `NamedPredictor` names one and two of them cannot borrow the same slot.
    saved_lm_0: Option<std::sync::Arc<dyn crate::lm::DynChatModel>>,
    signature: Signature,
    config: Sampling,
    hint: Option<String>,
    pub(crate) demos: Vec<Example>,
    answers: Answers,
    calls: Mutex<Vec<Call>>,
}

impl Solver {
    pub(crate) fn new(answers: Answers) -> Self {
        Self {
            saved_lm_0: None,
            signature: Signature::single_input("Answer.", Vec::new()),
            config: Sampling::default(),
            hint: None,
            demos: Vec::new(),
            answers,
            calls: Mutex::new(Vec::new()),
        }
    }

    pub(crate) fn calls(&self) -> Vec<Call> {
        self.calls.lock().expect("not poisoned").clone()
    }
}

/// The capital table, or a wrong answer that is wrong the same way every time.
fn answer(question: &str, correct: bool) -> &'static str {
    match (correct, question.contains("France")) {
        (false, _) => "no idea",
        (true, true) => "Paris",
        (true, false) => "Berlin",
    }
}

impl Module for Solver {
    fn forward<'a>(
        &'a self,
        inputs: Example,
    ) -> Pin<Box<dyn Future<Output = Result<Prediction>> + Send + 'a>> {
        Box::pin(async move {
            let question = inputs
                .get("question")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            let asked_before = {
                let mut calls = self.calls.lock().expect("not poisoned");
                let before = calls
                    .iter()
                    .filter(|call| call.question == question)
                    .count();
                calls.push(Call {
                    question: question.clone(),
                    demos: self.demos.clone(),
                    config: self.config.clone(),
                    hint: self.hint.clone(),
                });
                before
            };
            let correct = match self.answers {
                Answers::Correctly => true,
                Answers::Wrongly => false,
                Answers::RightOnRound(round) => asked_before + 1 >= round,
                Answers::RightOnlyOnRound(round) => asked_before + 1 == round,
                Answers::FailingAfter(good) if asked_before >= good => {
                    return Err(anyhow!("the provider is down"));
                }
                Answers::FailingAfter(_) => false,
                Answers::Failing => return Err(anyhow!("the provider is down")),
            };
            Ok(Prediction::new(
                Example::new([("answer", json!(answer(&question, correct)))]),
                "raw",
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
            lm: &mut self.saved_lm_0,
        }]
    }
}

/// Two predictors, so the decisions `_train` makes per predictor are observable. It answers
/// correctly and records nothing: what is under test is which demos each half ends up with.
pub(crate) struct Pair {
    /// This double's pinned model, which a saved program records. Never set — it is here
    /// because a `NamedPredictor` names one and two of them cannot borrow the same slot.
    saved_lm_0: Option<std::sync::Arc<dyn crate::lm::DynChatModel>>,
    /// This double's pinned model, which a saved program records. Never set — it is here
    /// because a `NamedPredictor` names one and two of them cannot borrow the same slot.
    saved_lm_1: Option<std::sync::Arc<dyn crate::lm::DynChatModel>>,
    first_sampling: Sampling,
    first_hint: Option<String>,
    second_sampling: Sampling,
    second_hint: Option<String>,
    first: Signature,
    pub(crate) first_demos: Vec<Example>,
    second: Signature,
    pub(crate) second_demos: Vec<Example>,
}

impl Pair {
    pub(crate) fn new() -> Self {
        Self {
            saved_lm_0: None,
            saved_lm_1: None,
            first: Signature::single_input("Answer.", Vec::new()),
            first_demos: Vec::new(),
            first_sampling: Sampling::default(),
            first_hint: None,
            second: Signature::single_input("Answer.", Vec::new()),
            second_demos: Vec::new(),
            second_sampling: Sampling::default(),
            second_hint: None,
        }
    }
}

/// The intermediate the first half hands the second, so a demo earned by one is not a demo the
/// other could have earned. Interchangeable halves would hide misattribution rather than show it.
fn drafted(answer: &str) -> String {
    format!("draft: {answer}")
}

impl Module for Pair {
    fn forward<'a>(
        &'a self,
        inputs: Example,
    ) -> Pin<Box<dyn Future<Output = Result<Prediction>> + Send + 'a>> {
        Box::pin(async move {
            let question = inputs
                .get("question")
                .and_then(Value::as_str)
                .unwrap_or_default();
            Ok(Prediction::new(
                Example::new([("answer", json!(answer(question, true)))]),
                "raw",
            ))
        })
    }

    /// Two calls: the first drafts from the question, the second answers from the draft.
    fn forward_traced<'a>(
        &'a self,
        inputs: Example,
        trace: &'a mut Vec<TraceStep>,
    ) -> Pin<Box<dyn Future<Output = Result<Prediction>> + Send + 'a>> {
        Box::pin(async move {
            let prediction = self.forward(inputs.clone()).await?;
            let answered = prediction
                .get("answer")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            let draft = drafted(&answered);

            trace.push(TraceStep {
                predictor: "first".to_owned(),
                inputs,
                outputs: Example::new([("draft", json!(draft))]),
                signature: self.first.clone(),
            });
            trace.push(TraceStep {
                predictor: "second".to_owned(),
                inputs: Example::new([("draft", json!(draft))]),
                outputs: Example::new([("answer", json!(answered))]),
                signature: self.second.clone(),
            });
            Ok(prediction)
        })
    }

    fn named_predictors(&mut self) -> Vec<NamedPredictor<'_>> {
        vec![
            NamedPredictor {
                name: "first".to_owned(),
                signature: &mut self.first,
                demos: &mut self.first_demos,
                config: &mut self.first_sampling,
                hint: &mut self.first_hint,
                lm: &mut self.saved_lm_0,
            },
            NamedPredictor {
                name: "second".to_owned(),
                signature: &mut self.second,
                demos: &mut self.second_demos,
                config: &mut self.second_sampling,
                hint: &mut self.second_hint,
                lm: &mut self.saved_lm_1,
            },
        ]
    }
}

/// Two predictors of which only one ever runs, which is what a branch in a program looks like to
/// an optimizer. dspy starts every predictor's traces at an empty list, so the idle half is
/// taught by nothing rather than by its sibling's work.
pub(crate) struct Lopsided {
    /// This double's pinned model, which a saved program records. Never set — it is here
    /// because a `NamedPredictor` names one and two of them cannot borrow the same slot.
    saved_lm_0: Option<std::sync::Arc<dyn crate::lm::DynChatModel>>,
    /// This double's pinned model, which a saved program records. Never set — it is here
    /// because a `NamedPredictor` names one and two of them cannot borrow the same slot.
    saved_lm_1: Option<std::sync::Arc<dyn crate::lm::DynChatModel>>,
    ran_sampling: Sampling,
    ran_hint: Option<String>,
    idle_sampling: Sampling,
    idle_hint: Option<String>,
    ran: Signature,
    pub(crate) ran_demos: Vec<Example>,
    idle: Signature,
    pub(crate) idle_demos: Vec<Example>,
}

impl Lopsided {
    pub(crate) fn new() -> Self {
        Self {
            saved_lm_0: None,
            saved_lm_1: None,
            ran: Signature::single_input("Answer.", Vec::new()),
            ran_demos: Vec::new(),
            ran_sampling: Sampling::default(),
            ran_hint: None,
            idle: Signature::single_input("Answer.", Vec::new()),
            idle_demos: Vec::new(),
            idle_sampling: Sampling::default(),
            idle_hint: None,
        }
    }
}

impl Module for Lopsided {
    fn forward<'a>(
        &'a self,
        inputs: Example,
    ) -> Pin<Box<dyn Future<Output = Result<Prediction>> + Send + 'a>> {
        Box::pin(async move {
            let question = inputs
                .get("question")
                .and_then(Value::as_str)
                .unwrap_or_default();
            Ok(Prediction::new(
                Example::new([("answer", json!(answer(question, true)))]),
                "raw",
            ))
        })
    }

    fn forward_traced<'a>(
        &'a self,
        inputs: Example,
        trace: &'a mut Vec<TraceStep>,
    ) -> Pin<Box<dyn Future<Output = Result<Prediction>> + Send + 'a>> {
        Box::pin(async move {
            let prediction = self.forward(inputs.clone()).await?;
            trace.push(TraceStep {
                predictor: "ran".to_owned(),
                inputs,
                outputs: prediction.example.clone(),
                signature: self.ran.clone(),
            });
            Ok(prediction)
        })
    }

    fn named_predictors(&mut self) -> Vec<NamedPredictor<'_>> {
        vec![
            NamedPredictor {
                name: "ran".to_owned(),
                signature: &mut self.ran,
                demos: &mut self.ran_demos,
                config: &mut self.ran_sampling,
                hint: &mut self.ran_hint,
                lm: &mut self.saved_lm_0,
            },
            NamedPredictor {
                name: "idle".to_owned(),
                signature: &mut self.idle,
                demos: &mut self.idle_demos,
                config: &mut self.idle_sampling,
                hint: &mut self.idle_hint,
                lm: &mut self.saved_lm_1,
            },
        ]
    }
}

/// Two examples the capital table solves, and enough unsolvable ones to leave a validation set
/// behind whenever the bootstrap budget is smaller than the trainset.
///
/// Every example carries a distinct answer, so a test that reports which demos a predictor ended
/// up with can tell any two of them apart. Interchangeable answers hide the difference between a
/// draw that was made once and a draw that was made per predictor.
pub(crate) fn trainset() -> Vec<Example> {
    let mut examples = vec![
        example! { question: "capital of France?", answer: "Paris" }.with_inputs(["question"]),
        example! { question: "capital of Germany?", answer: "Berlin" }.with_inputs(["question"]),
    ];
    examples.extend((0..4).map(|index| {
        example! { question: format!("riddle {index}?"), answer: format!("riddle {index}!") }
            .with_inputs(["question"])
    }));
    examples
}

/// The `answer` field of each demo, which is all most assertions need to identify one.
pub(crate) fn answers(demos: &[Example]) -> Vec<String> {
    demos
        .iter()
        .filter_map(|demo| demo.get("answer")?.as_str().map(str::to_owned))
        .collect()
}
