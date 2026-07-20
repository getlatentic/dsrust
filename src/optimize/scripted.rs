//! Programs a compile can be run against without a provider: they answer from a table and
//! record what each call saw, so a test can assert on the decisions an optimizer made rather
//! than on a model's mood.

use std::pin::Pin;
use std::sync::Mutex;

use anyhow::{Result, anyhow};
use serde_json::{Value, json};

use crate::example;
use crate::example::{Example, Prediction};
use crate::module::{Module, NamedPredictor};
use crate::signature::Signature;

/// What a program was looking at when it was asked one question.
#[derive(Clone)]
pub(super) struct Call {
    pub(super) question: String,
    /// The demos the predictor held at the moment of the call. dspy hides the example being
    /// solved for exactly this window and puts it back afterwards, so a test that only reads
    /// the demos after a compile cannot see the difference.
    pub(super) demos: Vec<Example>,
}

/// How a scripted program answers.
pub(super) enum Answers {
    /// From the capital table, so `exact_match` scores it 1.0.
    Correctly,
    /// The same wrong answer every time.
    Wrongly,
    /// Wrong until the same question has been asked this many times, then right — the shape
    /// `max_rounds` exists for.
    RightOnRound(usize),
    /// Every call fails, which is what the error budget counts.
    Failing,
}

/// One predictor, one rule for answering it.
pub(super) struct Solver {
    signature: Signature,
    pub(super) demos: Vec<Example>,
    answers: Answers,
    calls: Mutex<Vec<Call>>,
}

impl Solver {
    pub(super) fn new(answers: Answers) -> Self {
        Self {
            signature: Signature::single_input("Answer.", Vec::new()),
            demos: Vec::new(),
            answers,
            calls: Mutex::new(Vec::new()),
        }
    }

    pub(super) fn calls(&self) -> Vec<Call> {
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
                });
                before
            };
            let correct = match self.answers {
                Answers::Correctly => true,
                Answers::Wrongly => false,
                Answers::RightOnRound(round) => asked_before + 1 >= round,
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
        }]
    }
}

/// Two predictors, so the decisions `_train` makes per predictor are observable. It answers
/// correctly and records nothing: what is under test is which demos each half ends up with.
pub(super) struct Pair {
    first: Signature,
    pub(super) first_demos: Vec<Example>,
    second: Signature,
    pub(super) second_demos: Vec<Example>,
}

impl Pair {
    pub(super) fn new() -> Self {
        Self {
            first: Signature::single_input("Answer.", Vec::new()),
            first_demos: Vec::new(),
            second: Signature::single_input("Answer.", Vec::new()),
            second_demos: Vec::new(),
        }
    }
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

    fn named_predictors(&mut self) -> Vec<NamedPredictor<'_>> {
        vec![
            NamedPredictor {
                name: "first".to_owned(),
                signature: &mut self.first,
                demos: &mut self.first_demos,
            },
            NamedPredictor {
                name: "second".to_owned(),
                signature: &mut self.second,
                demos: &mut self.second_demos,
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
pub(super) fn trainset() -> Vec<Example> {
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
pub(super) fn answers(demos: &[Example]) -> Vec<String> {
    demos
        .iter()
        .filter_map(|demo| demo.get("answer")?.as_str().map(str::to_owned))
        .collect()
}
