//! GEPA end to end: the crate's optimizer against a scripted task model and reflection model.
//!
//! The comparison against dspy's own runs lives in [`replay`], which asks a different question of
//! a different harness — see its own docstring.
//!
//! The sanity test drives the whole wiring — the [`gepa`] engine over the dsrust adapter — and checks
//! that GEPA reflects the seed instruction into the one that scores, accepts it on the minibatch, and
//! leaves the student holding it. The dspy comparison (`gepa_makes_the_decisions_dspy_makes`) replays
//! runs recorded from dspy's GEPA teleprompter in the regime where the search dynamics decide: the
//! reflection model proposes a *distinct* instruction per ask, each with its own score profile, so
//! whether a proposal survives its seeded minibatch decides what enters the pool — and different
//! seeds genuinely compile different instructions. What is compared is the whole evolution: every
//! candidate in discovery order, its parents, its validation score, the eval bookkeeping, the
//! reflection-call count, and the winner.

use gepa::Candidate;
use std::sync::Arc;

use anyhow::Result;

use super::{Feedback, GEPA, MetricContext};
use crate::example;
use crate::example::{Example, Prediction};
use crate::lm::ChatModel;
use crate::lm::api::{self, LmResponse};
use crate::predict::Predict;

mod replay;

const TABLE: [(&str, &str); 3] = [
    ("capital of France?", "Paris"),
    ("capital of Germany?", "Berlin"),
    ("capital of Spain?", "Madrid"),
];

const PROPOSAL: &str = "Answer with GOOD precision.";

/// The task model: it answers a question correctly only when the instruction in force carries `GOOD`
/// (which it reads from the system prompt), so a candidate carrying `GOOD` outscores the seed.
struct TaskCoach;

impl ChatModel for TaskCoach {
    async fn forward(&self, request: &api::LmRequest) -> Result<LmResponse> {
        let has_good = request.system().contains("GOOD");
        let last = request
            .messages
            .last()
            .and_then(|message| message.text())
            .unwrap_or_default();
        let answer = TABLE
            .iter()
            .find(|(question, _)| last.contains(question))
            .map(|(_, correct)| if has_good { *correct } else { "wrong" })
            .unwrap_or("wrong");
        Ok(LmResponse::text(format!(
            "[[ ## answer ## ]]\n{answer}\n\n[[ ## completed ## ]]"
        )))
    }
}

/// The reflection model: whatever it is shown, it proposes the instruction carrying `GOOD`, wrapped in
/// a fenced block for [`gepa::extract_new_instruction`].
struct Reflector;

impl ChatModel for Reflector {
    async fn forward(&self, _request: &api::LmRequest) -> Result<LmResponse> {
        Ok(LmResponse::text(format!("```\n{PROPOSAL}\n```")))
    }
}

/// A GEPA feedback metric: exact-match on the answer, with a word of feedback either way. The
/// context is unread on purpose — dspy's protocol calls this the program level.
fn metric(gold: &Example, pred: &Prediction, _: &MetricContext<'_>) -> Feedback {
    let correct = gold.get("answer") == pred.get("answer");
    if correct {
        Feedback::new(1.0, "Correct.")
    } else {
        Feedback::new(0.0, "Wrong answer; be more precise.")
    }
}

fn trainset() -> Vec<Example> {
    TABLE
        .iter()
        .map(|(q, a)| example! { question: *q, answer: *a }.with_inputs(["question"]))
        .collect()
}

/// GEPA reflects the seed into the `GOOD` instruction, which scores 100% against the seed's 0%, so the
/// search accepts it and the student is left holding it.
#[tokio::test]
async fn gepa_evolves_the_instruction_that_scores() {
    let task = Arc::new(TaskCoach);
    let mut student = Predict::parse("question -> answer")
        .expect("parses")
        .set_lm(task);
    student.signature.instructions = "Answer the question.".to_owned();

    GEPA::new(metric, Arc::new(Reflector))
        .max_metric_calls(20)
        .reflection_minibatch_size(2)
        .compile(&mut student, &trainset(), &trainset())
        .await
        .expect("compiles");

    assert_eq!(student.signature.instructions, PROPOSAL);
}

/// A caller's proposer replaces the reflection tree entirely — upstream's "overrides everything".
///
/// The reflection model is scripted to a proposal that would win if it were asked. It is not asked:
/// the compiled instruction is the caller's, and the reflector recorded no call.
#[tokio::test]
async fn a_callers_proposer_replaces_the_reflection_tree() {
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Counting(AtomicUsize);
    impl ChatModel for Counting {
        async fn forward(&self, _request: &api::LmRequest) -> Result<LmResponse> {
            self.0.fetch_add(1, Ordering::Relaxed);
            // Bare, not fenced: the reflection tree would strip a fence, and the proposer here
            // uses the reply as written — so a fenced reply would prove nothing about which path
            // produced the instruction.
            Ok(LmResponse::text(PROPOSAL))
        }
    }

    struct Asking;
    impl super::InstructionProposer for Asking {
        fn propose<'a>(
            &'a self,
            reflection: &'a Arc<dyn crate::lm::DynChatModel>,
            candidate: &'a Candidate,
            components: &'a [String],
            datasets: &'a BTreeMap<String, super::ReflectiveDataset>,
        ) -> std::pin::Pin<Box<dyn Future<Output = Candidate> + Send + 'a>> {
            Box::pin(async move {
                let mut proposed = Candidate::new();
                for name in components {
                    // The dataset it was handed is the one the reflection tree would have read.
                    assert!(!datasets[name].is_empty(), "handed an empty dataset");
                    let _ = &candidate[name];
                    // The model GEPA was built with reaches the proposer, which is what upstream's
                    // `with dspy.context(lm=reflection_lm)` gives its own. Asking it is how this
                    // test tells "handed the model" from "handed nothing".
                    let asked = reflection
                        .forward_dyn(
                            &api::LmRequest::from_items("", ["propose something better"])
                                .expect("one string normalises"),
                        )
                        .await
                        .expect("the reflection model answers");
                    proposed.insert(name.clone(), asked.first_text());
                }
                proposed
            })
        }
    }

    let reflector = Arc::new(Counting(AtomicUsize::new(0)));
    let mut student = Predict::parse("question -> answer")
        .expect("parses")
        .set_lm(Arc::new(TaskCoach));
    student.signature.instructions = "Answer the question.".to_owned();

    GEPA::new(metric, reflector.clone())
        .max_metric_calls(20)
        .reflection_minibatch_size(2)
        .instruction_proposer(Arc::new(Asking))
        .compile(&mut student, &trainset(), &trainset())
        .await
        .expect("compiles");

    // Asked by the proposer and by nothing else: GEPA's own reflection tree was not consulted, so
    // every call on the counter is one the caller's proposer made.
    assert!(
        reflector.0.load(Ordering::Relaxed) > 0,
        "the proposer could not reach the reflection model"
    );
    assert_eq!(
        student.signature.instructions, PROPOSAL,
        "the caller's proposal did not reach the student"
    );
}

/// Evaluating several examples at once changes when they run and not what the run decides.
///
/// The claim `num_threads` is documented with. Evaluation is order-preserving, so a trace still
/// lines up with the example that produced it and the reflection reads the same dataset — but
/// nothing said so until this compared the two runs, and `buffer_unordered` would have passed every
/// other test in the file.
#[tokio::test]
async fn a_threaded_evaluation_reaches_the_same_instruction() {
    let mut compiled = Vec::new();
    for threads in [1, 4] {
        let mut student = Predict::parse("question -> answer")
            .expect("parses")
            .set_lm(Arc::new(TaskCoach));
        student.signature.instructions = "Answer the question.".to_owned();

        GEPA::new(metric, Arc::new(Reflector))
            .max_metric_calls(20)
            .reflection_minibatch_size(2)
            .num_threads(threads)
            .compile(&mut student, &trainset(), &trainset())
            .await
            .expect("compiles");
        compiled.push(student.signature.instructions.clone());
    }
    assert_eq!(
        compiled[0], compiled[1],
        "the thread count moved the search"
    );
    assert_eq!(compiled[0], PROPOSAL);
}

/// The `GOOD-<k>` marker in an instruction, if any. Written by hand because the crate carries no
/// regex; the marker is always `GOOD-` followed by digits.
fn marker(text: &str) -> Option<u64> {
    let start = text.find("GOOD-")? + "GOOD-".len();
    let digits: String = text[start..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    digits.parse().ok()
}

/// What a GEPA run puts on the wire, and the one span it does *not* open.
///
/// GEPA scores through its own adapter rather than through [`Evaluate`](crate::Evaluate), so no
/// `evaluate` span is opened — which is the half of the `GEPA.log_dir` reason that was true. The
/// other half said the run emitted nothing at all, and it emits the student's spans like any other
/// `GepaOutcome` reports every field dspy's `DspyGEPAResult` does, and they agree with each other.
///
/// It was missing two the state already had: `val_subscores` — every candidate's per-example
/// scores, which `val_aggregate_scores` is the mean of — and `per_val_instance_best_candidates`,
/// the Pareto front the search itself selects from. A row said the result was "surfaced through
/// the gepa crate's own result type", which was true of nine of its eleven fields.
#[tokio::test]
async fn the_outcome_reports_what_dspys_result_reports() {
    let task = Arc::new(TaskCoach);
    let mut student = Predict::parse("question -> answer")
        .expect("parses")
        .set_lm(task);
    student.signature.instructions = "Answer the question.".to_owned();
    let outcome = GEPA::new(metric, Arc::new(Reflector))
        .max_metric_calls(20)
        .reflection_minibatch_size(2)
        .compile(&mut student, &trainset(), &trainset())
        .await
        .expect("compiles");

    let rows = trainset().len();
    assert_eq!(outcome.val_subscores.len(), outcome.candidates.len());
    for (candidate, scores) in outcome.val_subscores.iter().enumerate() {
        assert_eq!(
            scores.len(),
            rows,
            "candidate {candidate} was scored on every row"
        );
        let mean = scores.iter().sum::<f64>() / rows as f64;
        assert!(
            (mean - outcome.val_aggregate_scores[candidate]).abs() < 1e-12,
            "candidate {candidate}: the aggregate is the mean of the subscores"
        );
    }

    assert_eq!(outcome.per_val_instance_best_candidates.len(), rows);
    for (row, front) in outcome.per_val_instance_best_candidates.iter().enumerate() {
        assert!(!front.is_empty(), "row {row} has an empty front");
        let best = front
            .iter()
            .map(|&candidate| outcome.val_subscores[candidate][row])
            .fold(f64::NEG_INFINITY, f64::max);
        for (candidate, scores) in outcome.val_subscores.iter().enumerate() {
            assert!(
                scores[row] <= best,
                "row {row}: candidate {candidate} beats its own front"
            );
        }
    }
}

/// gepa's `track_best_outputs`: what the best programs answered on each validation example.
///
/// Off by default and reporting nothing, on by request and reporting one list per example — every
/// program on that example's Pareto front, paired with what it answered there. The seed is program
/// 0 and starts on every front, so an example nothing beats it on still reports its answer.
///
/// The ledger said `GepaOutcome` already carried this. It carried the best *candidate*, which is a
/// map of component instructions; nothing kept a prediction past the score it earned.
#[tokio::test]
async fn tracking_best_outputs_reports_what_each_front_answered() {
    let run = |track: bool| async move {
        let task = Arc::new(TaskCoach);
        let mut student = Predict::parse("question -> answer")
            .expect("parses")
            .set_lm(task);
        student.signature.instructions = "Answer the question.".to_owned();
        GEPA::new(metric, Arc::new(Reflector))
            .max_metric_calls(20)
            .reflection_minibatch_size(2)
            .track_best_outputs(track)
            .compile(&mut student, &trainset(), &trainset())
            .await
            .expect("compiles")
    };

    assert!(
        run(false).await.best_outputs_valset.is_none(),
        "outputs were kept without being asked for"
    );

    let outcome = run(true).await;
    let tracked = outcome
        .best_outputs_valset
        .expect("asked for the best outputs");
    assert_eq!(
        tracked.len(),
        trainset().len(),
        "one list per valset example"
    );
    for (example, answers) in tracked.iter().enumerate() {
        assert!(
            !answers.is_empty(),
            "example {example} has a front and nothing answered it"
        );
        for (program, _) in answers {
            assert!(
                *program < outcome.candidates.len(),
                "example {example} names program {program}, and the run has {} of them",
                outcome.candidates.len()
            );
        }
    }
}

/// caller. Both halves are held here rather than read off the source.
#[test]
fn a_gepa_run_opens_the_students_spans_and_no_evaluate_span() {
    let opened = crate::observe::spans_opened_by(async {
        let task = Arc::new(TaskCoach);
        let mut student = Predict::parse("question -> answer")
            .expect("parses")
            .set_lm(task);
        student.signature.instructions = "Answer the question.".to_owned();
        GEPA::new(metric, Arc::new(Reflector))
            .max_metric_calls(20)
            .reflection_minibatch_size(2)
            .compile(&mut student, &trainset(), &trainset())
            .await
            .expect("compiles");
    });

    assert_eq!(
        opened.get("evaluate"),
        None,
        "GEPA scored through `Evaluate` after all: {opened:?}"
    );
    for beneath in ["module", "lm", "adapter"] {
        assert!(
            opened.get(beneath).is_some_and(|&count| count > 0),
            "a GEPA run ran the student and opened no {beneath} span: {opened:?}"
        );
    }
}
