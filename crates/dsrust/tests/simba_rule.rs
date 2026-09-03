//! `append_a_rule` end to end: the crate's own SIMBA writes a rule and keeps the old instruction.
//!
//! The tests in `simba_conformance.rs` hold the *gate* to dspy's — which shapes open it — by reading
//! the golden. This one drives the crate: a scripted model answers both the task calls and the
//! advice ask, and what is asserted is what the compiled program's instructions say afterwards.
//!
//! Without it the gate could be right and the strategy still do nothing, which is exactly the state
//! this crate was in before the model call was wired: the arm appeared in every step's trace and
//! never applied.

use std::sync::{Arc, Mutex};

use dsrust::example;
use dsrust::lm::{dummy::DummyLM, global};
use dsrust::optimize::simba::search::{Simba, Strategy};
use dsrust::{Example, Module, Prediction};
use serde_json::Value;

static GLOBAL_LM: Mutex<()> = Mutex::new(());

fn install(lm: Arc<DummyLM>) -> std::sync::MutexGuard<'static, ()> {
    let guard = GLOBAL_LM
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    global::configure_model(reqwest::Client::new(), lm);
    guard
}

const ADVICE: &str = "Name the seat of government, not the largest city.";

fn trainset() -> Vec<Example> {
    [
        ("capital of France?", "Paris"),
        ("capital of Spain?", "Madrid"),
        ("capital of Italy?", "Rome"),
        ("capital of Japan?", "Tokyo"),
    ]
    .into_iter()
    .map(|(question, answer)| {
        Example::new([
            ("question", Value::from(question)),
            ("answer", Value::from(answer)),
        ])
        .with_inputs(["question"])
    })
    .collect()
}

/// A graded metric, which is what lets the rule gate open at all.
///
/// `append_a_rule` declines a bucket whose best is at or below the batch's 10th percentile *or*
/// whose worst is at or above the 90th. Under a binary metric every bucket is at one extreme or the
/// other, so the arm never fires — which is why dspy's own recorded run never applied it either.
/// Half marks put a bucket strictly between the two.
fn graded(example: &Example, prediction: &Prediction) -> f64 {
    let expected = example.get("answer").and_then(Value::as_str).unwrap_or("");
    let answered = prediction
        .get("answer")
        .and_then(Value::as_str)
        .unwrap_or("");
    if answered == expected {
        return 1.0;
    }
    match answered.is_empty() {
        true => 0.0,
        // A near miss — the first letter agrees — is worth half, which is what puts one bucket
        // between the batch's extremes.
        false => match answered.chars().next() == expected.chars().next() {
            true => 0.5,
            false => 0.0,
        },
    }
}

/// A rule written by the model reaches the compiled program's instructions, after the original.
#[allow(clippy::await_holding_lock)] // the installer's own note: `SERIAL` is a test token, taken by nothing under test
#[tokio::test]
async fn a_written_rule_is_appended_to_the_instruction() {
    let lm = Arc::new(DummyLM::keyed([
        // The task, answered right for two of the four so a bucket's scores vary.
        ("capital of France?", example! { answer: "Paris" }),
        // A near miss, worth half — the bucket that opens the gate.
        ("capital of Spain?", example! { answer: "Murcia" }),
        ("capital of Italy?", example! { answer: "Rome" }),
        ("capital of Japan?", example! { answer: "Osaka" }),
        // The advice ask, keyed on a field marker no task call carries. Two things decide this
        // key. The advice prompt embeds the trajectory, so it contains the questions too, and
        // `keyed` walks a `BTreeMap` first-match-wins — so the key has to sort ahead of
        // `"capital of ..."`, which `[` does. And `last_message` is the *user* turn, so a key
        // taken from the system message would never match at all.
        (
            "[[ ## oracle_metadata ## ]]",
            example! {
                discussion: "The worse run named the largest city.",
                module_advice: serde_json::json!({ "self": ADVICE })
            },
        ),
    ]));
    let _guard = install(lm.clone());

    let mut student = dsrust::Predict::parse("question -> answer").expect("parses");
    let before = student.signature.instructions.clone();

    let simba = Simba {
        // The whole trainset in one batch, so all four score levels are in the percentiles.
        bsize: 4,
        num_candidates: 2,
        max_steps: 2,
        // Zero leaves `append_a_rule` as the only strategy, so every bucket takes it — which is
        // what makes this a test of the rule rather than of the coin flip.
        max_demos: 0,
        seed: 0,
        ..Simba::new(graded)
    };
    let compiled = simba
        .compile_traced(&mut student, &trainset())
        .await
        .expect("the search runs");
    let steps = &compiled.steps;

    assert!(
        steps
            .iter()
            .flat_map(|step| &step.strategies)
            .all(|(strategy, _)| *strategy == Strategy::AppendARule),
        "max_demos = 0 leaves only the rule strategy"
    );
    let applied = steps
        .iter()
        .flat_map(|step| &step.strategies)
        .filter(|(_, applied)| *applied)
        .count();
    assert!(
        applied > 0,
        "no bucket wrote a rule, so this says nothing about the model call"
    );

    // The ask reached the model, which is the half that was missing before this was wired.
    let asked = lm.asked();
    let advice_ask = asked
        .iter()
        .find(|ask| ask.last_message().contains("[[ ## oracle_metadata ## ]]"))
        .expect("no OfferFeedback ask was made");
    let sent = advice_ask.last_message();

    // And its bytes: the trajectory is two-space JSON, which is `orjson.dumps(..., OPT_INDENT_2)`.
    assert!(
        sent.contains("[[ ## better_program_trajectory ## ]]"),
        "the ask is missing a field: {sent}"
    );
    // The scripted model answers the same however the prompt changes, so both runs of an example
    // score identically and dspy's **tie arm** fires: the better side is blanked — empty
    // trajectory, `{"N/A": "Prediction not available"}` outputs, and the *string* `N/A` in a field
    // the signature declares `float`. All three reach the model, and all three are upstream's.
    assert!(
        sent.contains("[[ ## better_program_trajectory ## ]]\n[]"),
        "the blanked side should send an empty trajectory: {sent}"
    );
    assert!(
        sent.contains("\"N/A\": \"Prediction not available\""),
        "the blanked side's outputs are dspy's placeholder: {sent}"
    );
    assert!(
        sent.contains("[[ ## better_reward_value ## ]]\nN/A"),
        "the blanked score reaches a float field as the string N/A: {sent}"
    );
    // The side that was *not* blanked carries the real trajectory, as two-space JSON —
    // `orjson.dumps(..., OPT_INDENT_2)`.
    assert!(
        sent.contains("\n    \"module_name\": \"self\""),
        "the kept trajectory reached the model unindented: {sent}"
    );
    assert!(
        sent.contains("capital of"),
        "the kept trajectory carries no inputs: {sent}"
    );

    // The compiled winner need not carry the advice, and does not here: a scripted model answers
    // the same however the prompt changes, so every candidate ties with the baseline and
    // `best_index` keeps the first — which is upstream's behaviour too. What is asserted is that
    // the rule was written, not that the search chose to keep it.
    assert!(
        student.signature.instructions.starts_with(&before),
        "the original instruction was replaced rather than added to"
    );

    // The advised candidate itself. It need not reach the final slate — under a scripted model
    // every candidate ties with the baseline — so it is looked for among the candidates each step
    // built, which is where a strategy's effect is visible at all.
    // Without this the append is untested: substituting the advice for the instruction instead of
    // appending it passes every other assertion here, because the discarded candidate is the only
    // place the difference shows.
    let advised = steps
        .iter()
        .flat_map(|step| &step.candidates)
        .map(|state| {
            let mut program = dsrust::Predict::parse("question -> answer").expect("parses");
            program.load_state(state).expect("a candidate loads");
            program.signature.instructions
        })
        .find(|instructions| instructions.contains(ADVICE))
        .expect("no candidate built in any step carries the advice");
    assert_eq!(
        advised,
        format!("{before}\n\n{ADVICE}"),
        "the advice replaced the instruction rather than following it after a blank line"
    );
}

/// A model that names no module leaves every instruction alone, and the strategy says so.
#[allow(clippy::await_holding_lock)] // the installer's own note: `SERIAL` is a test token, taken by nothing under test
#[tokio::test]
async fn advice_for_no_module_changes_nothing() {
    let lm = Arc::new(DummyLM::keyed([
        ("capital of France?", example! { answer: "Paris" }),
        // A near miss, worth half — the bucket that opens the gate.
        ("capital of Spain?", example! { answer: "Murcia" }),
        ("capital of Italy?", example! { answer: "Rome" }),
        ("capital of Japan?", example! { answer: "Osaka" }),
        (
            "[[ ## oracle_metadata ## ]]",
            example! {
                discussion: "Nothing to say.",
                module_advice: serde_json::json!({ "some_other_module": ADVICE })
            },
        ),
    ]));
    let _guard = install(lm.clone());

    let mut student = dsrust::Predict::parse("question -> answer").expect("parses");
    let before = student.signature.instructions.clone();
    let simba = Simba {
        // The whole trainset in one batch, so all four score levels are in the percentiles.
        bsize: 4,
        num_candidates: 2,
        max_steps: 2,
        max_demos: 0,
        seed: 0,
        ..Simba::new(graded)
    };
    let compiled = simba
        .compile_traced(&mut student, &trainset())
        .await
        .expect("the search runs");
    let steps = &compiled.steps;

    assert!(
        steps
            .iter()
            .flat_map(|step| &step.strategies)
            .all(|(_, applied)| !applied),
        "advice naming no predictor should not count as applied"
    );
    assert_eq!(
        student.signature.instructions, before,
        "an unnamed module's instruction was changed"
    );
}
