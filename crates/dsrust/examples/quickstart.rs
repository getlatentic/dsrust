//! The whole loop, with no provider and no API key: declare a task, ask it, score it, compile it.
//!
//!     cargo run --example quickstart
//!
//! The model is a `DummyLM` answering from a table, so this runs offline and always the same way.
//! Point `dsrust::configure` at a real provider and nothing else here changes.

use std::sync::Arc;

use dsrust::{
    BootstrapFewShot, DummyLM, Evaluate, Example, Module, call, configure_model, exact_match,
    example, predict,
};

fn text(record: &Example, field: &str) -> String {
    record
        .get(field)
        .and_then(|value| value.as_str())
        .unwrap_or_default()
        .to_owned()
}

/// Six questions, two of which the stand-in model happens to know.
fn trainset() -> Vec<Example> {
    [
        ("What is the capital of France?", "Paris"),
        ("What is the capital of Japan?", "Tokyo"),
        ("What is the capital of Peru?", "Lima"),
        ("What is the capital of Kenya?", "Nairobi"),
        ("What is the capital of Nepal?", "Kathmandu"),
        ("What is the capital of Chile?", "Santiago"),
    ]
    .into_iter()
    .map(|(question, answer)| {
        example! { question: question, answer: answer }.with_inputs(["question"])
    })
    .collect()
}

/// A model that knows two of the six, so compiling has something to find and something to miss.
fn stand_in() -> DummyLM {
    DummyLM::keyed([
        ("France", example! { answer: "Paris" }),
        ("Japan", example! { answer: "Tokyo" }),
    ])
    .with_fallback(example! { answer: "I don't know" })
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    configure_model(reqwest::Client::new(), Arc::new(stand_in()));

    // 1. Declare a task by naming its fields. The spelling is checked as this file compiles.
    let mut program = predict!("question -> answer");

    // 2. Ask it. Nothing has been learned yet, so the prompt carries no examples.
    let asked = call!(program, question = "What is the capital of France?").await?;
    println!("answer          {}", text(&asked.example, "answer"));
    println!("demos in prompt {}", program.demos.len());

    // 3. Score it over the trainset, which is what an optimizer optimizes against.
    let before = Evaluate::new(trainset(), |inputs| program.forward(inputs), exact_match)
        .run()
        .await;
    println!(
        "before compiling {:.0}% of {} examples",
        before.score * 100.0,
        trainset().len()
    );

    // 4. Compile: run the program, keep the attempts a metric accepts, and write them back as
    //    demos. The program is the same value afterwards, with a better prompt inside it.
    let kept = BootstrapFewShot::new(exact_match)
        .compile(&mut program, &trainset())
        .await?;
    println!("\ncompiled        {kept} of {} solved", trainset().len());
    println!("demos in prompt {}", program.demos.len());
    for (at, demo) in program.demos.iter().enumerate() {
        // The solved ones lead; the rest fill the budget out of the trainset, which is what
        // `max_labeled_demos` is for.
        let earned = if at < kept { "earned  " } else { "labelled" };
        println!(
            "  {earned}      {} -> {}",
            text(demo, "question"),
            text(demo, "answer")
        );
    }

    // 5. What changed is the prompt, not this line. A stand-in model answers from a table and
    //    cannot be taught, so the honest thing to show is that the examples are now in front of
    //    it — with a real provider those examples are what moves the score.
    let again = call!(program, question = "What is the capital of Peru?").await?;
    println!("\nasked again     {}", text(&again.example, "answer"));
    println!("still unknown to a stand-in model, but a real one now sees {kept} solved examples");
    Ok(())
}
