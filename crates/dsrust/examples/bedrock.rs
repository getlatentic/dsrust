//! The quickstart loop against a real model, on Amazon Bedrock's OpenAI-compatible endpoint.
//!
//!     export OPENAI_BASE_URL=https://<your-bedrock-host>/v1   # or OPENAI_API_BASE
//!     export OPENAI_API_KEY=ABSK...                           # a Bedrock API key
//!     cargo run --example bedrock
//!
//! There is no Bedrock-specific code here, and no builder call either: those are the variables
//! litellm reads, this crate reads them too, and Bedrock serves `/v1/chat/completions` in OpenAI's
//! own shape. The same two variables point the same file at Groq, Together, vLLM or LM Studio.
//! `BEDROCK_MODEL` overrides the model; `LM::builder(..).api_base(..).api_key(..)` is there for a
//! program that would rather not read the environment at all.
//!
//! The key goes out as a bearer token. A Bedrock API key — the `ABSK...` kind — is not SigV4-signed,
//! which is why no AWS access key or region appears anywhere in this file.
//!
//! The host is read from the environment rather than written here, because which Bedrock endpoint
//! an account reaches is not this repository's fact to publish.
//!
//! The model id carries its own `openai.` prefix — `openai/openai.gpt-oss-120b` — because the first
//! half is how *this* crate routes to the OpenAI wire and the second is what Bedrock calls the
//! model. `GET /v1/models` lists them.
//!
//! Replies are cached under `~/.dsrs_cache`, as dspy caches under `~/.dspy_cache`. Worth knowing
//! before changing an endpoint and concluding it works: a second run of an unchanged program
//! answers from disk and never opens a socket. `DSRS_CACHEDIR` points it somewhere throwaway.
//!
//! One thing worth knowing before a bill arrives: `gpt-oss` reasons before it answers, and those
//! tokens come out of the same ceiling as the reply. dspy decides between `max_tokens` and
//! `max_completion_tokens` by whether the name starts with `o1`, `o3`, `o4` or `gpt-5`, and
//! `openai.gpt-oss-120b` starts with none of them — so this crate sends `max_tokens`, faithfully.
//! Bedrock accepts either, so the only consequence is that the ceiling has to cover the thinking
//! as well as the answer. A few hundred tokens is not enough; the default below is 4096.

use std::sync::Arc;

use dsrust::{
    BootstrapFewShot, Evaluate, Example, LM, Module, Predict, call, configure_model, exact_match,
    example,
};

const DEFAULT_MODEL: &str = "openai/openai.gpt-oss-120b";

fn text(record: &Example, field: &str) -> String {
    record
        .get(field)
        .and_then(|value| value.as_str())
        .unwrap_or_default()
        .to_owned()
}

/// Six capitals, which a 120b model knows and which therefore make a poor benchmark and a good
/// smoke test: anything less than six out of six is the wiring, not the model.
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

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let model = std::env::var("BEDROCK_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_owned());

    let lm = LM::builder(&model)
        // Room for the reasoning as well as the answer; see the note at the top of this file.
        .max_tokens(4096)
        .build()?;
    configure_model(reqwest::Client::new(), Arc::new(lm));

    let mut program = Predict!("question -> answer");

    let asked = call!(program, question = "What is the capital of France?").await?;
    println!("model           {model}");
    println!("answer          {}", text(&asked.example, "answer"));

    let before = Evaluate::new(trainset(), |inputs| program.forward(inputs), exact_match)
        .run()
        .await?;
    println!(
        "before compiling {:.0}% of {} examples",
        before.score,
        trainset().len()
    );

    let kept = BootstrapFewShot::new(exact_match)
        .compile(&mut program, &trainset())
        .await?;
    println!("compiled        {kept} of {} solved", trainset().len());
    println!("demos in prompt {}", program.demos.len());
    for demo in &program.demos {
        println!("  {} -> {}", text(demo, "question"), text(demo, "answer"));
    }
    Ok(())
}
