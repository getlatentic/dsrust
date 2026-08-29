//! The quickstart loop against a real model, on Amazon Bedrock's OpenAI-compatible endpoint.
//!
//!     export BEDROCK_BASE_URL=https://<your-bedrock-host>/v1
//!     export AWS_BEARER_TOKEN_BEDROCK=ABSK...
//!     cargo run --example bedrock
//!
//! Bedrock serves `/v1/chat/completions` in OpenAI's own shape, so nothing here is Bedrock-specific
//! beyond two strings: the base URL and the model id. The same file points at Groq, Together, vLLM
//! or LM Studio by changing `BEDROCK_BASE_URL` and `BEDROCK_MODEL`.
//!
//! The host is read from the environment rather than written here, because which Bedrock endpoint
//! an account reaches is not this repository's fact to publish.
//!
//! The model id carries its own `openai.` prefix — `openai/openai.gpt-oss-120b` — because the first
//! half is how *this* crate routes to the OpenAI wire and the second is what Bedrock calls the
//! model. `GET /v1/models` lists them.
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
    let key = std::env::var("AWS_BEARER_TOKEN_BEDROCK").map_err(|_| {
        anyhow::anyhow!(
            "AWS_BEARER_TOKEN_BEDROCK is not set. It is a Bedrock API key, the `ABSK...` kind, and \
             it goes out as a bearer token rather than being signed with SigV4."
        )
    })?;
    let base_url = std::env::var("BEDROCK_BASE_URL").map_err(|_| {
        anyhow::anyhow!(
            "BEDROCK_BASE_URL is not set — the OpenAI-compatible base for your account, ending \
             `/v1`. This crate appends `/chat/completions` to it."
        )
    })?;
    let model = std::env::var("BEDROCK_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_owned());

    let lm = LM::builder(&model)
        .api_base(base_url)
        .api_key(key)
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
