//! What this crate returns for a failing provider call, as JSON, so it can be compared with what
//! dspy raises for the same call. Driven by `scripts/compare_error_shapes.py`.

use dsrust::lm::{LM, LmFailure, configure};
use dsrust::{Module, Predict, Signature};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::var("PROBE_MODEL").unwrap_or_else(|_| "gemma".to_owned());
    let base =
        std::env::var("PROBE_BASE").unwrap_or_else(|_| "http://127.0.0.1:8080/v1".to_owned());
    configure(
        LM::new(format!("openai/{model}"))?
            .openai_base_url(&base)
            .openai_api_key("x")
            .cache(false),
    );

    let asked = Predict::from_signature("question -> answer".parse::<Signature>()?)
        .forward(dsrust::example! { question: "hi" })
        .await;
    match asked {
        Ok(_) => println!("NO ERROR"),
        Err(error) => {
            let described = match error.downcast_ref::<LmFailure>() {
                Some(failure) => json!({
                    "class": "LmFailure",
                    "code": failure.kind.code(),
                    "status": failure.status,
                    "retryable": failure.is_retryable(),
                    "message": failure.message.chars().take(120).collect::<String>(),
                }),
                None => json!({
                    "class": "untyped",
                    "message": format!("{error:#}").chars().take(120).collect::<String>(),
                }),
            };
            println!("{described}");
        }
    }
    Ok(())
}
