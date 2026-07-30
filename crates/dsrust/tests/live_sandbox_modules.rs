//! The code-writing modules end to end: a real model writes the code, the real sandbox runs it.
//!
//! `live_code_modules.rs` scripts the interpreter so the *loop* can be checked without deno; this
//! is the other half, and the one that says the pieces fit. Nothing is canned — the model chooses
//! what to compute, Pyodide computes it, and the answer comes back through `SUBMIT`.
//!
//! ```sh
//! cargo test --test live_sandbox_modules -- --ignored --nocapture --test-threads=1
//! ```

use std::sync::Arc;

use dsrust::interpreter::DenoInterpreter;
use dsrust::lm::{LM, configure};
use dsrust::{CodeAct, FnTool, Module, ProgramOfThought, Signature, Tool, example};
use serde_json::json;

/// The live model, from the environment or a local `llama-server`.
fn configure_live() -> String {
    let base_url =
        std::env::var("BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:8080/v1".to_owned());
    let model =
        std::env::var("MODEL").unwrap_or_else(|_| "ggml-org/gemma-3-1b-it-GGUF:Q4_K_M".to_owned());
    configure(
        LM::new(&format!("openai/{model}"))
            .expect("a model id")
            .with_openai_base_url(&base_url)
            .with_openai_key("not-needed-locally"),
    );
    assert!(
        DenoInterpreter::available(),
        "this test needs deno on the path"
    );
    model
}

/// The default constructor reaches a working sandbox, which is the whole point of defaulting it.
///
/// A model this small often writes code that fails, and that is fine here: what is under test is
/// that the loop reaches the sandbox, runs Python, and feeds the result back. An episode that
/// errors inside Pyodide still proves every link — the failure text came from real CPython.
#[tokio::test]
#[ignore = "needs deno and a live model"]
async fn program_of_thought_runs_its_code_in_the_real_sandbox() {
    let model = configure_live();
    let pot = ProgramOfThought::new("question -> answer".parse::<Signature>().expect("parses"));

    let prediction = pot
        .forward(example! { question: "What is 5 factorial? Compute it in Python." })
        .await
        .expect("the loop completes");

    println!("[{model}] ProgramOfThought answered: {prediction:?}");
    assert!(
        prediction.get("answer").is_some(),
        "an answer field came back"
    );
}

/// CodeAct's tools are host callbacks now, so the sandboxed code calls back into Rust mid-run.
///
/// This one needs a model that can hold the output format for a multi-field turn. A small model
/// tends to answer `finished` with a fenced ```` ```python\nTrue\n``` ````, which fails to coerce —
/// and that refusal is upstream's: `parse_value("```python\nTrue\n```", bool)` raises a pydantic
/// `ValidationError` in dspy too, checked against the pinned version rather than assumed.
#[tokio::test]
#[ignore = "needs deno and a live model that holds the output format"]
async fn code_act_calls_a_rust_tool_from_inside_the_sandbox() {
    let model = configure_live();
    let tools: Vec<Arc<dyn Tool>> = vec![Arc::new(FnTool::new(
        "shipping_cost",
        "The shipping cost in pounds for a given weight in kilograms.",
        json!({ "kilograms": { "type": "number" } }),
        |args| {
            let kilograms = args["kilograms"].as_f64().unwrap_or_default();
            Ok(format!("{:.2}", 4.0 + kilograms * 1.5))
        },
    ))];
    let act = CodeAct::new(
        "question -> answer".parse::<Signature>().expect("parses"),
        tools,
    );

    let prediction = act
        .forward(example! { question: "What does it cost to ship 3 kilograms? Use shipping_cost." })
        .await
        .expect("the loop completes");

    println!("[{model}] CodeAct answered: {prediction:?}");
    assert!(
        prediction.get("answer").is_some(),
        "an answer field came back"
    );
}
