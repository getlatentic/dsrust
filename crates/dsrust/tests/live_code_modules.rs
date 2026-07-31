//! The code-writing modules against a real model — ignored by default, like the other live tests.
//!
//! Everything else about these modules is checked against dspy: the prompts byte for byte, and the
//! loops against a scripted model that always answers in perfect format. That leaves one thing
//! unchecked, and it is the thing that decides whether they work at all — **does a real model,
//! reading the prompt we send, answer in a shape the adapter can parse and the loop can act on?**
//! A scripted model can never fail that, because the script is what a passing reply looks like.
//!
//! The interpreter here is scripted while the *model* is real, which is deliberate on two counts.
//! It isolates the half under test — the model's reply, not the sandbox. And it means no
//! model-written code is executed: these tests read what the model wrote and hand back a canned
//! result, so running them cannot run anything the model invented.
//!
//! Run them **serialized**: one local daemon cannot serve three multi-turn conversations at once,
//! and concurrent runs time out against a model that answers any one of them fine.
//!
//! ```text
//! cargo test --test live_code_modules -- --ignored --nocapture --test-threads=1
//! LIVE_LM=ollama_chat/qwen2.5:7b-instruct LIVE_LM_TIMEOUT=180 \
//!   cargo test --test live_code_modules -- --ignored --nocapture --test-threads=1
//! ```
//!
//! What three local models did with them, since the failures are the provider's rather than the
//! module's and are worth recognising:
//!
//! - `gemma3:4b` — all three pass, which is why it is the default here.
//! - `qwen2.5:7b-instruct` — RLM sends a far larger prompt than the other two (its action template,
//!   the variables and the whole session), and this size of model spent over twenty seconds on it.
//!   That measurement is why the crate's default bound is no longer twenty seconds: it was tighter
//!   than litellm's, so a call dspy completes was one this crate abandoned.
//! - `gpt-oss:20b` — CodeAct passes. ProgramOfThought wants a raised bound as above. RLM gets an
//!   ollama 500 that no timeout helps: its harmony parser reads the model's Python as a malformed
//!   tool call (`error parsing tool call: raw='print(len(context))'`). A model whose output the
//!   provider parses for tool calls cannot be used to *write* code through it. That message is only
//!   legible because the ollama client surfaces the body.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use dsrust::interpreter::{CodeInterpreter, Executed};
use dsrust::{CodeAct, Example, LM, Module, ProgramOfThought, Rlm, Signature, example};
use serde_json::{Value, json};

/// The model the env asks for, or a local ollama.
fn live_model() -> String {
    std::env::var("LIVE_LM").unwrap_or_else(|_| "ollama_chat/gemma3:4b".to_owned())
}

/// How long one call may take, in seconds. The crate's default is deliberately tight for a hosted
/// model; a larger local one reading an RLM session wants more, and `LIVE_LM_TIMEOUT` is how a run
/// against one says so.
fn live_timeout() -> Option<Duration> {
    let seconds: u64 = std::env::var("LIVE_LM_TIMEOUT").ok()?.parse().ok()?;
    Some(Duration::from_secs(seconds))
}

fn configure_live() -> String {
    let model = live_model();
    let mut lm = LM::new(&model)
        .expect("a valid model ref")
        .with_cache(false);
    if let Some(timeout) = live_timeout() {
        lm = lm.with_timeout(timeout);
    }
    dsrust::configure(lm);
    model
}

/// An interpreter that records what the model wrote and answers with a canned result.
///
/// It never executes anything. `answer` is what a run of that code would have produced, so the
/// module can carry on and the model can be asked to read the result — which is the part under
/// test.
struct Canned {
    answer: Executed,
    ran: Mutex<Vec<String>>,
}

impl Canned {
    fn new(answer: Executed) -> Arc<Self> {
        Arc::new(Self {
            answer,
            ran: Mutex::new(Vec::new()),
        })
    }

    fn wrote(&self) -> Vec<String> {
        self.ran.lock().expect("ran").clone()
    }
}

impl CodeInterpreter for Canned {
    fn execute(
        &self,
        code: &str,
        _variables: &serde_json::Map<String, Value>,
    ) -> anyhow::Result<Executed> {
        self.ran.lock().expect("ran").push(code.to_owned());
        Ok(self.answer.clone())
    }
}

/// Whatever a field came back as, for a human reading `--nocapture`.
fn field(prediction: &dsrust::Prediction, name: &str) -> String {
    prediction
        .get(name)
        .map(|value| match value {
            Value::String(text) => text.clone(),
            other => other.to_string(),
        })
        .unwrap_or_default()
}

/// ProgramOfThought: the model writes code, is shown what it produced, and states the answer.
#[tokio::test]
#[ignore = "needs a live model; set LIVE_LM or run a local ollama"]
async fn program_of_thought_runs_against_a_real_model() {
    let model = configure_live();
    let interpreter = Canned::new(Executed::Submitted(json!({ "answer": "120" })));
    let pot = ProgramOfThought::interpreter(
        "question -> answer".parse::<Signature>().expect("parses"),
        interpreter.clone(),
    );

    let prediction = pot
        .forward(example! { question: "What is 5 factorial? Compute it in code." })
        .await
        .expect("the loop completes against a real model");

    let wrote = interpreter.wrote();
    println!(
        "[{model}] ProgramOfThought wrote {} snippet(s)",
        wrote.len()
    );
    for code in &wrote {
        println!("--- code ---\n{code}");
    }
    println!("--- answer ---\n{}", field(&prediction, "answer"));

    // The model's code parsed well enough to reach the interpreter, and the final ask answered.
    assert!(
        !wrote.is_empty(),
        "the model's code never reached the interpreter"
    );
    assert!(
        !field(&prediction, "answer").is_empty(),
        "no answer came back"
    );
}

/// CodeAct: the model writes a snippet per turn and marks itself finished.
#[tokio::test]
#[ignore = "needs a live model; set LIVE_LM or run a local ollama"]
async fn code_act_runs_against_a_real_model() {
    let model = configure_live();
    let interpreter = Canned::new(Executed::Printed(json!("120")));
    let act = CodeAct::interpreter(
        "question -> answer".parse::<Signature>().expect("parses"),
        Vec::new(),
        interpreter.clone(),
    )
    .max_iters(3);

    let prediction = act
        .forward(example! { question: "What is 5 factorial? Print it." })
        .await
        .expect("the loop completes against a real model");

    let wrote = interpreter.wrote();
    println!("[{model}] CodeAct wrote {} snippet(s)", wrote.len());
    for code in &wrote {
        println!("--- code ---\n{code}");
    }
    println!("--- answer ---\n{}", field(&prediction, "answer"));
    println!("--- trajectory ---\n{}", field(&prediction, "trajectory"));

    assert!(
        !wrote.is_empty(),
        "the model's code never reached the interpreter"
    );
    assert!(
        !field(&prediction, "answer").is_empty(),
        "no answer came back"
    );
}

/// RLM: the model drives a REPL over a long input, then submits or is extracted from.
///
/// Which of the two ends the run is the model's business, and both are valid — what is asserted is
/// that the loop reached one of them rather than failing to parse a reply along the way.
#[tokio::test]
#[ignore = "needs a live model; set LIVE_LM or run a local ollama"]
async fn rlm_drives_a_repl_against_a_real_model() {
    let model = configure_live();
    let interpreter = Canned::new(Executed::Printed(json!(
        "the document mentions Paris 3 times"
    )));
    let rlm = Rlm::interpreter(
        "context -> answer".parse::<Signature>().expect("parses"),
        interpreter.clone(),
    )
    .max_iterations(3);

    let context = "Paris is the capital of France. ".repeat(50);
    let prediction = rlm
        .forward(Example::new([("context", json!(context))]))
        .await
        .expect("the loop completes against a real model");

    let wrote = interpreter.wrote();
    println!("[{model}] RLM wrote {} snippet(s)", wrote.len());
    for code in &wrote {
        println!("--- code ---\n{code}");
    }
    println!(
        "--- final_reasoning ---\n{}",
        field(&prediction, "final_reasoning")
    );
    println!("--- answer ---\n{}", field(&prediction, "answer"));

    assert!(
        !wrote.is_empty(),
        "the model's code never reached the interpreter"
    );
    assert!(
        !field(&prediction, "answer").is_empty(),
        "no answer came back"
    );
}
