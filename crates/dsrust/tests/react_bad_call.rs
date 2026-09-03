//! A tool called badly — a missing argument, a mistyped one, one the tool does not declare — lands
//! in the trajectory as the observation dspy records: `Execution error in {tool}: …` ending in the
//! exception's own line. The traceback dspy prints between those is CPython's frames; the line it
//! ends with is what the model reads and what this reproduces. Recorded by
//! `scripts/generate_bad_call_fixture.py`.

use std::sync::Arc;

use dsrust::lm::DynChatModel;
use dsrust::{DummyLM, Module, ReAct, Tool, example, tool};
use serde_json::Value;

#[tool]
/// Search Wikipedia.
fn wikipedia_search(query: String) -> anyhow::Result<Vec<String>> {
    Ok(vec![format!("Page about {query}")])
}

#[tool]
/// Count.
fn count_to(limit: u32) -> anyhow::Result<String> {
    Ok((0..limit)
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join(","))
}

#[tool]
/// Pair.
fn pair(prompt: String, answer: String) -> anyhow::Result<String> {
    Ok(format!("{prompt}|{answer}"))
}

#[tool]
/// Triple.
fn triple(a: String, b: String, c: String) -> anyhow::Result<String> {
    Ok(format!("{a}{b}{c}"))
}

#[tool]
/// Optional by type, not by default.
fn with_optional(prompt: String, worked: Option<String>) -> anyhow::Result<String> {
    Ok(format!("{prompt}|{}", worked.as_deref().unwrap_or("None")))
}

fn golden() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/conformance/react/bad_call_observation.json"
    );
    serde_json::from_str(&std::fs::read_to_string(path).expect("committed")).expect("parses")
}

fn tools() -> Vec<Box<dyn Tool>> {
    vec![
        Box::new(WikipediaSearch),
        Box::new(CountTo),
        Box::new(Pair),
        Box::new(Triple),
        Box::new(WithOptional),
    ]
}

async fn observation_for(tool: &str, args: &Value) -> Value {
    let scripted = Arc::new(DummyLM::new([
        example! { next_thought: "call it", next_tool_name: tool.to_owned(), next_tool_args: args.clone() },
        example! { next_thought: "done", next_tool_name: "finish", next_tool_args: Value::Object(Default::default()) },
        example! { reasoning: "r", answer: "a" },
    ])) as Arc<dyn DynChatModel>;
    let mut agent = ReAct::parse("question -> answer", tools())
        .expect("parses")
        .max_iters(2);
    for predictor in agent.named_predictors() {
        *predictor.lm = Some(scripted.clone());
    }
    let out = agent
        .forward(example! { question: "q" })
        .await
        .expect("the agent finishes");
    out.get("trajectory").expect("the trajectory travels back")["observation_0"].clone()
}

#[tokio::test]
async fn a_bad_call_is_observed_as_dspy_observes_it() {
    let golden = golden();
    let cases = golden["cases"].as_object().expect("cases");
    assert!(cases.len() >= 11, "the fixture holds every case recorded");
    let mut wrong = Vec::new();
    for (label, case) in cases {
        let tool = case["tool"].as_str().expect("tool");
        let observed = observation_for(tool, &case["args"]).await;
        let expected = match (case["exception_line"].as_str(), case["parser"].as_bool()) {
            (Some(_), Some(true)) => {
                let prefix = format!("Execution error in {tool}: ");
                match observed
                    .as_str()
                    .is_some_and(|text| text.starts_with(&prefix))
                {
                    true => continue,
                    false => Value::String(format!("{prefix}<the parser's own reason>")),
                }
            }
            (Some(line), _) => Value::String(format!("Execution error in {tool}: {line}")),
            (None, _) => case["observation_0"].clone(),
        };
        if observed != expected {
            wrong.push(format!("{label}: expected {expected}, observed {observed}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
