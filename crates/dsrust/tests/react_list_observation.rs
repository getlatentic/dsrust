//! A tool that answers with a list reaches the trajectory as a list, and the prompt as dspy's.
//!
//! `#[tool]` used to type every answer as a `String`, so a tool declared `-> list[str]` — the
//! getting-started page's `wikipedia_search` — could not be written at all. dspy keeps the list
//! itself in `trajectory["observation_0"]` and renders it into the next turn as its enumerated
//! `[1] «…»` form. Both are held here to what dspy recorded for the same run.
//!
//! Recorded by driving `dspy.ReAct("question -> answer", tools=[wikipedia_search])` under a
//! `DummyLM` scripting a search turn, a finish turn, and the extraction; the trajectory is
//! `out.trajectory`, the block is the second turn's rendered `[[ ## trajectory ## ]]`.

use std::sync::Arc;

use anyhow::Result;
use dsrust::lm::DynChatModel;
use dsrust::{DummyLM, Module, ReAct, example, tool};
use serde_json::{Value, json};

/// Search Wikipedia for the given query and return a list of page titles.
#[tool]
fn wikipedia_search(query: String) -> Result<Vec<String>> {
    Ok(vec![
        format!("Page about {query}"),
        "Another page".to_owned(),
    ])
}

fn golden() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/conformance/react/list_observation.json"
    );
    serde_json::from_str(&std::fs::read_to_string(path).expect("committed")).expect("parses")
}

#[tokio::test]
async fn a_list_answer_is_kept_as_a_list_in_the_trajectory() {
    let scripted = Arc::new(DummyLM::new([
        example! { next_thought: "search", next_tool_name: "wikipedia_search", next_tool_args: json!({ "query": "haiku" }) },
        example! { next_thought: "done", next_tool_name: "finish", next_tool_args: json!({}) },
        example! { reasoning: "r", answer: "a" },
    ])) as Arc<dyn DynChatModel>;
    let mut agent = ReAct::parse("question -> answer", vec![Box::new(WikipediaSearch)])
        .expect("parses")
        .max_iters(5);
    for predictor in agent.named_predictors() {
        *predictor.lm = Some(scripted.clone());
    }
    let out = agent
        .forward(example! { question: "q" })
        .await
        .expect("runs");
    let trajectory = out.get("trajectory").expect("the trajectory travels back");
    assert_eq!(
        trajectory["observation_0"],
        golden()["trajectory"]["observation_0"],
        "the observation is not the list dspy keeps"
    );
    assert_eq!(
        trajectory["observation_0"],
        json!(["Page about haiku", "Another page"])
    );
}
