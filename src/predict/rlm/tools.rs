//! dspy `RLM._make_llm_tools`: the sub-LLM the model's own code can reach, and the budget it
//! spends from.

use std::sync::{Arc, Mutex};

use anyhow::{Result, bail};
use serde_json::{Value, json};

use crate::adapter::python_json::json_dumps;
use crate::react::Tool;

/// The shared budget the two sub-LLM tools spend from.
struct CallBudget {
    spent: Mutex<usize>,
    max: usize,
}

impl CallBudget {
    /// dspy `_check_and_increment`: the whole batch is charged before any of it runs, so a batch
    /// that would overrun is refused rather than half-answered.
    fn charge(&self, calls: usize) -> Result<()> {
        let mut spent = self.spent.lock().expect("the budget lock");
        if *spent + calls > self.max {
            bail!(
                "LLM call limit exceeded: {spent} + {calls} > {}. Use Python code for aggregation \
                 instead of making more LLM calls.",
                self.max
            );
        }
        *spent += calls;
        Ok(())
    }
}

/// dspy's `llm_query`: one prompt to a sub-LLM, charged against the budget.
struct LlmQuery<A> {
    budget: Arc<CallBudget>,
    ask: Arc<A>,
    args: Value,
}

impl<A: Fn(&str) -> Result<String> + Send + Sync> Tool for LlmQuery<A> {
    fn name(&self) -> &str {
        "llm_query"
    }

    fn description(&self) -> &str {
        "Query the LLM with a prompt string."
    }

    fn args(&self) -> &Value {
        &self.args
    }

    fn call(&self, args: &Value) -> Result<String> {
        let prompt = args
            .get("prompt")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if prompt.is_empty() {
            bail!("prompt cannot be empty");
        }
        self.budget.charge(1)?;
        (self.ask)(prompt)
    }
}

/// dspy's `llm_query_batched`: several prompts at once, answered in the order they were given.
struct LlmQueryBatched<A> {
    budget: Arc<CallBudget>,
    ask: Arc<A>,
    args: Value,
}

impl<A: Fn(&str) -> Result<String> + Send + Sync> Tool for LlmQueryBatched<A> {
    fn name(&self) -> &str {
        "llm_query_batched"
    }

    fn description(&self) -> &str {
        "Query the LLM with multiple prompts concurrently."
    }

    fn args(&self) -> &Value {
        &self.args
    }

    fn call(&self, args: &Value) -> Result<String> {
        Ok(json_dumps(&self.call_value(args)?))
    }

    /// The answers as a list, which is what the code that called this reads.
    ///
    /// dspy runs the prompts on a thread pool; that is a speed property rather than an observable
    /// one, since it reassembles the answers in the order the prompts were given either way. The
    /// ask here is synchronous — [`Tool::call`] is — so they run in that order to begin with.
    fn call_value(&self, args: &Value) -> Result<Value> {
        let Some(prompts) = args.get("prompts").and_then(Value::as_array) else {
            bail!("prompts must be a list");
        };
        // An empty batch is answered with an empty list and costs nothing.
        if prompts.is_empty() {
            return Ok(json!([]));
        }
        self.budget.charge(prompts.len())?;
        let answers: Vec<Value> = prompts
            .iter()
            .map(|prompt| {
                let prompt = prompt.as_str().unwrap_or_default();
                // dspy answers a prompt that failed with the error in place, so one bad prompt
                // does not lose the rest of the batch.
                match (self.ask)(prompt) {
                    Ok(answer) => json!(answer),
                    Err(error) => json!(format!("[ERROR] {error}")),
                }
            })
            .collect();
        Ok(Value::Array(answers))
    }
}

/// The `llm_query` pair the REPL code can call, sharing one budget.
///
/// `ask` is the caller's bridge to a sub-LLM, synchronous because [`Tool::call`] is — the same
/// contract [`mcp_tool`](crate::react::mcp_tool) states, and a caller driving an async model blocks
/// on it. Hand the pair to [`Rlm::with_tools`](super::Rlm::with_tools) and they reach the sandbox
/// through [`define_tools`](crate::interpreter::CodeInterpreter::define_tools) with the caller's
/// own tools.
pub fn llm_query_tools<A>(max_llm_calls: usize, ask: A) -> Vec<Arc<dyn Tool>>
where
    A: Fn(&str) -> Result<String> + Send + Sync + 'static,
{
    let budget = Arc::new(CallBudget {
        spent: Mutex::new(0),
        max: max_llm_calls,
    });
    let ask = Arc::new(ask);
    vec![
        Arc::new(LlmQuery {
            budget: budget.clone(),
            ask: ask.clone(),
            args: json!({ "prompt": { "type": "string" } }),
        }),
        Arc::new(LlmQueryBatched {
            budget,
            ask,
            args: json!({ "prompts": { "type": "array", "items": { "type": "string" } } }),
        }),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tools(max: usize) -> Vec<Arc<dyn Tool>> {
        llm_query_tools(max, |prompt| match prompt {
            "boom" => bail!("the sub-LLM failed"),
            other => Ok(format!("answered: {other}")),
        })
    }

    #[test]
    fn one_query_spends_one_call_and_an_empty_prompt_is_refused() {
        let tools = tools(2);
        let query = &tools[0];
        assert_eq!(query.name(), "llm_query");
        assert_eq!(
            query.call(&json!({ "prompt": "hi" })).expect("answers"),
            "answered: hi"
        );
        assert!(
            query.call(&json!({ "prompt": "" })).is_err(),
            "an empty prompt is refused"
        );
    }

    /// The two tools share one budget, and overrunning it says what to do instead.
    #[test]
    fn the_budget_is_shared_and_refuses_an_overrun() {
        let tools = tools(2);
        let (query, batched) = (&tools[0], &tools[1]);
        query.call(&json!({ "prompt": "one" })).expect("answers");
        // One spent, so a batch of two would overrun and is refused whole.
        let error = batched
            .call_value(&json!({ "prompts": ["a", "b"] }))
            .expect_err("refuses");
        assert!(
            error
                .to_string()
                .starts_with("LLM call limit exceeded: 1 + 2 > 2."),
            "got: {error}"
        );
        assert!(
            error
                .to_string()
                .contains("Use Python code for aggregation"),
            "got: {error}"
        );
        // The refused batch was not charged, so one call remains.
        query
            .call(&json!({ "prompt": "two" }))
            .expect("the last call");
        assert!(
            query.call(&json!({ "prompt": "three" })).is_err(),
            "the budget is spent"
        );
    }

    /// A batch answers in the order it was given, and one failed prompt does not lose the rest.
    #[test]
    fn a_batch_keeps_its_order_and_reports_a_failure_in_place() {
        let tools = tools(10);
        let batched = &tools[1];
        let answers = batched
            .call_value(&json!({ "prompts": ["a", "boom", "c"] }))
            .expect("answers");
        assert_eq!(answers[0], json!("answered: a"));
        assert_eq!(answers[2], json!("answered: c"));
        assert!(
            answers[1]
                .as_str()
                .expect("an error")
                .starts_with("[ERROR] "),
            "got: {}",
            answers[1]
        );
        // An empty batch costs nothing and answers with nothing.
        assert_eq!(
            batched
                .call_value(&json!({ "prompts": [] }))
                .expect("answers"),
            json!([])
        );
    }
}
