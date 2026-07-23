//! dspy `adapters/types/tool.py`: the calls a model asks for, and what running them returned.
//!
//! A tool *call* is what the model produces — a name and its arguments — and a tool *result* is
//! what came back. They travel together on one field so a conversation can be replayed: the
//! assistant turn states the calls, and the results follow it, either as `tool` messages when the
//! provider called natively or as a rendered field when it did not.

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

/// dspy `ToolCalls.ToolCall`: one call the model asked for.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ToolCall {
    /// The provider's id for this call, present when it came back from a native tool call. dspy
    /// keeps it off both the schema and [`ToolCall::format`] — it is transport, not content — but
    /// it is what pairs a call with its result.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub name: String,
    #[serde(default)]
    pub args: Map<String, Value>,
}

impl ToolCall {
    pub fn new(name: impl Into<String>, args: Map<String, Value>) -> Self {
        Self { id: None, name: name.into(), args }
    }

    /// The same call carrying the provider's id.
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// dspy `ToolCall.format`: the name and arguments alone — the id is transport and never
    /// reaches the model.
    pub fn format(&self) -> Value {
        json!({ "name": self.name, "args": Value::Object(self.args.clone()) })
    }
}

/// dspy `ToolCalls`: the calls a model asked for, and — once they have run — their results.
#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
pub struct ToolCalls {
    pub tool_calls: Vec<ToolCall>,
    /// Present only after the calls have run. dspy keeps this off the schema the model is shown,
    /// because the model produces calls and never their results.
    #[serde(default)]
    pub tool_call_results: Option<ToolCallResults>,
}

impl ToolCalls {
    pub fn new(tool_calls: Vec<ToolCall>) -> Self {
        Self { tool_calls, tool_call_results: None }
    }

    /// The same calls carrying what running them returned.
    pub fn with_results(mut self, results: ToolCallResults) -> Self {
        self.tool_call_results = Some(results);
        self
    }

    /// dspy `ToolCalls.description`: what the field's line tells the model this type must be.
    pub fn description() -> &'static str {
        "Tool calls must be a JSON object with `tool_calls`, a list of calls. \
         Each call must include `name` and `args`. \
         Example: {\"tool_calls\": [{\"name\": \"search\", \"args\": {\"query\": \"cats\"}}]}"
    }

    /// dspy `ToolCalls.format`: the calls alone, results excluded.
    pub fn format(&self) -> Value {
        json!({ "tool_calls": self.tool_calls.iter().map(ToolCall::format).collect::<Vec<_>>() })
    }

    /// The calls with their results dropped, which is how dspy renders the assistant turn of a
    /// replayed conversation — the turn states what was asked for, and the results follow it.
    pub fn without_results(&self) -> Self {
        Self { tool_calls: self.tool_calls.clone(), tool_call_results: None }
    }

    /// Whether every call carries an id and the results answer exactly those ids, in order. dspy
    /// drops the results when they do not, because a provider replaying them needs each `tool`
    /// message to name the call it answers.
    pub fn results_match_calls(&self) -> bool {
        let Some(results) = &self.tool_call_results else {
            return false;
        };
        let ids: Vec<&Option<String>> = self.tool_calls.iter().map(|call| &call.id).collect();
        let answered: Vec<&Option<String>> =
            results.tool_call_results.iter().map(|result| &result.call_id).collect();
        ids == answered && ids.iter().all(|id| id.as_ref().is_some_and(|id| !id.is_empty()))
    }
}

impl Serialize for ToolCalls {
    /// dspy `serialize_model`: the formatted calls, with the results appended only where they
    /// exist — so a value that has not run yet reads exactly as the model produced it.
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut data = self.format();
        if let Some(results) = &self.tool_call_results
            && let Some(object) = data.as_object_mut()
        {
            object.insert(
                "tool_call_results".to_owned(),
                serde_json::to_value(results).map_err(serde::ser::Error::custom)?,
            );
        }
        data.serialize(serializer)
    }
}

/// dspy `ToolCallResults.ToolCallResult`: what one call returned.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ToolCallResult {
    /// The id of the call this answers, so a provider can pair them up.
    #[serde(default)]
    pub call_id: Option<String>,
    pub name: String,
    pub value: Value,
    #[serde(default)]
    pub is_error: bool,
}

/// dspy `ToolCallResults`: what a round of calls returned, in the order they were asked for.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ToolCallResults {
    pub tool_call_results: Vec<ToolCallResult>,
}

impl ToolCallResults {
    /// dspy `from_tool_calls_and_values`: pair each call with what it returned. The lengths must
    /// agree — a result that cannot be matched to a call would be reported against the wrong one.
    pub fn from_tool_calls_and_values(
        tool_calls: &[ToolCall],
        values: Vec<Value>,
        is_errors: Option<Vec<bool>>,
    ) -> Result<Self> {
        if tool_calls.len() != values.len() {
            return Err(anyhow!("`tool_calls` and `values` must have the same length."));
        }
        let is_errors = match is_errors {
            None => vec![false; tool_calls.len()],
            Some(flags) if flags.len() == tool_calls.len() => flags,
            Some(_) => {
                return Err(anyhow!(
                    "`is_errors` must have the same length as `tool_calls` when provided."
                ));
            }
        };
        Ok(Self {
            tool_call_results: tool_calls
                .iter()
                .zip(values)
                .zip(is_errors)
                .map(|((call, value), is_error)| ToolCallResult {
                    call_id: call.id.clone(),
                    name: call.name.clone(),
                    value,
                    is_error,
                })
                .collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn search_call() -> ToolCall {
        ToolCall::new("search", json!({ "query": "cats" }).as_object().expect("object").clone())
            .with_id("call_1")
    }

    /// dspy `ToolCall.format` states the name and args; the provider's id is transport and is
    /// deliberately absent, as is the schema entry for it.
    #[test]
    fn a_call_states_its_name_and_arguments_but_not_its_id() {
        assert_eq!(
            search_call().format(),
            json!({ "name": "search", "args": { "query": "cats" } })
        );
    }

    /// The description upstream puts on the field's line, verbatim.
    #[test]
    fn the_type_describes_itself_the_way_dspy_does() {
        assert_eq!(
            ToolCalls::description(),
            "Tool calls must be a JSON object with `tool_calls`, a list of calls. Each call must \
             include `name` and `args`. Example: {\"tool_calls\": [{\"name\": \"search\", \"args\": \
             {\"query\": \"cats\"}}]}"
        );
    }

    /// dspy's serializer writes the formatted calls, and appends results only once they exist.
    #[test]
    fn it_writes_the_calls_alone_until_they_have_run() {
        let calls = ToolCalls::new(vec![search_call()]);
        assert_eq!(
            serde_json::to_value(&calls).expect("serializes"),
            json!({ "tool_calls": [{ "name": "search", "args": { "query": "cats" } }] })
        );

        let results =
            ToolCallResults::from_tool_calls_and_values(&calls.tool_calls, vec![json!("cat")], None)
                .expect("pairs up");
        // The results dump as their own model, so they nest under a second `tool_call_results`.
        // Checked against dspy 3.3.0b1's `ToolCalls.model_dump()` rather than assumed.
        assert_eq!(
            serde_json::to_value(calls.with_results(results)).expect("serializes"),
            json!({
                "tool_calls": [{ "name": "search", "args": { "query": "cats" } }],
                "tool_call_results": {
                    "tool_call_results": [
                        { "call_id": "call_1", "name": "search", "value": "cat", "is_error": false }
                    ],
                },
            })
        );
    }

    #[test]
    fn results_are_paired_with_the_calls_that_produced_them() {
        let calls = vec![search_call()];
        let results =
            ToolCallResults::from_tool_calls_and_values(&calls, vec![json!("cat")], None).expect("pairs");
        assert_eq!(
            results.tool_call_results[0],
            ToolCallResult {
                call_id: Some("call_1".to_owned()),
                name: "search".to_owned(),
                value: json!("cat"),
                is_error: false,
            }
        );
        // A mismatched count would report a result against the wrong call, so it is refused.
        assert!(ToolCallResults::from_tool_calls_and_values(&calls, vec![], None).is_err());
        assert!(
            ToolCallResults::from_tool_calls_and_values(&calls, vec![json!("cat")], Some(vec![]))
                .is_err()
        );
    }

    /// dspy drops the results when they do not answer exactly the calls that were made, because a
    /// replayed `tool` message has to name the call it belongs to.
    #[test]
    fn results_only_replay_when_they_answer_the_calls_made() {
        let calls = vec![search_call()];
        let matching =
            ToolCallResults::from_tool_calls_and_values(&calls, vec![json!("cat")], None).expect("pairs");
        assert!(ToolCalls::new(calls.clone()).with_results(matching).results_match_calls());

        // No results at all: nothing to replay.
        assert!(!ToolCalls::new(calls.clone()).results_match_calls());
        // A result answering a different call.
        let mismatched = ToolCallResults {
            tool_call_results: vec![ToolCallResult {
                call_id: Some("other".to_owned()),
                name: "search".to_owned(),
                value: json!("cat"),
                is_error: false,
            }],
        };
        assert!(!ToolCalls::new(calls).with_results(mismatched).results_match_calls());
        // A call with no id cannot be answered by name alone.
        let anonymous = vec![ToolCall::new("search", Map::new())];
        let results =
            ToolCallResults::from_tool_calls_and_values(&anonymous, vec![json!("cat")], None).expect("pairs");
        assert!(!ToolCalls::new(anonymous).with_results(results).results_match_calls());
    }
}
