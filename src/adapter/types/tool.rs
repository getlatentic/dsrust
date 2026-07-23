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
    /// The annotation dspy prints for this type, and so the name a field carrying it is known by.
    /// Upstream compares the annotation itself; across the bridge the printed name is that
    /// identity, which is how [`history`](super::super::history) recognises a `History` field too.
    pub const ANNOTATION: &'static str = "ToolCalls";

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

    /// dspy `ToolCalls.from_dict_list`: read back a list of calls a provider (or a caller) wrote.
    /// Each entry may be shaped as the provider sends one — a `function` holding the name and the
    /// arguments as text — or as this type writes one, with `args` already structured.
    pub fn from_dict_list(values: &[Value]) -> Result<Self> {
        Ok(Self::new(values.iter().map(normalized_call).collect::<Result<_>>()?))
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

/// dspy `_normalize_tool_call_dict`: one written call read back into a [`ToolCall`].
///
/// A provider states the call under `function`, with its arguments as JSON *text*; this type
/// states them under `args`, already structured. Both are accepted, and the id may arrive under
/// either `id` or `call_id`.
fn normalized_call(data: &Value) -> Result<ToolCall> {
    let Some(fields) = data.as_object() else {
        return Err(anyhow!("Received invalid tool call value for `ToolCalls`: {data}"));
    };
    let (arguments, name) = match fields.get("function") {
        // dspy's `data.get("function") or {}`: a null function states nothing rather than
        // failing, but one that is present and not a mapping is an error.
        Some(function) => {
            let function = match function {
                Value::Object(function) => Some(function),
                Value::Null => None,
                other => {
                    return Err(anyhow!("Received invalid function value for `ToolCalls`: {other}"));
                }
            };
            let name = written(function.and_then(|f| f.get("name")))
                .or_else(|| written(fields.get("name")));
            (function.and_then(|f| f.get("arguments")), name)
        }
        None => (
            fields.get("args").or_else(|| fields.get("arguments")),
            written(fields.get("name")),
        ),
    };
    Ok(ToolCall {
        id: written(fields.get("id")).or_else(|| written(fields.get("call_id"))),
        name: name.unwrap_or_default(),
        args: arguments.map(structured_args).unwrap_or_default(),
    })
}

/// dspy hands a provider's arguments through json-repair before reading them, because a model
/// writing JSON by hand produces text a strict reader rejects — a trailing comma, a single quote.
/// Anything that is neither text nor a mapping states no arguments at all.
fn structured_args(arguments: &Value) -> Map<String, Value> {
    match arguments {
        Value::Object(args) => args.clone(),
        Value::String(text) => serde_json::from_str::<Value>(text)
            .ok()
            .or_else(|| crate::adapter::parse::repair::python_literal(text))
            .and_then(|value| value.as_object().cloned())
            .unwrap_or_default(),
        _ => Map::new(),
    }
}

/// What a field actually says, for the ones dspy reads with `or` — `data.get("id") or
/// data.get("call_id")`, `function.get("name") or data.get("name")`.
///
/// Python's `or` falls through everything falsy, so a key that is absent, null, or an empty string
/// all reach the next spelling; `Option::or_else` alone only falls through the absent one, which
/// would keep a provider's `"id": null` and lose the `call_id` beside it.
fn written(value: Option<&Value>) -> Option<String> {
    value.and_then(Value::as_str).filter(|text| !text.is_empty()).map(str::to_owned)
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

    /// A provider states the call under `function` with its arguments as text, and may name the id
    /// `call_id`. Both spellings read back the same way, and the id survives — it is what pairs a
    /// result to its call.
    #[test]
    fn a_provider_written_call_reads_back_with_its_id() {
        let calls = ToolCalls::from_dict_list(&[json!({
            "function": { "name": "search", "arguments": "{\"query\": \"cats\"}" },
            "call_id": "call_from_responses",
            "type": "function",
        })])
        .expect("reads back");
        assert_eq!(calls.tool_calls[0].name, "search");
        assert_eq!(calls.tool_calls[0].id.as_deref(), Some("call_from_responses"));
        assert_eq!(calls.tool_calls[0].args, *json!({ "query": "cats" }).as_object().unwrap());
    }

    /// The spelling dspy did not use is often *written* rather than absent — a provider object
    /// read field by field gives up `"id": null` beside the `call_id` that holds the value. Python's
    /// `or` falls through it; an `Option` chain that only falls through an absent key would keep the
    /// null and lose the id, and with it the pairing between a call and its result.
    #[test]
    fn a_spelling_written_as_null_falls_through_to_the_next() {
        let calls = ToolCalls::from_dict_list(&[json!({
            "id": null,
            "call_id": "call_from_responses",
            "function": { "name": null, "arguments": "{\"query\": \"cats\"}" },
            "name": "search",
        })])
        .expect("reads back");
        assert_eq!(calls.tool_calls[0].id.as_deref(), Some("call_from_responses"));
        assert_eq!(calls.tool_calls[0].name, "search");
        // Empty is falsy to Python too, so it falls through the same way.
        let calls = ToolCalls::from_dict_list(&[json!({ "id": "", "call_id": "call_1" })])
            .expect("reads back");
        assert_eq!(calls.tool_calls[0].id.as_deref(), Some("call_1"));
    }

    /// dspy reads a provider's arguments through json-repair, so text a strict reader rejects still
    /// yields the call. The trailing comma is upstream's own case.
    #[test]
    fn malformed_arguments_are_repaired_rather_than_dropped() {
        let calls = ToolCalls::from_dict_list(&[json!({
            "function": { "name": "search", "arguments": "{\"query\": \"cats\",}" },
            "id": "call_1",
        })])
        .expect("reads back");
        assert_eq!(calls.tool_calls[0].args, *json!({ "query": "cats" }).as_object().unwrap());
    }

    /// Arguments that are neither text nor a mapping state nothing, and this type's own spelling
    /// (`args`, already structured) reads back too.
    #[test]
    fn arguments_that_say_nothing_leave_the_call_bare() {
        let calls = ToolCalls::from_dict_list(&[
            json!({ "name": "search", "args": { "query": "cats" } }),
            json!({ "function": { "name": "noop", "arguments": 3 } }),
            json!({ "function": null, "name": "bare" }),
        ])
        .expect("reads back");
        assert_eq!(calls.tool_calls[0].args, *json!({ "query": "cats" }).as_object().unwrap());
        assert!(calls.tool_calls[1].args.is_empty());
        assert_eq!(calls.tool_calls[2].name, "bare");
        // A call that is not a mapping at all is refused rather than guessed at.
        assert!(ToolCalls::from_dict_list(&[json!("search")]).is_err());
        assert!(ToolCalls::from_dict_list(&[json!({ "function": "search" })]).is_err());
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
