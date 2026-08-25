//! What running a model's tool calls returned.
//!
//! Split from the calls themselves because the two halves answer different questions: a
//! `ToolCall` is what the model asked for, and this is what came back — including the shapes
//! upstream's `validate_input` accepts on the way in, which a derive does not.

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::super::refusal;
use super::ToolCall;

/// dspy `ToolCallResults.ToolCallResult`: what one call returned.
///
/// ```
/// use dsrust::ToolCallResult;
///
/// // `call_id` is what pairs a result with the call it answers; a provider that does not issue
/// // ids leaves it empty, and the pairing falls back to order.
/// let returned = ToolCallResult {
///     call_id: Some("call_1".to_owned()),
///     name: "search".to_owned(),
///     value: serde_json::json!({ "hits": 3 }),
///     is_error: false,
/// };
/// assert!(!returned.is_error, "a failure is a result too, flagged rather than raised");
/// ```
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
#[derive(Debug, Clone, PartialEq, Default, Serialize)]
pub struct ToolCallResults {
    pub tool_call_results: Vec<ToolCallResult>,
}

impl<'de> Deserialize<'de> for ToolCallResults {
    /// dspy `ToolCallResults.validate_input`: a bare list is the results, a map carrying
    /// `tool_call_results` is the whole value, and a lone `{name, value}` is one result.
    ///
    /// A derive read only the third of those. A provider returning a bare list — which is the
    /// shape `ToolCalls` accepts on the way out, so the natural one to return — came back as
    /// `invalid type: map, expected a sequence`.
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error as _;
        let data = Value::deserialize(deserializer)?;
        let results = match &data {
            Value::Array(_) => Some(data.clone()),
            Value::Object(fields) => match fields.get("tool_call_results") {
                Some(results) => Some(results.clone()),
                // Upstream tests for both keys, so a map holding one of them is not a lone result.
                None if fields.contains_key("name") && fields.contains_key("value") => {
                    Some(Value::Array(vec![data.clone()]))
                }
                None => None,
            },
            _ => None,
        };
        let Some(results) = results else {
            return Err(D::Error::custom(format!(
                "Received invalid value for `dspy.ToolCallResults`: {}",
                refusal::python_str(&data)
            )));
        };
        Ok(Self {
            tool_call_results: Vec::<ToolCallResult>::deserialize(&results)
                .map_err(D::Error::custom)?,
        })
    }
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
            return Err(anyhow!(
                "`tool_calls` and `values` must have the same length."
            ));
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
