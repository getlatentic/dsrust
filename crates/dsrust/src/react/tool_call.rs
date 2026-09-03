//! What dspy does with the arguments a tool is called with, and what it raises when they are wrong.
//!
//! `Tool.__call__` runs `_validate_and_parse_args`: each argument the caller gave, in the order
//! given, must be one the tool declares and must satisfy its schema; then pydantic parses each to
//! the parameter's type; then Python itself refuses the call if a required parameter is absent.
//! ReAct renders whichever exception lands as `Execution error in {tool}: …` ending in the
//! exception's own line, so the errors here carry that line as their message.

use anyhow::{Result, anyhow};
use serde_json::{Map, Value};

use super::schema_check;

/// The arguments the body runs with, or the error dspy raises: `ValueError` for an argument the
/// tool does not declare or one its schema refuses, `TypeError` for a required argument absent.
///
/// A schema is checked when it states a `type` other than `Any`, which is dspy's own gate; an
/// argument whose schema states none reaches the parser as it is.
pub fn parsed_args(
    tool: &str,
    given: &Value,
    declared: &Value,
    required: &[&str],
) -> Result<Map<String, Value>> {
    let empty = Map::new();
    let given = given.as_object().unwrap_or(&empty);
    for (key, value) in given {
        let Some(schema) = declared.get(key) else {
            return Err(anyhow!("ValueError: Arg {key} is not in the tool's args."));
        };
        let checked = schema.get("type").is_some_and(|kind| kind != "Any");
        if let Some(message) = checked
            .then(|| schema_check::message(value, schema))
            .flatten()
        {
            return Err(anyhow!("ValueError: Arg {key} is invalid: {message}"));
        }
    }
    let missing: Vec<&str> = required
        .iter()
        .copied()
        .filter(|name| !given.contains_key(*name))
        .collect();
    if !missing.is_empty() {
        return Err(anyhow!(
            "TypeError: {tool}() missing {} required positional argument{}: {}",
            missing.len(),
            if missing.len() == 1 { "" } else { "s" },
            named(&missing),
        ));
    }
    Ok(given
        .iter()
        .map(|(key, value)| (key.clone(), as_pydantic_parses(value, declared.get(key))))
        .collect())
}

/// An argument the schema admitted but the parameter's Rust type does not: the same `ValueError`
/// dspy raises for an invalid argument, carrying the parser's reason.
pub fn invalid_argument(key: &str, reason: impl std::fmt::Display) -> anyhow::Error {
    anyhow!("ValueError: Arg {key} is invalid: {reason}")
}

/// The text form of a tool's answer, for [`Tool::call`](super::Tool::call): a string bare, anything
/// else as this crate prints a value into a prompt. An agent observes the value itself.
pub fn observation_text(value: Value) -> String {
    match value {
        Value::String(text) => text,
        other => crate::adapter::python_json::json_dumps(&other),
    }
}

/// What pydantic's lax parsing changes about a JSON value on its way to a typed parameter: a float
/// with no fractional part becomes an integer wherever the schema asks for one.
fn as_pydantic_parses(value: &Value, schema: Option<&Value>) -> Value {
    let Some(schema) = schema else {
        return value.clone();
    };
    let admitted = schema
        .get("anyOf")
        .and_then(Value::as_array)
        .and_then(|branches| {
            branches
                .iter()
                .find(|branch| schema_check::message(value, branch).is_none())
        });
    if let Some(branch) = admitted {
        return as_pydantic_parses(value, Some(branch));
    }
    match (value, schema.get("type").and_then(Value::as_str)) {
        (Value::Number(number), Some("integer")) => match number.as_f64() {
            Some(float) if !number.is_i64() && !number.is_u64() && float.fract() == 0.0 => {
                Value::from(float as i64)
            }
            _ => value.clone(),
        },
        (Value::Array(items), _) => Value::Array(
            items
                .iter()
                .enumerate()
                .map(|(index, item)| {
                    let stated = schema
                        .get("prefixItems")
                        .and_then(|prefixes| prefixes.get(index))
                        .or_else(|| schema.get("items"));
                    as_pydantic_parses(item, stated)
                })
                .collect(),
        ),
        (Value::Object(fields), _) => Value::Object(
            fields
                .iter()
                .map(|(key, field)| {
                    let stated = schema
                        .get("properties")
                        .and_then(|properties| properties.get(key))
                        .or_else(|| {
                            schema
                                .get("additionalProperties")
                                .filter(|each| each.is_object())
                        });
                    (key.clone(), as_pydantic_parses(field, stated))
                })
                .collect(),
        ),
        _ => value.clone(),
    }
}

/// Python's own list of missing names: `'a'`, `'a' and 'b'`, `'a', 'b', and 'c'`.
fn named(missing: &[&str]) -> String {
    let quoted: Vec<String> = missing.iter().map(|name| format!("'{name}'")).collect();
    match quoted.as_slice() {
        [one] => one.clone(),
        [first, second] => format!("{first} and {second}"),
        [all @ .., last] => format!("{}, and {last}", all.join(", ")),
        [] => String::new(),
    }
}
