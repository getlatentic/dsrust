//! pydantic's lax casting of a structured value, by the field's JSON schema.
//!
//! dspy hands every parsed value to `TypeAdapter(annotation).validate_python`, which reads a
//! container's leaves as the types the annotation declares: `"1"` becomes `1` under `list[int]`,
//! `"yes"` becomes `True` under `list[bool]`, and a leaf that will not read is a parse failure.
//! Every rule here was measured on pydantic 2.13 and holds only as far as JSON can carry it — a
//! model's extra keys, a `datetime`, an `inf` stay as they arrived.
use anyhow::{Result, anyhow};
use serde_json::{Map, Value};

use super::OutField;

/// The field's value cast by its schema, in place. A value with no schema to read it by is kept.
pub(crate) fn cast(field: &OutField, value: &mut Value) -> Result<()> {
    // One of dspy's own types validates itself — `ToolCalls` accepts a call written with either
    // `args` or `arguments`, `Code` takes a bare string — so its schema is a description of the
    // shape rather than the contract, and reading a value by it would refuse what the type takes.
    // Upstream hands these to the type, and so does this crate, in `coerce_value`.
    if super::annotation::is_dspy_type(&field.annotation()) {
        return Ok(());
    }
    let Some(schema) = field.schema.as_ref() else {
        return Ok(());
    };
    let definitions = schema
        .get("$defs")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    cast_by(schema, &definitions, value, &field.name, Mode::Lax)
}

/// pydantic validates a union in smart mode: every arm strictly first, then every arm laxly.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Strict,
    Lax,
}

fn cast_by(
    schema: &Value,
    definitions: &Map<String, Value>,
    value: &mut Value,
    path: &str,
    mode: Mode,
) -> Result<()> {
    let schema = resolve(schema, definitions);
    if let Some(arms) = schema
        .get("anyOf")
        .and_then(Value::as_array)
        .filter(|arms| !arms.is_empty())
    {
        return cast_union(arms, definitions, value, path, mode);
    }
    if let Some(members) = schema.get("enum").and_then(Value::as_array) {
        return match members.iter().find(|member| python_equal(member, value)) {
            Some(member) => {
                *value = member.clone();
                Ok(())
            }
            None => Err(anyhow!(
                "{path} must be one of {}, got {}",
                crate::python::tuple(members),
                crate::python::repr(value)
            )),
        };
    }
    match schema.get("type").and_then(Value::as_str) {
        Some("integer") => cast_integer(value, path, mode),
        Some("number") => cast_number(value, path, mode),
        Some("boolean") => cast_boolean(value, path, mode),
        Some("string") if value.is_string() => Ok(()),
        Some("string") => Err(anyhow!(
            "{path} must be a string, got {}",
            crate::python::repr(value)
        )),
        Some("null") if value.is_null() => Ok(()),
        Some("null") => Err(anyhow!(
            "{path} must be None, got {}",
            crate::python::repr(value)
        )),
        Some("array") => cast_array(schema, definitions, value, path, mode),
        Some("object") => cast_object(schema, definitions, value, path, mode),
        _ => Ok(()),
    }
}

fn cast_union(
    arms: &[Value],
    definitions: &Map<String, Value>,
    value: &mut Value,
    path: &str,
    mode: Mode,
) -> Result<()> {
    let passes: &[Mode] = match mode {
        Mode::Strict => &[Mode::Strict],
        Mode::Lax => &[Mode::Strict, Mode::Lax],
    };
    let mut last = None;
    for pass in passes {
        for arm in arms {
            let mut candidate = value.clone();
            match cast_by(arm, definitions, &mut candidate, path, *pass) {
                Ok(()) => {
                    *value = candidate;
                    return Ok(());
                }
                Err(error) => last = Some(error),
            }
        }
    }
    Err(last.unwrap_or_else(|| anyhow!("{path} matches no arm of its union")))
}

/// A JSON integer; laxly also a whole-valued float, a bool, or text spelling a whole number — with
/// a sign, digit-group underscores, or a zero fraction.
fn cast_integer(value: &mut Value, path: &str, mode: Mode) -> Result<()> {
    if value.as_i64().is_some() || value.as_u64().is_some() {
        return Ok(());
    }
    let shown = crate::python::repr(value);
    let refusal = || anyhow!("{path} must be an integer, got {shown}");
    if mode == Mode::Strict {
        return Err(refusal());
    }
    let parsed = match &*value {
        Value::Number(number) => number
            .as_f64()
            .filter(|f| f.fract() == 0.0 && f.is_finite())
            .map(|f| f as i64),
        Value::Bool(flag) => Some(i64::from(*flag)),
        Value::String(text) => whole_number(text),
        _ => None,
    };
    match parsed {
        Some(parsed) => {
            *value = Value::from(parsed);
            Ok(())
        }
        None => Err(refusal()),
    }
}

/// Python's own `int()` reading of text, plus pydantic's acceptance of a zero fraction.
fn whole_number(text: &str) -> Option<i64> {
    let text = text.trim();
    let (sign, digits) = match text.strip_prefix('-') {
        Some(rest) => (-1, rest),
        None => (1, text.strip_prefix('+').unwrap_or(text)),
    };
    let (whole, fraction) = digits.split_once('.').unwrap_or((digits, ""));
    if whole.is_empty() || !grouped_digits(whole) || !fraction.chars().all(|c| c == '0') {
        return None;
    }
    whole.replace('_', "").parse::<i64>().ok().map(|n| sign * n)
}

/// ASCII digits with single underscores between them.
fn grouped_digits(text: &str) -> bool {
    !text.starts_with('_')
        && !text.ends_with('_')
        && !text.contains("__")
        && text.chars().all(|c| c.is_ascii_digit() || c == '_')
}

/// A JSON number; laxly also a bool or text spelling a number. `inf` and `nan` are numbers to
/// pydantic and not to JSON, so they stay the text they were.
fn cast_number(value: &mut Value, path: &str, mode: Mode) -> Result<()> {
    if value.is_number() {
        return Ok(());
    }
    let shown = crate::python::repr(value);
    let refusal = || anyhow!("{path} must be a number, got {shown}");
    if mode == Mode::Strict {
        return Err(refusal());
    }
    let parsed = match &*value {
        Value::Bool(flag) => Some(Some(f64::from(u8::from(*flag)))),
        Value::String(text) => {
            let plain = text.trim().replace('_', "");
            match plain.to_ascii_lowercase().as_str() {
                "inf" | "+inf" | "-inf" | "infinity" | "+infinity" | "-infinity" | "nan" => {
                    Some(None)
                }
                _ => plain
                    .parse::<f64>()
                    .ok()
                    .filter(|f| f.is_finite())
                    .map(Some),
            }
        }
        _ => None,
    };
    match parsed {
        Some(Some(number)) => {
            *value = Value::from(number);
            Ok(())
        }
        Some(None) => Ok(()),
        None => Err(refusal()),
    }
}

/// A JSON bool; laxly also `0`/`1` as a number, or one of pydantic's twelve words in any case.
fn cast_boolean(value: &mut Value, path: &str, mode: Mode) -> Result<()> {
    if value.is_boolean() {
        return Ok(());
    }
    let shown = crate::python::repr(value);
    let refusal = || anyhow!("{path} must be a bool, got {shown}");
    if mode == Mode::Strict {
        return Err(refusal());
    }
    let parsed = match &*value {
        Value::Number(number) => match number.as_f64() {
            Some(1.0) => Some(true),
            Some(0.0) => Some(false),
            _ => None,
        },
        Value::String(text) => match text.to_ascii_lowercase().as_str() {
            "true" | "yes" | "on" | "1" | "t" | "y" => Some(true),
            "false" | "no" | "off" | "0" | "f" | "n" => Some(false),
            _ => None,
        },
        _ => None,
    };
    match parsed {
        Some(parsed) => {
            *value = Value::Bool(parsed);
            Ok(())
        }
        None => Err(refusal()),
    }
}

fn cast_array(
    schema: &Value,
    definitions: &Map<String, Value>,
    value: &mut Value,
    path: &str,
    mode: Mode,
) -> Result<()> {
    let Value::Array(items) = value else {
        return Err(anyhow!(
            "{path} must be a list, got {}",
            crate::python::repr(value)
        ));
    };
    if let Some(positions) = schema.get("prefixItems").and_then(Value::as_array) {
        if positions.len() != items.len() {
            return Err(anyhow!(
                "{path} must hold {} items, got {}",
                positions.len(),
                items.len()
            ));
        }
        for (index, (item, item_schema)) in items.iter_mut().zip(positions).enumerate() {
            cast_by(
                item_schema,
                definitions,
                item,
                &format!("{path}.{index}"),
                mode,
            )?;
        }
        return Ok(());
    }
    let item_schema = schema
        .get("items")
        .cloned()
        .unwrap_or_else(|| Value::Object(Map::new()));
    for (index, item) in items.iter_mut().enumerate() {
        cast_by(
            &item_schema,
            definitions,
            item,
            &format!("{path}.{index}"),
            mode,
        )?;
    }
    Ok(())
}

fn cast_object(
    schema: &Value,
    definitions: &Map<String, Value>,
    value: &mut Value,
    path: &str,
    mode: Mode,
) -> Result<()> {
    let Value::Object(fields) = value else {
        return Err(anyhow!(
            "{path} must be a mapping, got {}",
            crate::python::repr(value)
        ));
    };
    let properties = schema.get("properties").and_then(Value::as_object);
    if let Some(required) = schema.get("required").and_then(Value::as_array) {
        for name in required.iter().filter_map(Value::as_str) {
            if !fields.contains_key(name) {
                return Err(anyhow!("{path}.{name} is required"));
            }
        }
    }
    let additional = schema
        .get("additionalProperties")
        .filter(|extra| extra.is_object());
    for (name, field) in fields.iter_mut() {
        let child = properties
            .and_then(|properties| properties.get(name))
            .or(additional);
        if let Some(child) = child {
            cast_by(child, definitions, field, &format!("{path}.{name}"), mode)?;
        }
    }
    Ok(())
}

fn resolve<'a>(schema: &'a Value, definitions: &'a Map<String, Value>) -> &'a Value {
    schema
        .get("$ref")
        .and_then(Value::as_str)
        .and_then(|reference| definitions.get(reference.rsplit('/').next().unwrap_or(reference)))
        .unwrap_or(schema)
}

/// Python's `==` between a `Literal` member and a value: `1 == True` and `0 == False`.
fn python_equal(member: &Value, value: &Value) -> bool {
    if member == value {
        return true;
    }
    match (member, value) {
        (Value::Bool(flag), Value::Number(number)) | (Value::Number(number), Value::Bool(flag)) => {
            number.as_f64() == Some(f64::from(u8::from(*flag)))
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signature::annotation::schema_for;
    use serde_json::json;

    fn read(annotation: &str, value: Value) -> Result<Value> {
        let field = OutField {
            name: "field".to_owned(),
            schema: schema_for(annotation),
            ..OutField::default()
        };
        let mut value = value;
        cast(&field, &mut value).map(|()| value)
    }

    /// `TypeAdapter(annotation).validate_python(value)` on pydantic 2.13, each row measured.
    #[test]
    fn leaves_read_as_pydantic_reads_them() {
        assert_eq!(
            read(
                "list[int]",
                json!(["1", " 1 ", "1.0", 1.0, true, "1_000", "+1", " -2"])
            )
            .unwrap(),
            json!([1, 1, 1, 1, 1, 1000, 1, -2])
        );
        for bad in ["1.5", "abc", "", "0x10", "١"] {
            assert!(read("list[int]", json!([bad])).is_err(), "{bad}");
        }
        assert!(read("list[int]", json!([1.5])).is_err());
        assert_eq!(
            read(
                "list[float]",
                json!(["1", "1.5", "1e3", 1, true, " 2.5 ", "1_000.5"])
            )
            .unwrap(),
            json!([1.0, 1.5, 1000.0, 1, 1.0, 2.5, 1000.5])
        );
        assert!(
            read("list[float]", json!(["abc"])).is_err()
                && read("list[float]", json!([""])).is_err()
        );
        assert_eq!(
            read(
                "list[bool]",
                json!([
                    "true", "TRUE", "yes", "no", "on", "off", "1", "0", "t", "f", "y", "n", 1, 0,
                    1.0
                ])
            )
            .unwrap(),
            json!([
                true, true, true, false, true, false, true, false, true, false, true, false, true,
                false, true
            ])
        );
        for bad in [json!("2"), json!(""), json!(" true "), json!(2)] {
            assert!(read("list[bool]", json!([bad])).is_err(), "{bad}");
        }
        for bad in [json!(1), json!(1.5), json!(true), json!(null), json!(["a"])] {
            assert!(read("list[str]", json!([bad])).is_err(), "{bad}");
        }
    }

    #[test]
    fn containers_and_unions_read_as_pydantic_reads_them() {
        assert_eq!(
            read("dict[str, int]", json!({"a": "1"})).unwrap(),
            json!({"a": 1})
        );
        assert!(read("dict[str, int]", json!({"a": "x"})).is_err());
        assert!(
            read("list[int]", json!("1")).is_err(),
            "a string is not a list"
        );
        assert_eq!(read("list[Any]", json!(["1", 2])).unwrap(), json!(["1", 2]));
        assert_eq!(
            read("dict[str, Any]", json!({"a": "1"})).unwrap(),
            json!({"a": "1"})
        );
        assert_eq!(
            read("list[Optional[int]]", json!(["1", null])).unwrap(),
            json!([1, null])
        );
        assert!(read("list[Optional[int]]", json!([""])).is_err());
        assert_eq!(
            read("list[int | str]", json!(["1", 1, "a", 1.0, true])).unwrap(),
            json!(["1", 1, "a", 1, 1])
        );
        assert_eq!(
            read("list[str | int]", json!([1, "1"])).unwrap(),
            json!([1, "1"])
        );
        assert!(read("list[str | int]", json!([1.5])).is_err());
        assert_eq!(
            read("list[float | int]", json!([1, "1", "1.5", 1.5])).unwrap(),
            json!([1, 1.0, 1.5, 1.5])
        );
        assert_eq!(
            read("list[int | float]", json!([1.5, "1.5", "1", 1])).unwrap(),
            json!([1.5, 1.5, 1, 1])
        );
        assert_eq!(
            read("list[bool | int]", json!([1, "1", "yes", true])).unwrap(),
            json!([1, true, true, true])
        );
        assert!(read("list[Optional[str]]", json!([1])).is_err());
        assert_eq!(
            read("tuple[int, int]", json!(["1", "2"])).unwrap(),
            json!([1, 2])
        );
        assert!(read("tuple[int, int]", json!(["1"])).is_err());
    }

    /// dspy's own types read their own values: `ToolCalls` takes a call whose arguments are under
    /// `arguments` as readily as one under `args`, which its JSON schema does not say.
    #[test]
    fn one_of_dspys_own_types_is_left_to_itself() {
        let field = OutField {
            name: "tool_calls".to_owned(),
            kind: crate::signature::FieldKind::Json(crate::signature::JsonType {
                annotation: "ToolCalls".to_owned(),
                ..Default::default()
            }),
            schema: Some(crate::ToolCalls::output_schema()),
            ..OutField::default()
        };
        let written =
            json!({ "tool_calls": [{ "name": "submit", "arguments": { "answer": "done" } }] });
        let mut value = written.clone();
        cast(&field, &mut value).expect("the type reads it, not the schema");
        assert_eq!(value, written, "the value reached the type unchanged");
    }
}
