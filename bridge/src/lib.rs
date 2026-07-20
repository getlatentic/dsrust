//! Expose this crate's adapters to Python, so Python DSPy's own test suite can exercise them.
//!
//! The alternative is transliterating upstream's tests into Rust, which tests the
//! transliteration as much as the code. Here their pytest runs unchanged and the renderer
//! under it is ours.
//!
//! Python owns the DSPy signature object; this boundary takes the plain description of it —
//! instructions, ordered fields, and the already-formatted input values — because those are
//! the only things the renderer needs.

use dsrs::adapter::parse::FieldMismatch;
use dsrs::signature::{
    FieldKind, InField, JsonType, LiteralValue, OutField, Signature, TypeDescription,
};
use dsrs::{Adapter, ChatAdapter, Example, JsonAdapter};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::Value;

/// One input field as Python describes it: name, kind, description, any closed set, and the
/// prose any custom type in its annotation contributes.
type PyInField = (String, String, String, Option<String>, Option<String>);
/// One output field, which additionally carries the nested schema of a `Json` field.
type PyOutField = (
    String,
    String,
    String,
    Option<String>,
    Option<String>,
    Option<String>,
);

fn kind_from(name: &str, descriptions: Option<String>) -> PyResult<FieldKind> {
    match name {
        "str" => Ok(FieldKind::Str),
        "int" => Ok(FieldKind::Int),
        "float" => Ok(FieldKind::Float),
        "bool" => Ok(FieldKind::Bool),
        // Anything not a scalar arrives as `json:<annotation>`, so the Python type name
        // dspy prints survives the crossing instead of being flattened to "json".
        other => match other.strip_prefix("json:") {
            Some(annotation) => Ok(FieldKind::Json(JsonType {
                annotation: annotation.to_owned(),
                descriptions: type_descriptions_from(descriptions)?,
            })),
            None => Err(PyValueError::new_err(format!(
                "unsupported field kind: {other}"
            ))),
        },
    }
}

/// Which custom types an annotation names is Python reflection, so Python extracts the pairs
/// and this side renders them: the crossing carries `[[name, prose], ...]` as JSON text.
fn type_descriptions_from(descriptions: Option<String>) -> PyResult<Vec<TypeDescription>> {
    let Some(text) = descriptions else {
        return Ok(Vec::new());
    };
    let entries: Vec<Value> = serde_json::from_str(&text)
        .map_err(|error| PyValueError::new_err(format!("bad type descriptions: {error}")))?;
    entries
        .iter()
        .map(|entry| {
            let field = |name: &str| {
                entry.get(name).and_then(Value::as_str).ok_or_else(|| {
                    PyValueError::new_err(format!("type description is missing `{name}`"))
                })
            };
            Ok(TypeDescription {
                name: field("name")?.to_owned(),
                text: field("text")?.to_owned(),
                replaces_schema: entry
                    .get("replaces_schema")
                    .and_then(Value::as_bool)
                    .unwrap_or_default(),
            })
        })
        .collect()
}

/// A closed set crosses as JSON text, the way a field schema does: Python's `Literal` mixes
/// member types freely and JSON is the one spelling that carries each of them intact.
fn closed_set_from(values: Option<String>) -> PyResult<Option<Vec<LiteralValue>>> {
    let Some(text) = values else {
        return Ok(None);
    };
    let members: Vec<Value> = serde_json::from_str(&text)
        .map_err(|error| PyValueError::new_err(format!("bad closed set: {error}")))?;
    members
        .into_iter()
        .map(literal_from)
        .collect::<PyResult<Vec<_>>>()
        .map(Some)
}

fn literal_from(member: Value) -> PyResult<LiteralValue> {
    match member {
        Value::String(text) => Ok(LiteralValue::Str(text)),
        Value::Bool(flag) => Ok(LiteralValue::Bool(flag)),
        Value::Number(number) => number.as_i64().map(LiteralValue::Int).ok_or_else(|| {
            PyValueError::new_err(format!("closed set member is not an integer: {number}"))
        }),
        other => Err(PyValueError::new_err(format!(
            "closed set member has no Literal spelling: {other}"
        ))),
    }
}

fn build_signature(
    instructions: &str,
    inputs: Vec<PyInField>,
    outputs: Vec<PyOutField>,
) -> PyResult<Signature> {
    let inputs = inputs
        .into_iter()
        .map(|(name, kind, desc, values, descriptions)| {
            Ok(InField {
                name,
                desc,
                kind: kind_from(&kind, descriptions)?,
                values: closed_set_from(values)?,
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    let outputs = outputs
        .into_iter()
        .map(|(name, kind, desc, schema, values, descriptions)| {
            let schema = schema
                .map(|text| serde_json::from_str(&text))
                .transpose()
                .map_err(|error| PyValueError::new_err(format!("bad field schema: {error}")))?;
            Ok(OutField {
                name,
                desc,
                kind: kind_from(&kind, descriptions)?,
                values: closed_set_from(values)?,
                schema,
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(Signature {
        instructions: instructions.to_owned(),
        inputs,
        outputs,
    })
}

/// Render one exchange for the named adapter, as `(system, [(role, content), ...])`.
///
/// Each content crosses as JSON, because a message carrying a custom type is a list of blocks
/// rather than a string and both spellings have to survive intact.
#[pyfunction]
#[pyo3(signature = (adapter, instructions, inputs, outputs, values, demos = None))]
fn format_messages(
    adapter: &str,
    instructions: &str,
    inputs: Vec<PyInField>,
    outputs: Vec<PyOutField>,
    values: Vec<(String, String)>,
    demos: Option<Vec<Vec<(String, String)>>>,
) -> PyResult<(String, Vec<(String, String)>)> {
    let signature = build_signature(instructions, inputs, outputs)?;
    // Python sends each value as JSON text so its structure survives the crossing; the
    // adapter renders it, which is where dspy renders too.
    let pairs: Vec<(&str, Value)> = values
        .iter()
        .map(|(name, json)| {
            serde_json::from_str(json)
                .map(|value| (name.as_str(), value))
                .map_err(|error| PyValueError::new_err(format!("input `{name}`: {error}")))
        })
        .collect::<PyResult<_>>()?;
    let adapter: Box<dyn Adapter> = match adapter {
        "chat" => Box::new(ChatAdapter::default()),
        "json" => Box::new(JsonAdapter),
        other => return Err(PyValueError::new_err(format!("unknown adapter: {other}"))),
    };
    let demos: Vec<Example> = demos
        .unwrap_or_default()
        .into_iter()
        .map(|fields| {
            Example::new(
                fields
                    .into_iter()
                    .map(|(name, value)| (name, serde_json::Value::String(value))),
            )
        })
        .collect();
    let (system, turns) = adapter.format(&signature, &demos, &pairs);
    let turns = turns
        .into_iter()
        .map(|turn| {
            let content = serde_json::to_string(&turn.content)
                .map_err(|error| PyValueError::new_err(format!("bad content: {error}")))?;
            Ok((turn.role.as_str().to_owned(), content))
        })
        .collect::<PyResult<_>>()?;
    Ok((system, turns))
}

/// Parse a raw reply through the named adapter, returning JSON text.
#[pyfunction]
#[pyo3(signature = (adapter, instructions, inputs, outputs, raw))]
fn parse_reply(
    adapter: &str,
    instructions: &str,
    inputs: Vec<PyInField>,
    outputs: Vec<PyOutField>,
    raw: &str,
) -> PyResult<String> {
    let signature = build_signature(instructions, inputs, outputs)?;
    let adapter: Box<dyn Adapter> = match adapter {
        "chat" => Box::new(ChatAdapter::default()),
        "json" => Box::new(JsonAdapter),
        other => return Err(PyValueError::new_err(format!("unknown adapter: {other}"))),
    };
    adapter
        .parse(&signature, raw)
        .map(|value| value.to_string())
        // A reply that read as JSON but named the wrong fields carries whichever declared ones
        // it did have. dspy reports those on the error, so they cross as a second argument
        // rather than being flattened into the message.
        .map_err(|error| match error.downcast_ref::<FieldMismatch>() {
            Some(mismatch) => {
                PyValueError::new_err((format!("{error:#}"), mismatch.parsed.to_string()))
            }
            None => PyValueError::new_err(format!("{error:#}")),
        })
}

/// Whether the named adapter, configured this way, offers a fallback when a reply fails to
/// parse. The decision lives in Rust so the Python side cannot quietly answer it instead.
#[pyfunction]
#[pyo3(signature = (adapter, use_json_adapter_fallback))]
fn has_json_fallback(adapter: &str, use_json_adapter_fallback: bool) -> PyResult<bool> {
    let adapter: Box<dyn Adapter> = match adapter {
        "chat" => Box::new(ChatAdapter {
            use_json_adapter_fallback,
        }),
        "json" => Box::new(JsonAdapter),
        other => return Err(PyValueError::new_err(format!("unknown adapter: {other}"))),
    };
    Ok(adapter.json_fallback().is_some())
}

#[pymodule]
fn dsrs_bridge(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(format_messages, module)?)?;
    module.add_function(wrap_pyfunction!(parse_reply, module)?)?;
    module.add_function(wrap_pyfunction!(has_json_fallback, module)?)?;
    Ok(())
}
