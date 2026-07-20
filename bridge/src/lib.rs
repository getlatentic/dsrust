//! Expose this crate's adapters to Python, so Python DSPy's own test suite can exercise them.
//!
//! The alternative is transliterating upstream's tests into Rust, which tests the
//! transliteration as much as the code. Here their pytest runs unchanged and the renderer
//! under it is ours.
//!
//! Python owns the DSPy signature object; this boundary takes the plain description of it —
//! instructions, ordered fields, and the already-formatted input values — because those are
//! the only things the renderer needs.

use dsrs::signature::{FieldKind, InField, OutField, Signature};
use dsrs::{Adapter, ChatAdapter, Example, JsonAdapter};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn kind_from(name: &str) -> PyResult<FieldKind> {
    match name {
        "str" => Ok(FieldKind::Str),
        "int" => Ok(FieldKind::Int),
        "float" => Ok(FieldKind::Float),
        "bool" => Ok(FieldKind::Bool),
        // Anything not a scalar arrives as `json:<annotation>`, so the Python type name
        // dspy prints survives the crossing instead of being flattened to "json".
        other => match other.strip_prefix("json:") {
            Some(annotation) => Ok(FieldKind::Json(annotation.to_owned())),
            None => Err(PyValueError::new_err(format!(
                "unsupported field kind: {other}"
            ))),
        },
    }
}

fn build_signature(
    instructions: &str,
    inputs: Vec<(String, String, String)>,
    outputs: Vec<(String, String, String, Option<String>)>,
) -> PyResult<Signature> {
    let inputs = inputs
        .into_iter()
        .map(|(name, kind, desc)| {
            Ok(InField {
                name,
                desc,
                kind: kind_from(&kind)?,
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    let outputs = outputs
        .into_iter()
        .map(|(name, kind, desc, schema)| {
            let schema = schema
                .map(|text| serde_json::from_str(&text))
                .transpose()
                .map_err(|error| PyValueError::new_err(format!("bad field schema: {error}")))?;
            Ok(OutField {
                name,
                desc,
                kind: kind_from(&kind)?,
                values: None,
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
#[pyfunction]
#[pyo3(signature = (adapter, instructions, inputs, outputs, values, demos = None))]
fn format_messages(
    adapter: &str,
    instructions: &str,
    inputs: Vec<(String, String, String)>,
    outputs: Vec<(String, String, String, Option<String>)>,
    values: Vec<(String, String)>,
    demos: Option<Vec<Vec<(String, String)>>>,
) -> PyResult<(String, Vec<(String, String)>)> {
    let signature = build_signature(instructions, inputs, outputs)?;
    let pairs: Vec<(&str, String)> = values
        .iter()
        .map(|(name, value)| (name.as_str(), value.clone()))
        .collect();
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
        .map(|turn| (turn.role.as_str().to_owned(), turn.content))
        .collect();
    Ok((system, turns))
}

/// Parse a raw reply through the named adapter, returning JSON text.
#[pyfunction]
#[pyo3(signature = (adapter, instructions, inputs, outputs, raw))]
fn parse_reply(
    adapter: &str,
    instructions: &str,
    inputs: Vec<(String, String, String)>,
    outputs: Vec<(String, String, String, Option<String>)>,
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
        .map_err(|error| PyValueError::new_err(format!("{error:#}")))
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
