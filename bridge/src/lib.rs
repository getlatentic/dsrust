//! Expose this crate's adapters to Python, so Python DSPy's own test suite can exercise them.
//!
//! The alternative is transliterating upstream's tests into Rust, which tests the
//! transliteration as much as the code. Here their pytest runs unchanged and the renderer
//! under it is ours.
//!
//! Python owns the DSPy signature object; this boundary takes the plain description of it —
//! instructions, ordered fields, and the already-formatted input values — because those are
//! the only things the renderer needs.

mod code_modules;

use dsrust::adapter::Input;
use dsrust::adapter::parse::FieldMismatch;
use dsrust::adapter::xml::XmlAdapter;
use dsrust::lm::DynChatModel;
use dsrust::signature::{
    FieldKind, InField, JsonType, LiteralValue, OutField, Signature, TypeDescription,
};
use dsrust::{Adapter, BamlAdapter, ChatAdapter, Example, JsonAdapter, TwoStepAdapter};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::Value;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

/// A model that exists only to be unused. See [`adapter_named`].
struct NotOnThisSide;

impl DynChatModel for NotOnThisSide {
    fn forward_dyn<'a>(
        &'a self,
        _http: &'a reqwest::Client,
        _request: &'a dsrust::lm::api::LmRequest,
    ) -> Pin<Box<dyn Future<Output = anyhow::Result<dsrust::lm::api::LmResponse>> + Send + 'a>>
    {
        Box::pin(async {
            Err(anyhow::anyhow!(
                "the bridge does not call models; Python runs the extraction"
            ))
        })
    }

    /// Nothing, which is also what a model that never answers can honour.
    fn capabilities_dyn<'a>(
        &'a self,
        _http: &'a reqwest::Client,
    ) -> Pin<Box<dyn Future<Output = dsrust::lm::Capabilities> + Send + 'a>> {
        Box::pin(std::future::ready(dsrust::lm::Capabilities::default()))
    }

    fn native_reasoning_usable_dyn(&self) -> bool {
        true
    }
}

/// A model backed by a Python LM object, so this crate's own `Predict::forward` can run under
/// dspy's module tests, driven by the test's `DummyLM`. The reply is canned and synchronous, so
/// the async seam is met with a ready future rather than a runtime.
pub(crate) struct PyLM {
    pub(crate) inner: Py<PyAny>,
}

impl PyLM {
    /// Call the Python LM the way a dspy module does — `lm(messages=..., n=...)`, which is
    /// `BaseLM.__call__` — so it records `lm.history` (tests read it) and returns the completions
    /// list. `DummyLM.forward` would skip the history and read a false green, so it is not used.
    fn answer(
        &self,
        request: &dsrust::lm::api::LmRequest,
    ) -> anyhow::Result<dsrust::lm::api::LmResponse> {
        Python::attach(|py| {
            // Cross the rendered messages as JSON text, the way every value crosses this bridge,
            // and let Python rebuild the list — no Rust-to-Python object mapping to keep in step.
            let messages_json = serde_json::to_string(&request.wire_messages())?;
            let messages = py.import("json")?.call_method1("loads", (messages_json,))?;
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("messages", messages)?;
            // `n` is the one kwarg a `DummyLM` reads — how many canned choices to return.
            if let Some(n) = request.config.n {
                kwargs.set_item("n", n)?;
            }
            // Whatever the normalized config does not model travels in `extensions`, and dspy's own
            // `LM.forward` opens with `kwargs = {**extensions, **kwargs}`. Passing them on is what
            // lets the crate's decision about one — a predicted output lifted off the inputs —
            // reach litellm, rather than the Python side having decided it before Rust ran.
            for (key, value) in &request.config.extensions {
                let crossed = py
                    .import("json")?
                    .call_method1("loads", (value.to_string(),))?;
                kwargs.set_item(key.as_str(), crossed)?;
            }
            let replies = self
                .inner
                .bind(py)
                .call((), Some(&kwargs))
                .map_err(|error| anyhow::anyhow!("the python LM raised: {error}"))?;
            let mut texts = Vec::new();
            for reply in replies.try_iter()? {
                match reply?.extract::<String>() {
                    Ok(text) => texts.push(text),
                    // dspy flattens a text-only reply to a bare string; a dict means a channel
                    // beside the text — tool calls, native reasoning, citations. The text-only
                    // beachhead cannot answer for those, and taking the `text` key while dropping
                    // the rest would be a false green, so it stops here for the shim to mark xfail.
                    Err(_) => anyhow::bail!(MODULE_UNSUPPORTED),
                }
            }
            Ok(dsrust::lm::api::LmResponse::completions(texts))
        })
    }
}

/// The sentinel a reply-with-extra-channels raises with, which the Python shim turns into the
/// bridge's `Unsupported` so the case is a tracked xfail rather than a silent pass or a hard error.
const MODULE_UNSUPPORTED: &str = "MODULE_UNSUPPORTED: the LM reply carried non-text channels";

impl DynChatModel for PyLM {
    fn forward_dyn<'a>(
        &'a self,
        _http: &'a reqwest::Client,
        request: &'a dsrust::lm::api::LmRequest,
    ) -> Pin<Box<dyn Future<Output = anyhow::Result<dsrust::lm::api::LmResponse>> + Send + 'a>>
    {
        Box::pin(std::future::ready(self.answer(request)))
    }

    fn capabilities_dyn<'a>(
        &'a self,
        _http: &'a reqwest::Client,
    ) -> Pin<Box<dyn Future<Output = dsrust::lm::Capabilities> + Send + 'a>> {
        Box::pin(std::future::ready(dsrust::lm::Capabilities::default()))
    }

    fn native_reasoning_usable_dyn(&self) -> bool {
        true
    }
}

/// One input field as Python describes it: name, kind, description, any closed set, the prose
/// any custom type in its annotation contributes, the annotation's reflected structure, and
/// what its pydantic constraints read as.
pub(crate) type PyInField = (
    String,
    String,
    String,
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
);
/// One output field, which additionally carries the nested schema of a `Json` field.
pub(crate) type PyOutField = (
    String,
    String,
    String,
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
);

fn kind_from(
    name: &str,
    descriptions: Option<String>,
    reflection: Option<String>,
) -> PyResult<FieldKind> {
    match name {
        "str" => Ok(FieldKind::Str),
        "int" => Ok(FieldKind::Int),
        "float" => Ok(FieldKind::Float),
        "bool" => Ok(FieldKind::Bool),
        // dspy 3.3's str-like `Reasoning`: prints "str" and carries no schema, but is not the
        // `str` type, so it keeps the output-requirement hint.
        "reasoning" => Ok(FieldKind::Reasoning),
        // Anything not a scalar arrives as `json:<annotation>`, so the Python type name
        // dspy prints survives the crossing instead of being flattened to "json".
        other if other.starts_with("enum:") => Ok(FieldKind::Enum(
            other.trim_start_matches("enum:").to_owned(),
        )),
        other => match other.strip_prefix("json:") {
            Some(annotation) => Ok(FieldKind::Json(JsonType {
                annotation: annotation.to_owned(),
                descriptions: type_descriptions_from(descriptions)?,
                reflection: json_text(reflection, "type reflection")?,
            })),
            None => Err(PyValueError::new_err(format!(
                "unsupported field kind: {other}"
            ))),
        },
    }
}

/// A structure Python reflected, as the JSON text it crossed in. Only Python can read a type
/// off an annotation, and only this crate decides how the result reads, so what travels is the
/// description itself rather than anything rendered from it.
fn json_text(text: Option<String>, what: &str) -> PyResult<Option<Value>> {
    text.map(|text| {
        serde_json::from_str(&text)
            .map_err(|error| PyValueError::new_err(format!("bad {what}: {error}")))
    })
    .transpose()
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
        // `{"bare": "Colour.RED"}` — a member Python prints as itself rather than as a literal.
        Value::Object(ref fields) if fields.contains_key("bare") => fields["bare"]
            .as_str()
            .map(|text| LiteralValue::Bare(text.to_owned()))
            .ok_or_else(|| PyValueError::new_err("bare closed set member is not text")),
        other => Err(PyValueError::new_err(format!(
            "closed set member has no Literal spelling: {other}"
        ))),
    }
}

pub(crate) fn build_signature(
    instructions: &str,
    inputs: Vec<PyInField>,
    outputs: Vec<PyOutField>,
) -> PyResult<Signature> {
    let inputs = inputs
        .into_iter()
        .map(
            |(name, kind, desc, values, descriptions, reflection, constraints)| {
                Ok(InField {
                    name,
                    desc,
                    kind: kind_from(&kind, descriptions, reflection)?,
                    values: closed_set_from(values)?,
                    constraints,
                    // Python owns the prefix; nothing crossing here has one set, and the
                    // renderer infers it from the name exactly as dspy does.
                    prefix: None,
                })
            },
        )
        .collect::<PyResult<Vec<_>>>()?;
    let outputs = outputs
        .into_iter()
        .map(
            |(name, kind, desc, schema, values, descriptions, reflection, constraints)| {
                Ok(OutField {
                    name,
                    desc,
                    kind: kind_from(&kind, descriptions, reflection)?,
                    values: closed_set_from(values)?,
                    schema: json_text(schema, "field schema")?,
                    constraints,
                    prefix: None,
                })
            },
        )
        .collect::<PyResult<Vec<_>>>()?;
    Ok(Signature {
        instructions: instructions.to_owned(),
        inputs,
        outputs,
    })
}

/// The crate's adapter for the name Python sends.
fn adapter_named(adapter: &str) -> PyResult<Box<dyn Adapter>> {
    configured_adapter(adapter, false)
}

/// The same, carrying dspy's `use_native_function_calling`. Only the two formats that model the
/// setting read it; the rest render as an adapter with it off, which is upstream's default.
fn configured_adapter(adapter: &str, native: bool) -> PyResult<Box<dyn Adapter>> {
    match adapter {
        "chat" => Ok(Box::new(
            ChatAdapter::default().with_native_function_calling(native),
        )),
        "json" => Ok(Box::new(JsonAdapter {
            use_native_function_calling: native,
            parallel_tool_calls: None,
        })),
        "xml" => Ok(Box::new(XmlAdapter)),
        "baml" => Ok(Box::new(BamlAdapter)),
        // Rendering a two-step exchange never reaches the extraction model — Python holds the
        // models on this side of the bridge and runs that second ask itself. This stands in for
        // one, and says so loudly rather than quietly answering if that ever stops being true.
        "two_step" => Ok(Box::new(TwoStepAdapter::new(Arc::new(NotOnThisSide)))),
        other => Err(PyValueError::new_err(format!("unknown adapter: {other}"))),
    }
}

/// dspy `format_field_description`: the fields named, ahead of the structure section.
#[pyfunction]
fn field_description(
    instructions: &str,
    inputs: Vec<PyInField>,
    outputs: Vec<PyOutField>,
) -> PyResult<String> {
    let signature = build_signature(instructions, inputs, outputs)?;
    Ok(dsrust::adapter::field_description(&signature))
}

/// dspy's `infer_prefix`: the label an adapter prints in front of a field, from its name.
///
/// Pure string work with no annotation to walk, so the whole decision belongs on this side of
/// the boundary rather than only the rendering of its result.
#[pyfunction]
fn infer_prefix(name: &str) -> String {
    dsrust::signature::infer_prefix(name)
}

/// dspy `majority`: which of several answers wins, as an index into them.
///
/// The vote is the decision and it crosses; Python keeps the completions and returns the one at
/// the index, because which object comes back is its container's business rather than the vote's.
#[pyfunction]
fn majority_index(values: Vec<String>, mode: &str) -> PyResult<usize> {
    let normalize = match mode {
        "default" => dsrust::predict::Normalize::Default,
        "identity" => dsrust::predict::Normalize::AsWritten,
        "text" => dsrust::predict::Normalize::Text,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown normalize mode {other:?}"
            )));
        }
    };
    let completions: Vec<dsrust::Example> = values
        .iter()
        .map(|value| dsrust::Example::new([("value", Value::String(value.clone()))]))
        .collect();
    let winner =
        dsrust::predict::majority(&completions, &normalize, None).map_err(to_value_error)?;
    let won = winner
        .get("value")
        .and_then(Value::as_str)
        .unwrap_or_default();
    completions
        .iter()
        .position(|c| c.get("value").and_then(Value::as_str) == Some(won))
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("the winner is not among the votes"))
}

/// dspy `Example.inputs()` / `Example.labels()`: which of a record's fields are which.
///
/// The split is the decision — a field is an input because it was declared one, and a label
/// because it was not — and it belongs to the crate. Python keeps the values and rebuilds the
/// record around the answer, which is the same division the adapters follow.
///
/// `declared` being absent is upstream's `ValueError`, raised rather than answered, because a
/// record that never said which fields it was asked about cannot be split at all.
#[pyfunction]
fn split_example(
    names: Vec<String>,
    declared: Option<Vec<String>>,
) -> PyResult<(Vec<String>, Vec<String>)> {
    let mut example = Example::new(names.iter().map(|name| (name.as_str(), Value::Null)));
    if let Some(declared) = declared {
        example = example.with_inputs(declared);
    }
    let named = |split: Example| split.fields().map(|(name, _)| name.to_owned()).collect();
    let inputs = example.inputs().map_err(to_value_error)?;
    let labels = example.labels().map_err(to_value_error)?;
    Ok((named(inputs), named(labels)))
}

pub(crate) fn to_value_error(error: anyhow::Error) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(error.to_string())
}

/// The instruction dspy's `_create_extractor_signature` writes for the second ask.
///
/// The extractor's *fields* stay Python's: their annotations are Python types this side cannot
/// build. What it asks for is the crate's, so it is written here and crosses as text.
#[pyfunction]
fn extractor_instructions(outputs: Vec<PyOutField>) -> PyResult<String> {
    let signature = build_signature("", Vec::new(), outputs)?;
    Ok(dsrust::adapter::extractor_signature(&signature).instructions)
}

/// Render one exchange for the named adapter, as `(system, [(role, content), ...])`.
///
/// Each content crosses as JSON, because a message carrying a custom type is a list of blocks
/// rather than a string and both spellings have to survive intact.
#[pyfunction]
#[pyo3(signature = (adapter, instructions, inputs, outputs, values, demos = None, use_native_function_calling = false))]
fn format_messages(
    adapter: &str,
    instructions: &str,
    inputs: Vec<PyInField>,
    outputs: Vec<PyOutField>,
    values: Vec<(String, String, bool)>,
    demos: Option<Vec<Vec<(String, String)>>>,
    use_native_function_calling: bool,
) -> PyResult<(String, Vec<String>)> {
    let signature = build_signature(instructions, inputs, outputs)?;
    // Python sends each value as JSON text so its structure survives the crossing; the
    // adapter renders it, which is where dspy renders too. The flag beside it is Python's own
    // `isinstance(value, BaseModel)` — the question dspy asks and the one JSON cannot answer,
    // since a dumped model and a mapping written by hand are the same text.
    let pairs: Vec<Input<'_>> = values
        .iter()
        .map(|(name, json, record)| {
            serde_json::from_str(json)
                .map(|value| match record {
                    true => Input::record(name.as_str(), value),
                    false => Input::new(name.as_str(), value),
                })
                .map_err(|error| PyValueError::new_err(format!("input `{name}`: {error}")))
        })
        .collect::<PyResult<_>>()?;
    let adapter = configured_adapter(adapter, use_native_function_calling)?;
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
    let (system, turns) = adapter
        .format(&signature, &demos, &pairs)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    // The whole message, not a role and a content: a turn carrying tool calls or a tool result
    // has keys beside `content`, and the crate is what states their shape.
    let turns = dsrust::lm::api::wire_messages_of(&turns)
        .iter()
        .map(|message| {
            serde_json::to_string(message)
                .map_err(|error| PyValueError::new_err(format!("bad message: {error}")))
        })
        .collect::<PyResult<_>>()?;
    Ok((system, turns))
}

/// Run this crate's own `Predict` for a signature, driven by a Python LM — the module-level
/// crossing. dspy's `test_predict` builds a `DummyLM` and a `dspy.Predict`; the shim points the
/// latter here, so `Predict::forward` renders, calls back into that `DummyLM`, and parses, all in
/// Rust. Returns the prediction's output fields as JSON text and the raw reply the model gave.
#[pyfunction]
#[pyo3(signature = (adapter, instructions, inputs, outputs, values, py_lm, demos = None, n = None))]
fn predict_forward(
    adapter: &str,
    instructions: &str,
    inputs: Vec<PyInField>,
    outputs: Vec<PyOutField>,
    values: Vec<(String, String, bool)>,
    py_lm: Py<PyAny>,
    demos: Option<Vec<Vec<(String, String)>>>,
    n: Option<u32>,
) -> PyResult<(String, String)> {
    let signature = build_signature(instructions, inputs, outputs)?;
    let mut fields = Vec::new();
    for (name, json, _record) in &values {
        let value: Value = serde_json::from_str(json)
            .map_err(|error| PyValueError::new_err(format!("input `{name}`: {error}")))?;
        fields.push((name.clone(), value));
    }
    let example = Example::new(fields);
    let demos: Vec<Example> = demos
        .unwrap_or_default()
        .into_iter()
        .map(|fields| {
            Example::new(
                fields
                    .into_iter()
                    .map(|(name, value)| (name, Value::String(value))),
            )
        })
        .collect();
    // The module-level config spells the completion count `completions`; it becomes the wire
    // request's `n`, which is the kwarg a `DummyLM` reads.
    let mut config = dsrust::lm::LmConfig::default();
    config.completions = n;
    let predict = dsrust::predict::Predict::from_signature(signature)
        .with_config(config)
        .with_demos(demos)
        .with_lm(Arc::new(PyLM { inner: py_lm }));
    // Honour the adapter dspy configured, since a test may set a JSON or XML one. `from_signature`
    // already defaults to `ChatAdapter`, so only the others need setting.
    let predict = match adapter {
        "json" => predict.with_adapter(JsonAdapter::default()),
        "xml" => predict.with_adapter(XmlAdapter),
        "baml" => predict.with_adapter(BamlAdapter),
        _ => predict,
    };
    // One candidate goes through the full `forward` — parse, coercion, the feedback retry, any
    // native or extraction path. Several candidates (dspy's `n`) go through `forward_completions`,
    // which reads every candidate the one response carried. Both return the parsed output fields,
    // always as a JSON array so the shim hands `Prediction.from_completions` a list either way.
    let predictions = if n.unwrap_or(1) > 1 {
        pollster::block_on(predict.forward_completions(example)).map_err(to_value_error)?
    } else {
        vec![
            pollster::block_on(dsrust::module::Module::forward(&predict, example))
                .map_err(to_value_error)?,
        ]
    };
    let completions: Vec<serde_json::Map<String, Value>> = predictions
        .iter()
        .map(|prediction| {
            prediction
                .example
                .fields()
                .map(|(name, value)| (name.to_owned(), value.clone()))
                .collect()
        })
        .collect();
    let output_json = serde_json::to_string(&completions)
        .map_err(|error| PyValueError::new_err(format!("bad prediction: {error}")))?;
    let raw = predictions
        .first()
        .map(|prediction| prediction.raw.clone())
        .unwrap_or_default();
    Ok((output_json, raw))
}

/// A tool backed by a Python callable (a `dspy.Tool`), so this crate's `ReAct` loop can call the
/// same tools upstream's tests hand it. Name, description and the arg schema are read off the
/// object once; `call` invokes it.
pub(crate) struct PyTool {
    name: String,
    description: String,
    args: Value,
    func: Py<PyAny>,
}

impl dsrust::Tool for PyTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn args(&self) -> &Value {
        &self.args
    }

    fn call(&self, args: &Value) -> anyhow::Result<String> {
        match self.call_value(args)? {
            Value::String(text) => Ok(text),
            other => Ok(other.to_string()),
        }
    }

    /// dspy keeps the tool's raw return as the observation, so read the value, not its text: its
    /// JSON form where it has one, and its string form when it does not (a live object such as an
    /// image). The `ReAct` loop renders whichever into the prompt.
    fn call_value(&self, args: &Value) -> anyhow::Result<Value> {
        Python::attach(|py| {
            let args_json = serde_json::to_string(args)?;
            let kwargs = py.import("json")?.call_method1("loads", (args_json,))?;
            let kwargs = kwargs.cast::<pyo3::types::PyDict>().map_err(|error| {
                anyhow::anyhow!("tool `{}` args are not an object: {error}", self.name)
            })?;
            let result = self
                .func
                .bind(py)
                .call((), Some(kwargs))
                .map_err(|error| anyhow::anyhow!("tool `{}` raised: {error}", self.name))?;
            match py.import("json")?.call_method1("dumps", (&result,)) {
                Ok(dumped) => Ok(serde_json::from_str(&dumped.extract::<String>()?)?),
                Err(_) => Ok(Value::String(result.str()?.extract::<String>()?)),
            }
        })
    }
}

/// One `dspy.Tool` as a Rust [`Tool`](dsrust::Tool), reading the name, description and argument
/// schema off the Python object.
pub(crate) fn py_tool(py: Python<'_>, tool: &Py<PyAny>) -> PyResult<PyTool> {
    let bound = tool.bind(py);
    let name: String = bound.getattr("name")?.extract()?;
    // A tool built from a function with no docstring carries `desc = None`.
    let description: String = bound
        .getattr("desc")?
        .extract::<Option<String>>()?
        .unwrap_or_default();
    let args_json: String = py
        .import("json")?
        .call_method1("dumps", (bound.getattr("args")?,))?
        .extract()?;
    let args: Value = serde_json::from_str(&args_json)
        .map_err(|error| PyValueError::new_err(format!("tool `{name}` args: {error}")))?;
    Ok(PyTool {
        name,
        description,
        args,
        func: tool.clone_ref(py),
    })
}

/// Every tool in the list, in order.
pub(crate) fn py_tools(
    py: Python<'_>,
    tools: &[Py<PyAny>],
) -> PyResult<Vec<Arc<dyn dsrust::Tool>>> {
    tools
        .iter()
        .map(|tool| py_tool(py, tool).map(|built| Arc::new(built) as Arc<dyn dsrust::Tool>))
        .collect()
}

/// Run this crate's own `ReAct` for a signature and a set of Python tools, driven by a Python LM —
/// the module-level crossing for the agent loop. dspy's `test_react` builds `dspy.ReAct(sig,
/// tools=[…])` with a `DummyLM`; the shim points its `forward` here, so the loop, the tool calls
/// and the extraction all run in Rust.
#[pyfunction]
#[pyo3(signature = (instructions, inputs, outputs, values, py_lm, tools, max_iters = None))]
fn react_forward(
    py: Python<'_>,
    instructions: &str,
    inputs: Vec<PyInField>,
    outputs: Vec<PyOutField>,
    values: Vec<(String, String, bool)>,
    py_lm: Py<PyAny>,
    tools: Vec<Py<PyAny>>,
    max_iters: Option<usize>,
) -> PyResult<(String, String)> {
    let signature = build_signature(instructions, inputs, outputs)?;
    let mut rust_tools: Vec<Box<dyn dsrust::Tool>> = Vec::new();
    for tool in &tools {
        let built = py_tool(py, tool)?;
        // dspy's tool dict carries its own `finish`; `ReAct::new` adds one, so skip the duplicate.
        if built.name != "finish" {
            rust_tools.push(Box::new(built));
        }
    }

    let mut react =
        dsrust::ReAct::new(signature, rust_tools).with_lm(Arc::new(PyLM { inner: py_lm }));
    if let Some(max_iters) = max_iters {
        react = react.with_max_iters(max_iters);
    }

    let mut fields = Vec::new();
    for (name, json, _record) in &values {
        let value: Value = serde_json::from_str(json)
            .map_err(|error| PyValueError::new_err(format!("input `{name}`: {error}")))?;
        fields.push((name.clone(), value));
    }
    let example = Example::new(fields);
    let prediction = pollster::block_on(dsrust::module::Module::forward(&react, example))
        .map_err(to_value_error)?;
    let output: serde_json::Map<String, Value> = prediction
        .example
        .fields()
        .map(|(name, value)| (name.to_owned(), value.clone()))
        .collect();
    let output_json = serde_json::to_string(&output)
        .map_err(|error| PyValueError::new_err(format!("bad prediction: {error}")))?;
    Ok((output_json, prediction.raw))
}

/// The system message the named adapter states for this signature.
///
/// dspy exposes this separately from a whole exchange, and a caller reading it should read the
/// crate's, not a reimplementation of it.
#[pyfunction]
#[pyo3(signature = (adapter, instructions, inputs, outputs))]
fn format_system_message(
    adapter: &str,
    instructions: &str,
    inputs: Vec<PyInField>,
    outputs: Vec<PyOutField>,
) -> PyResult<String> {
    let signature = build_signature(instructions, inputs, outputs)?;
    let adapter = adapter_named(adapter)?;
    adapter
        .system_message(&signature)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// The field-structure section of the BAML adapter's system message.
///
/// dspy declares `format_field_structure` on every adapter, but only this one states anything
/// there a caller reads on its own — the compact notation for each output's type — and only
/// this one can refuse a signature outright, which is what its recursion test asserts. The
/// others reach a caller through the whole system message.
#[pyfunction]
#[pyo3(signature = (instructions, inputs, outputs))]
fn baml_field_structure(
    instructions: &str,
    inputs: Vec<PyInField>,
    outputs: Vec<PyOutField>,
) -> PyResult<String> {
    let signature = build_signature(instructions, inputs, outputs)?;
    BamlAdapter
        .field_structure(&signature)
        .map_err(|error| PyValueError::new_err(format!("{error:#}")))
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
    let adapter = adapter_named(adapter)?;
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

/// dspy `_provider_tool_call_to_tool_call_dict`: one tool call a provider sent, read back into the
/// `{id, name, args}` this crate's `ToolCalls` is built from.
///
/// Reading an arbitrary provider object is reflection and stays on the Python side; deciding what
/// the written call *means* — which spelling holds the name, whether the arguments repair — is the
/// crate's, so the plain mapping crosses and this answers.
#[pyfunction]
#[pyo3(signature = (tool_call))]
fn normalize_tool_call(tool_call: &str) -> PyResult<String> {
    let written: Value = serde_json::from_str(tool_call)
        .map_err(|error| PyValueError::new_err(format!("bad tool call: {error}")))?;
    let calls = dsrust::adapter::ToolCalls::from_dict_list(std::slice::from_ref(&written))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let call = calls
        .tool_calls
        .first()
        .ok_or_else(|| PyValueError::new_err("no tool call"))?;
    serde_json::to_string(&serde_json::json!({
        "id": call.id,
        "name": call.name,
        "args": Value::Object(call.args.clone()),
    }))
    .map_err(|error| PyValueError::new_err(format!("bad tool call: {error}")))
}

/// The settings the crate's fallback carries, or `None` where it re-asks through nothing.
///
/// dspy's ChatAdapter builds its fallback as `JSONAdapter(use_native_function_calling=...,
/// parallel_tool_calls=...)`; both the decision and what it propagates are the crate's, so the
/// shim reads them back rather than reconstructing the rule on the Python side.
#[pyfunction]
#[pyo3(signature = (adapter, use_json_adapter_fallback, use_native_function_calling, parallel_tool_calls))]
fn json_fallback_settings(
    adapter: &str,
    use_json_adapter_fallback: bool,
    use_native_function_calling: bool,
    parallel_tool_calls: Option<bool>,
) -> PyResult<Option<(bool, Option<bool>)>> {
    if adapter == "chat" {
        let chat = ChatAdapter {
            use_json_adapter_fallback,
            use_native_function_calling,
            parallel_tool_calls,
        };
        return Ok(chat.json_fallback_adapter().map(|fallback| {
            (
                fallback.use_native_function_calling,
                fallback.parallel_tool_calls,
            )
        }));
    }
    // Every other wire format states only whether it re-asks; none carries these settings.
    Ok(adapter_named(adapter)?
        .json_fallback()
        .is_some()
        .then_some((false, None)))
}

#[pymodule]
fn dsrs_bridge(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(format_messages, module)?)?;
    module.add_function(wrap_pyfunction!(predict_forward, module)?)?;
    module.add_function(wrap_pyfunction!(react_forward, module)?)?;
    module.add_function(wrap_pyfunction!(code_modules::rlm_forward, module)?)?;
    module.add_function(wrap_pyfunction!(
        code_modules::program_of_thought_forward,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(code_modules::code_act_forward, module)?)?;
    module.add_function(wrap_pyfunction!(code_modules::answer_exact_match, module)?)?;
    module.add_function(wrap_pyfunction!(code_modules::normalize_message, module)?)?;
    module.add_function(wrap_pyfunction!(code_modules::cache_key, module)?)?;
    module.add_function(wrap_pyfunction!(
        code_modules::is_openai_reasoning_model,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(code_modules::responses_body, module)?)?;
    module.add_function(wrap_pyfunction!(code_modules::merge_usage, module)?)?;
    module.add_function(wrap_pyfunction!(format_system_message, module)?)?;
    module.add_function(wrap_pyfunction!(baml_field_structure, module)?)?;
    module.add_function(wrap_pyfunction!(parse_reply, module)?)?;
    module.add_function(wrap_pyfunction!(json_fallback_settings, module)?)?;
    module.add_function(wrap_pyfunction!(normalize_tool_call, module)?)?;
    module.add_function(wrap_pyfunction!(extractor_instructions, module)?)?;
    module.add_function(wrap_pyfunction!(field_description, module)?)?;
    module.add_function(wrap_pyfunction!(infer_prefix, module)?)?;
    module.add_function(wrap_pyfunction!(split_example, module)?)?;
    module.add_function(wrap_pyfunction!(majority_index, module)?)?;
    Ok(())
}
