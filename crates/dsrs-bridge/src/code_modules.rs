//! The three code-writing modules, driven by dspy's own tests: `ProgramOfThought`, `CodeAct`, `Rlm`.
//!
//! Each has a loop that a golden cannot reach. A fixture records bytes; which turn ends a run, what
//! lands in the trajectory, whether a failed snippet is rewritten or given up on — none of that is
//! bytes. Upstream's tests assert exactly that layer, so they run against these.
//!
//! All three drive a Python interpreter through [`PyInterpreter`]: dspy's `MockInterpreter` for the
//! RLM cases, and its real Deno sandbox for the `@pytest.mark.deno` ones, which means the code the
//! model wrote is genuinely executed. The model side is the existing [`PyLM`](crate::PyLM) — for
//! RLM the two predictors arrive already dressed as LMs Python-side, since upstream mocks at the
//! predictor level rather than the LM level.

use std::sync::Arc;

use dsrust::interpreter::{CodeInterpreter, Executed};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::Value;

use crate::{PyInField, PyLM, PyOutField, build_signature, to_value_error};

/// An interpreter that is a Python object — dspy's `MockInterpreter`, or its real Deno sandbox.
struct PyInterpreter {
    inner: Py<PyAny>,
}

impl PyInterpreter {
    /// What one `execute` produced, in the crate's vocabulary.
    ///
    /// dspy's interpreter answers four ways and the loop reads each differently: a `FinalOutput`
    /// ends the episode, a string or a list of lines is printed output, and `None` is silence. A
    /// raised `CodeInterpreterError` is the code's own failure, which the loop feeds back rather
    /// than propagates — so it crosses as an error and the Rust side decides that.
    fn executed(py: Python<'_>, result: &Bound<'_, PyAny>) -> anyhow::Result<Executed> {
        let final_output = py
            .import("dspy.primitives.code_interpreter")?
            .getattr("FinalOutput")?;
        if result.is_instance(&final_output)? {
            return Ok(Executed::Submitted(jsonable(
                py,
                &result.getattr("output")?,
            )?));
        }
        Ok(Executed::Printed(jsonable(py, result)?))
    }
}

/// A Python value as JSON, falling back to its `str()` where it has no JSON form — the same rule
/// [`PyTool`](crate::PyTool) reads a tool's return by.
fn jsonable(py: Python<'_>, value: &Bound<'_, PyAny>) -> anyhow::Result<Value> {
    match py.import("json")?.call_method1("dumps", (value,)) {
        Ok(dumped) => Ok(serde_json::from_str(&dumped.extract::<String>()?)?),
        Err(_) => Ok(Value::String(value.str()?.extract::<String>()?)),
    }
}

impl CodeInterpreter for PyInterpreter {
    fn execute(
        &self,
        code: &str,
        variables: &serde_json::Map<String, Value>,
    ) -> anyhow::Result<Executed> {
        Python::attach(|py| {
            let bound = py
                .import("json")?
                .call_method1("loads", (serde_json::to_string(variables)?,))?;
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("variables", bound)?;
            let result = self
                .inner
                .bind(py)
                .call_method("execute", (code,), Some(&kwargs))
                // dspy raises the message the loop shows the model, so it crosses as written.
                .map_err(|error| anyhow::anyhow!("{}", raised_message(py, &error)))?;
            PyInterpreter::executed(py, &result)
        })
    }

    fn start(&self) -> anyhow::Result<()> {
        Python::attach(|py| {
            self.inner.bind(py).call_method0("start")?;
            Ok(())
        })
    }

    fn shutdown(&self) {
        Python::attach(|py| {
            let _ = self.inner.bind(py).call_method0("shutdown");
        });
    }
}

/// What the exception says, without the traceback and class prefix pytest would print — the loop
/// puts this in a prompt, so it has to read the way upstream's does.
fn raised_message(py: Python<'_>, error: &PyErr) -> String {
    error
        .value(py)
        .str()
        .map(|text| text.to_string_lossy().into_owned())
        .unwrap_or_else(|_| error.to_string())
}

/// Run this crate's `Rlm` over a Python interpreter and a Python pair of predictors.
///
/// The two predictors are separate because upstream's tests replace them separately: several
/// script only `generate_action` and let the run end on a submission, and the max-iterations cases
/// script both.
#[pyfunction]
#[pyo3(signature = (
    instructions, inputs, outputs, values, interpreter, action_lm, extract_lm,
    max_iterations = None, max_llm_calls = None,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn rlm_forward(
    py: Python<'_>,
    instructions: &str,
    inputs: Vec<PyInField>,
    outputs: Vec<PyOutField>,
    values: Vec<(String, String)>,
    interpreter: Py<PyAny>,
    action_lm: Py<PyAny>,
    extract_lm: Py<PyAny>,
    max_iterations: Option<usize>,
    max_llm_calls: Option<usize>,
) -> PyResult<String> {
    let signature = build_signature(instructions, inputs, outputs)?;
    let mut rlm =
        dsrust::Rlm::with_interpreter(signature, Arc::new(PyInterpreter { inner: interpreter }));
    if let Some(max_llm_calls) = max_llm_calls {
        rlm = rlm.with_max_llm_calls(max_llm_calls);
    }
    if let Some(max_iterations) = max_iterations {
        rlm = rlm.with_max_iterations(max_iterations);
    }
    rlm = rlm
        .with_action_lm(Arc::new(PyLM { inner: action_lm }))
        .with_extract_lm(Arc::new(PyLM { inner: extract_lm }));

    answered(py, rlm, values)
}

/// Run this crate's `ProgramOfThought` over a Python interpreter, driven by a Python LM.
///
/// Upstream's cases are `@pytest.mark.deno`: a real sandbox executes what the model wrote, so what
/// is asserted is the whole loop — the code parsed, ran, and either answered or was rewritten.
#[pyfunction]
#[pyo3(signature = (instructions, inputs, outputs, values, interpreter, py_lm, max_iters = None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn program_of_thought_forward(
    py: Python<'_>,
    instructions: &str,
    inputs: Vec<PyInField>,
    outputs: Vec<PyOutField>,
    values: Vec<(String, String)>,
    interpreter: Py<PyAny>,
    py_lm: Py<PyAny>,
    max_iters: Option<usize>,
) -> PyResult<String> {
    let signature = build_signature(instructions, inputs, outputs)?;
    let mut pot = dsrust::ProgramOfThought::with_interpreter(
        signature,
        Arc::new(PyInterpreter { inner: interpreter }),
    );
    if let Some(max_iters) = max_iters {
        pot = pot.with_max_iters(max_iters);
    }
    answered(py, pot.with_lm(Arc::new(PyLM { inner: py_lm })), values)
}

/// Run this crate's `CodeAct` over a Python interpreter and a set of Python tools.
///
/// The tools reach the sandbox the way dspy puts them there — their *source*, executed before the
/// first turn — because that is the setup upstream's `forward` does and it is not the loop under
/// test. The crate's `define_tools` seam is what a Rust caller uses instead, and it is a no-op on
/// an interpreter that already has them.
#[pyfunction]
#[pyo3(signature = (instructions, inputs, outputs, values, interpreter, py_lm, tools, max_iters = None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn code_act_forward(
    py: Python<'_>,
    instructions: &str,
    inputs: Vec<PyInField>,
    outputs: Vec<PyOutField>,
    values: Vec<(String, String)>,
    interpreter: Py<PyAny>,
    py_lm: Py<PyAny>,
    tools: Vec<Py<PyAny>>,
    max_iters: Option<usize>,
) -> PyResult<String> {
    let signature = build_signature(instructions, inputs, outputs)?;
    let rust_tools = crate::py_tools(py, &tools)?;
    let mut act = dsrust::CodeAct::with_interpreter(
        signature,
        rust_tools,
        Arc::new(PyInterpreter { inner: interpreter }),
    );
    if let Some(max_iters) = max_iters {
        act = act.with_max_iters(max_iters);
    }
    answered(py, act.with_lm(Arc::new(PyLM { inner: py_lm })), values)
}

/// Run a module over the named input values and hand its prediction back as JSON.
fn answered<M: dsrust::module::Module>(
    py: Python<'_>,
    module: M,
    values: Vec<(String, String)>,
) -> PyResult<String> {
    let mut fields = Vec::new();
    for (name, json) in &values {
        let value: Value = serde_json::from_str(json)
            .map_err(|error| PyValueError::new_err(format!("input `{name}`: {error}")))?;
        fields.push((name.clone(), value));
    }
    // Released while the loop runs, so the interpreter and the LM can be called back into Python.
    let prediction = py
        .detach(|| {
            pollster::block_on(dsrust::module::Module::forward(
                &module,
                dsrust::Example::new(fields),
            ))
        })
        .map_err(to_value_error)?;
    let output: serde_json::Map<String, Value> = prediction
        .example
        .fields()
        .map(|(name, value)| (name.to_owned(), value.clone()))
        .collect();
    serde_json::to_string(&output)
        .map_err(|error| PyValueError::new_err(format!("bad prediction: {error}")))
}

/// dspy `answer_exact_match`, answered by the crate.
///
/// The example's gold answer and the prediction's are read Python-side — reflection over dspy's
/// own objects is Python's job — and what a *match* means is decided here: normalisation, the
/// article and punctuation rules, and the best score across several gold answers.
#[pyfunction]
pub(crate) fn answer_exact_match(answers: Vec<String>, answered: &str) -> f64 {
    match dsrust::evaluate::metrics::em(answered, &answers) {
        true => 1.0,
        false => 0.0,
    }
}

/// One message read by the crate's `LmMessage` and written back out.
///
/// dspy's `LMMessage` accepts either the typed shape or the one a provider writes, and normalises
/// the second into the first. The crate does the same, so a message crosses by round-trip: what
/// comes back is what *our* normaliser made of it, and a test asserting on the parts is asserting
/// on that.
#[pyfunction]
pub(crate) fn normalize_message(written: &str) -> PyResult<String> {
    let message: dsrust::lm::api::LmMessage =
        serde_json::from_str(written).map_err(|error| PyValueError::new_err(format!("{error}")))?;
    serde_json::to_string(&message).map_err(|error| PyValueError::new_err(format!("{error}")))
}

/// dspy `Cache.cache_key`, computed by the crate.
///
/// Upstream hashes its whole kwargs dict with sorted keys; so does this. What crosses is the rule
/// that decides whether two calls are the same call — and therefore whether one is answered with
/// the other's reply, or paid for again.
#[pyfunction]
#[pyo3(signature = (request, ignored = None))]
pub(crate) fn cache_key(request: &str, ignored: Option<Vec<String>>) -> PyResult<String> {
    let value: Value =
        serde_json::from_str(request).map_err(|error| PyValueError::new_err(format!("{error}")))?;
    Ok(dsrust::lm::cache::key_ignoring(
        &value,
        &ignored.unwrap_or_default(),
    ))
}

/// dspy `_is_openai_reasoning_model`, decided by the crate.
///
/// Which family a model belongs to is the whole of the decision behind two things a request
/// carries: whether the generation cap travels as `max_tokens` or `max_completion_tokens`, and
/// whether `temperature=1.0` and a 16k floor are required.
#[pyfunction]
pub(crate) fn is_openai_reasoning_model(model: &str) -> bool {
    use dsrust::lm::{TokenLimitField, TokenLimitRule};
    TokenLimitRule::ByOpenAiModelFamily.field_for(model) == TokenLimitField::MaxCompletionTokens
}

/// The Responses-API request body the crate builds for a chat-shaped request.
///
/// dspy has two routes to this body: a typed one over `LMRequest`, which the crate implements, and
/// `_convert_chat_request_to_responses_request`, which rewrites a chat dict. The crate has no
/// chat-dict route — its `LM` always holds a typed request — so the crossing is the *body*: the
/// same conversation, taken the crate's way, must reach the same wire.
#[pyfunction]
pub(crate) fn responses_body(request: &str) -> PyResult<String> {
    use dsrust::lm::api::{LmConfig, LmMessage, LmRequest};

    let written: Value =
        serde_json::from_str(request).map_err(|error| PyValueError::new_err(format!("{error}")))?;
    let model = written
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let messages: Vec<LmMessage> = written
        .get("messages")
        .cloned()
        .map(serde_json::from_value)
        .transpose()
        .map_err(|error| PyValueError::new_err(format!("messages: {error}")))?
        .unwrap_or_default();

    let mut config = LmConfig::default();
    if let Some(schema) = written.get("response_format") {
        config.response_format = Some(schema.clone());
    }
    let call = LmRequest::new(model, messages).configured(config);
    let body = dsrust::lm::openai::responses::request(model, &call, dsrust::lm::JsonFormat::Schema);
    serde_json::to_string(&body).map_err(|error| PyValueError::new_err(format!("{error}")))
}

/// dspy `UsageTracker._merge_usage_entries`, computed by the crate.
///
/// What two calls' counters come to together. A nested breakdown merges into itself and a number
/// adds, so a program's total is the sum of what it spent rather than the last call's.
#[pyfunction]
pub(crate) fn merge_usage(left: &str, right: &str) -> PyResult<String> {
    use dsrust::lm::LmUsage;
    let parse = |written: &str| -> PyResult<Option<LmUsage>> {
        let value: Value = serde_json::from_str(written)
            .map_err(|error| PyValueError::new_err(format!("{error}")))?;
        match value.as_object().is_some_and(serde_json::Map::is_empty) || value.is_null() {
            true => Ok(None),
            false => serde_json::from_value(value)
                .map(Some)
                .map_err(|error| PyValueError::new_err(format!("{error}"))),
        }
    };
    let merged = LmUsage::merge(parse(left)?, parse(right)?);
    serde_json::to_string(&merged.unwrap_or_default())
        .map_err(|error| PyValueError::new_err(format!("{error}")))
}
