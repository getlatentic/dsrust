//! The RLM crossing: this crate's `Rlm` loop, driven by dspy's own test doubles.
//!
//! `test_rlm.py` mocks the two predictors and the interpreter and then asserts what the loop did —
//! which turn ended it, what landed in the trajectory, when the extract fallback fires, and what a
//! malformed submission is answered with. That is exactly the layer no golden reaches: the prompts
//! are bytes a fixture can record, and the control flow between them is not.
//!
//! So both doubles cross. The interpreter arrives as a [`PyInterpreter`] calling the Python mock's
//! `execute`, and each predictor arrives already wrapped Python-side in an LM-shaped object, so the
//! existing [`PyLM`](crate::PyLM) carries it and the crate's own `Predict` is what asks.

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
            return Ok(Executed::Submitted(jsonable(py, &result.getattr("output")?)?));
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
    fn execute(&self, code: &str) -> anyhow::Result<Executed> {
        Python::attach(|py| {
            let result = self
                .inner
                .bind(py)
                .call_method1("execute", (code,))
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
    let mut rlm = dsrust::Rlm::new(signature, Arc::new(PyInterpreter { inner: interpreter }));
    if let Some(max_llm_calls) = max_llm_calls {
        rlm = rlm.with_max_llm_calls(max_llm_calls);
    }
    if let Some(max_iterations) = max_iterations {
        rlm = rlm.with_max_iterations(max_iterations);
    }
    rlm = rlm
        .with_action_lm(Arc::new(PyLM { inner: action_lm }))
        .with_extract_lm(Arc::new(PyLM { inner: extract_lm }));

    let mut fields = Vec::new();
    for (name, json) in &values {
        let value: Value = serde_json::from_str(json)
            .map_err(|error| PyValueError::new_err(format!("input `{name}`: {error}")))?;
        fields.push((name.clone(), value));
    }
    let prediction = py
        .detach(|| pollster::block_on(dsrust::module::Module::forward(&rlm, dsrust::Example::new(fields))))
        .map_err(to_value_error)?;
    let output: serde_json::Map<String, Value> = prediction
        .example
        .fields()
        .map(|(name, value)| (name.to_owned(), value.clone()))
        .collect();
    serde_json::to_string(&output)
        .map_err(|error| PyValueError::new_err(format!("bad prediction: {error}")))
}
