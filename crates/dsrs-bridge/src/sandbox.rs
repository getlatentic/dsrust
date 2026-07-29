//! The crate's Deno/Pyodide sandbox, exposed so dspy's own interpreter tests can drive it.
//!
//! `tests/primitives/test_python_interpreter.py` builds a `PythonInterpreter` and calls `execute`
//! on it. `RustPythonInterpreter` in the shim replaces only that method, so the tests exercise
//! `DenoInterpreter` while every other part of the class stays dspy's.
//!
//! Values cross as JSON text, the way everything crosses this bridge, rather than as mapped
//! objects — one shape to keep in step instead of two.

use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::{Map, Value};

use dsrust::interpreter::{CodeInterpreter, DenoInterpreter, Executed, OutputField, Permissions};
use dsrust::react::Tool;

/// One of upstream's tools: a plain Python callable, not a `dspy.Tool`.
///
/// Its arguments were read by `inspect.signature` on the Python side — reflection is what Python is
/// for here — and arrive as the schema map `Tool::args` answers with. What the sandbox is told
/// about them is the crate's decision, in `interpreter::deno::register`.
struct PyCallable {
    name: String,
    args: Value,
    call: Py<PyAny>,
}

impl Tool for PyCallable {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        ""
    }

    fn args(&self) -> &Value {
        &self.args
    }

    fn call(&self, args: &Value) -> anyhow::Result<String> {
        Ok(self
            .call_value(args)?
            .as_str()
            .unwrap_or_default()
            .to_owned())
    }

    /// The sandbox calls with keyword arguments, and whatever comes back crosses as JSON so the
    /// crate decides its shape rather than pyo3 guessing at it.
    fn call_value(&self, args: &Value) -> anyhow::Result<Value> {
        Python::attach(|py| {
            let json = py.import("json")?;
            let kwargs = json.call_method1("loads", (args.to_string(),))?;
            let kwargs = kwargs.cast::<pyo3::types::PyDict>().map_err(|_| {
                PyValueError::new_err(format!("tool `{}` was called with {args}", self.name))
            })?;
            let answered = self.call.bind(py).call((), Some(kwargs))?;
            // `default=str` so a value with no JSON form crosses as its `str`, which is what a
            // sandbox tool's observation is read as anyway.
            let options = pyo3::types::PyDict::new(py);
            options.set_item("default", py.get_type::<pyo3::types::PyString>())?;
            let crossed: String = json
                .call_method("dumps", (answered,), Some(&options))?
                .extract()?;
            Ok::<Value, PyErr>(serde_json::from_str(&crossed).unwrap_or(Value::Null))
        })
        .map_err(|error: PyErr| anyhow::anyhow!("{error}"))
    }
}

/// The crate's sandbox, held open across calls so a name bound by one `execute` survives to the
/// next — which is the behaviour half of upstream's tests are checking.
#[pyclass]
pub(crate) struct RustSandbox {
    inner: DenoInterpreter,
}

#[pymethods]
impl RustSandbox {
    /// Built from the same four grants `PythonInterpreter.__init__` takes.
    #[new]
    #[pyo3(signature = (env=None, read=None, write=None, network=None, tools=None, outputs=None))]
    fn new(
        env: Option<Vec<String>>,
        read: Option<Vec<String>>,
        write: Option<Vec<String>>,
        network: Option<Vec<String>>,
        tools: Option<Vec<(String, String, Py<PyAny>)>>,
        outputs: Option<String>,
    ) -> PyResult<Self> {
        let permissions = Permissions {
            env: env.unwrap_or_default(),
            read: read
                .unwrap_or_default()
                .into_iter()
                .map(Into::into)
                .collect(),
            write: write
                .unwrap_or_default()
                .into_iter()
                .map(Into::into)
                .collect(),
            network: network.unwrap_or_default(),
        };
        // dspy's `output_fields` is a list of `{"name": …, "type": …}`, so it crosses as JSON
        // rather than as names — the type is what makes the generated `SUBMIT` typed.
        let described: Vec<Value> = match outputs {
            None => Vec::new(),
            Some(json) => serde_json::from_str(&json)
                .map_err(|error| PyValueError::new_err(format!("output_fields: {error}")))?,
        };
        let sandbox = DenoInterpreter::with_permissions(permissions).with_output_fields(
            described.iter().map(|field| OutputField {
                name: field
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
                python_type: field.get("type").and_then(Value::as_str).map(str::to_owned),
            }),
        );
        let carried: Vec<Arc<dyn Tool>> = tools
            .unwrap_or_default()
            .into_iter()
            .map(|(name, args, call)| {
                let args = serde_json::from_str(&args).map_err(|error| {
                    PyValueError::new_err(format!("tool `{name}` args: {error}"))
                })?;
                Ok(Arc::new(PyCallable { name, args, call }) as Arc<dyn Tool>)
            })
            .collect::<PyResult<_>>()?;
        sandbox
            .define_tools(&carried)
            .map_err(|e| PyValueError::new_err(format!("{e:#}")))?;
        Ok(Self { inner: sandbox })
    }

    /// Run the code, answering with `(kind, json)` — `kind` says whether the code called `SUBMIT`,
    /// which upstream distinguishes by returning a `FinalOutput` rather than a bare value.
    ///
    /// A failure crosses as `ValueError` carrying the crate's message, which is dspy's own wording;
    /// the shim reads that text to pick between `SyntaxError` and `CodeInterpreterError`, exactly
    /// as the other code modules do, because an `anyhow::Error` has no class to cross as.
    fn execute(&self, code: &str, variables: &str) -> PyResult<(String, String)> {
        let given: Map<String, Value> = match variables.is_empty() {
            true => Map::new(),
            false => serde_json::from_str(variables)
                .map_err(|error| PyValueError::new_err(format!("variables: {error}")))?,
        };
        let ran = self
            .inner
            .execute(code, &given)
            .map_err(|error| PyValueError::new_err(format!("{error:#}")))?;
        let kind = match ran {
            Executed::Submitted(_) => "submitted",
            Executed::Printed(_) => "printed",
        };
        Ok((kind.to_owned(), ran.value().to_string()))
    }

    fn shutdown(&self) {
        self.inner.shutdown();
    }
}
