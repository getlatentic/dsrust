//! The crate's Deno/Pyodide sandbox, exposed so dspy's own interpreter tests can drive it.
//!
//! `tests/primitives/test_python_interpreter.py` builds a `PythonInterpreter` and calls `execute`
//! on it. `RustPythonInterpreter` in the shim replaces only that method, so the tests exercise
//! `DenoInterpreter` while every other part of the class stays dspy's.
//!
//! Values cross as JSON text, the way everything crosses this bridge, rather than as mapped
//! objects — one shape to keep in step instead of two.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::{Map, Value};

use dsrust::interpreter::{CodeInterpreter, DenoInterpreter, Executed, Permissions};

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
    #[pyo3(signature = (env=None, read=None, write=None, network=None))]
    fn new(
        env: Option<Vec<String>>,
        read: Option<Vec<String>>,
        write: Option<Vec<String>>,
        network: Option<Vec<String>>,
    ) -> Self {
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
        Self {
            inner: DenoInterpreter::with_permissions(permissions),
        }
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
