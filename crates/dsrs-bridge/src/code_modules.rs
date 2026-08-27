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

use dsrust::interpreter::InterpreterFailure;
use dsrust::interpreter::{CodeInterpreter, Executed};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::Value;

use crate::{PyInField, PyLM, PyOutField, build_signature, to_value_error};

/// An interpreter that is a Python object — dspy's `MockInterpreter`, or its real Deno sandbox.
struct PyInterpreter {
    inner: Py<PyAny>,
    /// The last exception this interpreter raised that the loop did *not* feed back.
    ///
    /// A terminal failure propagates, and upstream's callers catch it by its own class — a bare
    /// `ValueError` stays a `ValueError`. Only the message survives an `anyhow::Error`, so the
    /// exception itself is parked here and re-raised at the boundary. One slot rather than a
    /// channel: the loop stops at the first one, so there is never a second to hold.
    terminal: std::sync::Mutex<Option<PyErr>>,
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

    /// The exception a terminal failure came from, when it came from Python at all.
    ///
    /// Taken rather than read: the loop stops at the first terminal failure, so the one parked here
    /// belongs to the pass now unwinding and there is never a second to confuse it with.
    fn parked(&self) -> Option<PyErr> {
        self.terminal.lock().expect("the terminal slot").take()
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
                // dspy raises the message the loop shows the model, so it crosses as written —
                // and *which* failure it was crosses with it. dspy 3.3.0 splits
                // `CodeExecutionError` (the code's, recoverable, fed back to the model) from a
                // bare `CodeInterpreterError` (the interpreter's, terminal). Flattening both into
                // one anyhow error made a protocol failure look like something worth rewriting
                // the code over, and the loop swallowed it instead of propagating.
                //
                // The other direction was typed first; this is its mirror.
                .map_err(|error| {
                    let failure = interpreter_failure(py, &error);
                    if matches!(
                        failure.downcast_ref::<InterpreterFailure>(),
                        Some(InterpreterFailure::Session(_))
                    ) {
                        *self.terminal.lock().expect("the terminal slot") = Some(error);
                    }
                    failure
                })?;
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
    max_iters = None, max_llm_calls = None,
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
    max_iters: Option<usize>,
    max_llm_calls: Option<usize>,
) -> PyResult<String> {
    let signature = build_signature(instructions, inputs, outputs)?;
    let owner = Arc::new(PyInterpreter {
        inner: interpreter,
        terminal: Default::default(),
    });
    let repl: Arc<dyn dsrust::interpreter::CodeInterpreter> = owner.clone();
    let mut rlm = dsrust::Rlm::interpreter_factory(signature, refusing_factory());
    if let Some(max_llm_calls) = max_llm_calls {
        rlm = rlm.max_llm_calls(max_llm_calls);
    }
    if let Some(max_iters) = max_iters {
        rlm = rlm.max_iters(max_iters);
    }
    rlm = rlm
        .action_lm(Arc::new(PyLM { inner: action_lm }))
        .extract_lm(Arc::new(PyLM { inner: extract_lm }));

    answered_in(py, owner, repl, values, |repl, inputs| async move {
        rlm.ask_in(repl, inputs).await
    })
}

/// Run this crate's `ProgramOfThought` over a Python interpreter, driven by a Python LM.
///
/// Upstream's cases are `@pytest.mark.deno`: a real sandbox executes what the model wrote, so what
/// is asserted is the whole loop — the code parsed, ran, and either answered or was rewritten.
#[pyfunction]
#[pyo3(signature = (instructions, inputs, outputs, values, interpreter, generate_lm, regenerate_lm, answer_lm, max_iters = None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn program_of_thought_forward(
    py: Python<'_>,
    instructions: &str,
    inputs: Vec<PyInField>,
    outputs: Vec<PyOutField>,
    values: Vec<(String, String)>,
    interpreter: Py<PyAny>,
    generate_lm: Py<PyAny>,
    regenerate_lm: Py<PyAny>,
    answer_lm: Py<PyAny>,
    max_iters: Option<usize>,
) -> PyResult<String> {
    let signature = build_signature(instructions, inputs, outputs)?;
    let owner = Arc::new(PyInterpreter {
        inner: interpreter,
        terminal: Default::default(),
    });
    let repl: Arc<dyn dsrust::interpreter::CodeInterpreter> = owner.clone();
    let mut pot = dsrust::ProgramOfThought::interpreter_factory(signature, refusing_factory());
    if let Some(max_iters) = max_iters {
        pot = pot.max_iters(max_iters);
    }
    // One handle per stage, as `rlm_forward` carries `action_lm` beside `extract_lm`: upstream's
    // tests stub the module's predictors one at a time, and a single process LM cannot route to
    // two different stubs.
    let pot = pot
        .generate_lm(Arc::new(PyLM { inner: generate_lm }))
        .regenerate_lm(Arc::new(PyLM {
            inner: regenerate_lm,
        }))
        .answer_lm(Arc::new(PyLM { inner: answer_lm }));
    answered_in(py, owner, repl, values, |repl, inputs| async move {
        pot.ask_in(repl, inputs).await
    })
}

/// Run this crate's `CodeAct` over a Python interpreter and a set of Python tools.
///
/// The tools reach the sandbox the way dspy puts them there — their *source*, executed before the
/// first turn — because that is the setup upstream's `forward` does and it is not the loop under
/// test. The crate's `define_tools` seam is what a Rust caller uses instead, and it is a no-op on
/// an interpreter that already has them.
#[pyfunction]
#[pyo3(signature = (instructions, inputs, outputs, values, interpreter, codeact_lm, extract_lm, tools, max_iters = None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn code_act_forward(
    py: Python<'_>,
    instructions: &str,
    inputs: Vec<PyInField>,
    outputs: Vec<PyOutField>,
    values: Vec<(String, String)>,
    interpreter: Py<PyAny>,
    codeact_lm: Py<PyAny>,
    extract_lm: Py<PyAny>,
    tools: Vec<Py<PyAny>>,
    max_iters: Option<usize>,
) -> PyResult<String> {
    let signature = build_signature(instructions, inputs, outputs)?;
    let rust_tools = crate::py_tools(py, &tools)?;
    let owner = Arc::new(PyInterpreter {
        inner: interpreter,
        terminal: Default::default(),
    });
    let repl: Arc<dyn dsrust::interpreter::CodeInterpreter> = owner.clone();
    let mut act = dsrust::CodeAct::interpreter_factory(signature, rust_tools, refusing_factory());
    if let Some(max_iters) = max_iters {
        act = act.max_iters(max_iters);
    }
    // See `program_of_thought_forward` for why each stage carries its own handle.
    let act = act
        .codeact_lm(Arc::new(PyLM { inner: codeact_lm }))
        .extract_lm(Arc::new(PyLM { inner: extract_lm }));
    answered_in(py, owner, repl, values, |repl, inputs| async move {
        act.ask_in(repl, inputs).await
    })
}

/// Run a module over the named input values, in the caller's interpreter, and hand its
/// prediction back as JSON.
///
/// Through each module's `ask_in` — the borrowed arm — rather than `forward`, which would build
/// from the factory and own what it built. See [`refusing_factory`].
fn answered_in<Fut>(
    py: Python<'_>,
    owner: Arc<PyInterpreter>,
    repl: Arc<dyn dsrust::interpreter::CodeInterpreter>,
    values: Vec<(String, String)>,
    ask: impl FnOnce(Arc<dyn dsrust::interpreter::CodeInterpreter>, dsrust::Example) -> Fut + Send,
) -> PyResult<String>
where
    Fut: std::future::Future<Output = anyhow::Result<dsrust::Prediction>> + Send,
{
    let mut fields = Vec::new();
    for (name, json) in &values {
        let value: Value = serde_json::from_str(json)
            .map_err(|error| PyValueError::new_err(format!("input `{name}`: {error}")))?;
        fields.push((name.clone(), value));
    }
    // Released while the loop runs, so the interpreter and the LM can be called back into Python.
    let prediction = py
        .detach(|| pollster::block_on(ask(repl, dsrust::Example::new(fields))))
        // Only the *terminal* interpreter failure gets its own class here. Everything else a
        // module raises — a parse failure, "Max hops reached." — is the module's own error and
        // keeps crossing as `ValueError`, which is what the shim's `RuntimeError` restoration and
        // dspy's `AdapterParseError` path both read. Mapping all of them cost three tests.
        //
        // A terminal failure that began as a Python exception is re-raised *as that exception*.
        // Upstream catches `(CodeExecutionError, SyntaxError)` and nothing else, so everything
        // else leaves `forward` untouched — a `ValueError` an interpreter raised is still a
        // `ValueError` to the caller. Minting a fresh class here instead put every one of them
        // through the shim's `CodeInterpreterError`, which is a class the caller cannot catch by
        // the name it raised. Only a failure with no Python exception behind it — the crate's own
        // Deno sandbox — becomes the bridge's.
        .map_err(
            |error| match error.downcast_ref::<dsrust::interpreter::InterpreterFailure>() {
                Some(dsrust::interpreter::InterpreterFailure::Session(_)) => owner
                    .parked()
                    .unwrap_or_else(|| crate::sandbox::sandbox_failure(error)),
                _ => to_value_error(error),
            },
        )?;
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

/// dspy `to_openai_responses_request`, taken from the *typed* request the crate holds.
///
/// [`responses_body`] crosses the other of dspy's two routes — the legacy chat dict — and cannot
/// reach this one: a test calling `to_openai_responses_request(LMRequest.from_call(...))` never
/// touches the chat converter, so the tools, the tool choice, the reasoning config and the
/// requested schema were being asserted against dspy's own answer with the crate absent.
///
/// `JsonFormat::Schema` because a typed request naming a `response_format` is asking for the named
/// strict schema, which is the branch upstream takes here.
#[pyfunction]
pub(crate) fn responses_request(request: &str) -> PyResult<String> {
    let call: dsrust::lm::api::LmRequest =
        serde_json::from_str(request).map_err(|error| PyValueError::new_err(format!("{error}")))?;
    let body =
        dsrust::lm::openai::responses::request(&call.model, &call, dsrust::lm::JsonFormat::Schema)
            .map_err(|error| PyValueError::new_err(format!("{error:#}")))?;
    serde_json::to_string(&body).map_err(|error| PyValueError::new_err(format!("{error}")))
}

/// dspy `responses_to_lm_response`: a raw Responses body read into the typed response.
///
/// The pair to [`responses_request`], and the seam that says what a caller *sees* — which output
/// item became which part, what the usage adds up to, and what a tool call whose arguments would
/// not parse leaves behind.
#[pyfunction]
pub(crate) fn responses_outputs(body: &str, model: &str) -> PyResult<String> {
    let body: Value =
        serde_json::from_str(body).map_err(|error| PyValueError::new_err(format!("{error}")))?;
    let response = dsrust::lm::openai::responses::responses_to_lm_response(&body, model);
    serde_json::to_string(&response).map_err(|error| PyValueError::new_err(format!("{error}")))
}

/// The messages a typed request puts on the chat wire — dspy `to_openai_chat_request`'s `messages`.
///
/// The one part of that body the crate builds from parts rather than copying from config: a turn
/// of text collapses to a bare string, a turn carrying calls puts them beside its content, and a
/// tool result names the call it answers. A caller assembling a transcript by hand —
/// `User`/`Assistant(ToolCall)`/`ToolResult` — is asserting on exactly this.
#[pyfunction]
pub(crate) fn wire_messages(request: &str) -> PyResult<String> {
    let call: dsrust::lm::api::LmRequest =
        serde_json::from_str(request).map_err(|error| PyValueError::new_err(format!("{error}")))?;
    serde_json::to_string(&call.wire_messages())
        .map_err(|error| PyValueError::new_err(format!("{error}")))
}

/// dspy `PythonInterpreter._serialize_value`: one value as the Python source that rebuilds it.
///
/// The bytes pasted into the sandbox, so this is the boundary rather than a step near it. Turning
/// a Python object into something JSON can hold stays Python's — that is reflection, and dspy
/// raises there on a value with no JSON form, before its own serializer is reached.
#[pyfunction]
pub(crate) fn python_literal(value: &str) -> PyResult<String> {
    let value: Value =
        serde_json::from_str(value).map_err(|error| PyValueError::new_err(format!("{error}")))?;
    Ok(dsrust::interpreter::python_literal(&value))
}

/// dspy `_close_object_schemas`, applied by the crate.
///
/// The class-to-schema step stays Python's — that is pydantic reflection, and reflection is what
/// the shim is for. What crosses is the rule: which positions in a schema are subschemas to walk,
/// and which are values to leave alone.
#[pyfunction]
pub(crate) fn closed_object_schemas(schema: &str) -> PyResult<String> {
    let schema: Value =
        serde_json::from_str(schema).map_err(|error| PyValueError::new_err(format!("{error}")))?;
    let closed = dsrust::lm::openai::responses::close_object_schemas(&schema);
    serde_json::to_string(&closed).map_err(|error| PyValueError::new_err(format!("{error}")))
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

/// dspy `clients/lm.py::_is_openai_reasoning_model`, decided by the crate.
///
/// The *state* predicate, which is the one this stands in for: it decides what `dump_state` writes
/// and what `load_state` reads back. dspy's other one, in `clients/openai_format.py`, decides the
/// wire and is crossed by [`wire_reasoning_model`] beside it — the two disagree on five names, so
/// a single crossing would leave one of them answered by Python.
#[pyfunction]
pub(crate) fn is_openai_reasoning_model(model: &str) -> bool {
    dsrust::lm::reasoning_model::in_saved_state(model)
}

/// dspy `clients/openai_format.py::_is_openai_reasoning_model`, decided by the crate.
#[pyfunction]
pub(crate) fn wire_reasoning_model(model: &str) -> bool {
    dsrust::lm::reasoning_model::on_the_wire(model)
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
    // Upstream's own first step, and the reason this route tolerates what the typed one refuses:
    // a chat dict reaching here may be a provider-SDK dump or carry Responses-native blocks, and
    // `_sanitize_legacy_message` is where both are made readable. The rules are the crate's; only
    // the routing is here.
    let messages: Vec<LmMessage> = written
        .get("messages")
        .and_then(Value::as_array)
        .map(|messages| {
            messages
                .iter()
                .map(dsrust::lm::api::sanitized_legacy_message)
                .collect()
        })
        .map(Value::Array)
        .map(serde_json::from_value)
        .transpose()
        .map_err(|error| PyValueError::new_err(format!("messages: {error}")))?
        .unwrap_or_default();

    let mut config = LmConfig::default();
    if let Some(schema) = written.get("response_format") {
        config.response_format = Some(schema.clone());
    }
    let call = LmRequest::new(model, messages).configured(config);
    // Fallible since the builder validates first: an OpenAI reasoning model asked to reason at a
    // chosen temperature is refused before a body exists, which is dspy's own check.
    let body = dsrust::lm::openai::responses::request(model, &call, dsrust::lm::JsonFormat::Schema)
        .map_err(|error| PyValueError::new_err(format!("{error:#}")))?;
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

/// A factory that must never fire: every bridge call carries a caller-owned interpreter.
///
/// The Python side has already run dspy's `_interpreter_context` by the time the bridge is
/// called, so the ownership decision is made — the interpreter crossing here is *borrowed*, and
/// the crate must not shut it down. It did: `handing_back(...)` satisfied the constructor but made
/// the lease module-owned, so every pass shut down the caller's interpreter on its way out.
/// `PyInterpreter::shutdown` forwarded that to Python, the pooled interpreter's session ended, and
/// the pool's next `_pool_reset()` raised "session has ended" — 26 teardown errors whose cause was
/// one wrong arm in this file.
fn refusing_factory() -> dsrust::interpreter::InterpreterFactory {
    std::sync::Arc::new(|| {
        anyhow::bail!(
            "the bridge always passes a caller-owned interpreter; this factory must not fire"
        )
    })
}

/// A Python exception from dspy's interpreter, as the crate's typed failure.
///
/// `CodeExecutionError` subclasses `CodeInterpreterError` upstream, so the narrower class is
/// tested first — the other order calls every code failure terminal.
fn interpreter_failure(py: Python<'_>, error: &PyErr) -> anyhow::Error {
    let said = raised_message(py, error);
    let Ok(module) = py.import("dspy.primitives.code_interpreter") else {
        return anyhow::Error::new(InterpreterFailure::Execution(said));
    };
    let is = |name: &str| {
        module
            .getattr(name)
            .and_then(|class| error.value(py).is_instance(&class))
            .unwrap_or(false)
    };
    // Upstream's own rule, read off its `except` clause rather than assumed:
    // `ProgramOfThought._execute_code` catches `(CodeExecutionError, SyntaxError)` and nothing
    // else, so those two are the recoverable set and *everything else propagates* — a bare
    // `ValueError` from an interpreter included. This defaulted the other way, which fed an
    // unrecognised failure back to the model as something to rewrite and let the run continue.
    let syntax = error
        .value(py)
        .is_instance_of::<pyo3::exceptions::PySyntaxError>();
    match () {
        _ if syntax || is("CodeExecutionError") => {
            anyhow::Error::new(InterpreterFailure::Execution(said))
        }
        _ => anyhow::Error::new(InterpreterFailure::Session(said)),
    }
}
