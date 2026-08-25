//! The host side of the `dspy.Flex` bridge: what answers the sandbox while its code runs.
//!
//! The optimizer-authored module is Python and runs in the interpreter; the predictors it builds
//! are this crate's and run here, so the model calls are real. Upstream registers two callables on
//! the interpreter for that — `__dspy_construct__` and `__dspy_call__` — and this crate already has
//! the same contract under another name: a [`Tool`] the sandbox calls by name, answered on the RPC
//! loop in `interpreter/deno.rs`.
//!
//! **One boundary makes this more than wiring.** `CodeInterpreter::execute` and [`Tool::call`] are
//! synchronous the whole way down, and running a predictor is not. So the interpreter runs on a
//! thread of its own and the two host tools post their questions back to the asynchronous side,
//! which builds and calls the predictor and answers. A `std::thread` and futures channels rather
//! than `spawn_blocking`, because this crate takes tokio for `time` alone and stays runtime-neutral.

use std::sync::Arc;
use std::sync::mpsc as blocking;

use anyhow::{Result, anyhow, bail};
use futures_channel::mpsc;
use futures_util::StreamExt;
use serde_json::{Map, Value, json};

use crate::example::{Example, Prediction};
use crate::interpreter::Executed;
use crate::interpreter::InterpreterFactory;
use crate::react::Tool;

use super::built::Built;
use crate::signature::Signature;

/// dspy's `CONSTRUCT_TOOL`: build a predictor host-side and hand back its handle.
pub(super) const CONSTRUCT: &str = "__dspy_construct__";
/// dspy's `CALL_TOOL`: run one of those predictors and hand back its output fields.
pub(super) const CALL: &str = "__dspy_call__";

const INPUTS_VAR: &str = "__dspy_flex_inputs";
const INSTANCE_VAR: &str = "__dspy_flex_instance";
const OUT_VAR: &str = "__dspy_flex_out";
const JSON_VAR: &str = "__dspy_flex_json";

/// One question from the sandbox, and where its answer goes.
type Question = (String, Value, blocking::Sender<Result<Value>>);

/// What each host tool takes, which is what the sandbox writes its `def` from.
///
/// The shim calls both by keyword — `_dspy_host("__dspy_construct__", kind=..., signature=...)` —
/// so a stub declaring no parameters is a `TypeError` on the first construction. The types are left
/// unstated because none of these is a Python scalar: a signature is a string or a marker dict, and
/// `kwargs` and `inputs` are dicts of whatever the generated code passed.
fn takes(names: &[&str]) -> Value {
    Value::Object(
        names
            .iter()
            .map(|name| ((*name).to_owned(), json!({})))
            .collect(),
    )
}

/// A host tool that asks the asynchronous side and waits.
///
/// The wait is a blocking receive, which is correct here and only here: this runs on the
/// interpreter's own thread, which exists to be blocked.
struct Asking {
    name: &'static str,
    args: Value,
    questions: mpsc::UnboundedSender<Question>,
}

impl Asking {
    fn new(name: &'static str, questions: mpsc::UnboundedSender<Question>) -> Self {
        let args = match name {
            CONSTRUCT => takes(&["kind", "signature", "attr_name", "kwargs"]),
            _ => takes(&["handle", "inputs"]),
        };
        Self {
            name,
            args,
            questions,
        }
    }
}

impl Tool for Asking {
    fn name(&self) -> &str {
        self.name
    }

    fn description(&self) -> &str {
        "internal: the dspy.Flex bridge"
    }

    fn args(&self) -> &Value {
        &self.args
    }

    fn call(&self, args: &Value) -> Result<String> {
        Ok(self.call_value(args)?.to_string())
    }

    fn call_value(&self, args: &Value) -> Result<Value> {
        let (answer, wait) = blocking::channel();
        self.questions
            .unbounded_send((self.name.to_owned(), args.clone(), answer))
            .map_err(|_| anyhow!("the dspy.Flex host stopped listening"))?;
        wait.recv()
            .map_err(|_| anyhow!("the dspy.Flex host answered nothing"))?
    }
}

/// dspy's `_resolve_signature`: the shim's payload back into a signature.
///
/// `dspy.Signature(...)` cannot cross as itself, so the shim sends `{__dspy_sig__, signature,
/// instructions}` and a bare string travels as a string.
pub(super) fn signature_of(payload: Option<&Value>) -> Result<Signature> {
    let (text, instructions) = match payload {
        Some(Value::String(text)) => (text.as_str(), None),
        Some(Value::Object(marker)) if marker.contains_key("__dspy_sig__") => (
            marker
                .get("signature")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("dspy.Flex: a signature payload carried no signature"))?,
            marker.get("instructions").and_then(Value::as_str),
        ),
        _ => bail!("dspy.Flex: the sandbox asked for a predictor with no signature"),
    };
    let mut signature: Signature = text.parse()?;
    if let Some(instructions) = instructions {
        signature.instructions = instructions.to_owned();
    }
    Ok(signature)
}

/// The code that instantiates the bound class and runs it — dspy's driver, name for name.
fn driver(class_name: &str) -> String {
    format!(
        "{INSTANCE_VAR} = {class_name}()\n\
         {OUT_VAR} = {INSTANCE_VAR}.forward(**{INPUTS_VAR})\n\
         import json as {JSON_VAR}\n\
         {JSON_VAR}.dumps({OUT_VAR}._fields if hasattr({OUT_VAR}, '_fields') else {OUT_VAR})"
    )
}

/// Run `module_src` in a fresh interpreter, answering the predictors it asks for.
///
/// The thread owns the interpreter and the two host tools; when it finishes they drop, which closes
/// the question channel and ends the loop below. That is why the loop needs no sentinel: the
/// channel's own end is the signal, and a thread that panicked closes it exactly as one that
/// returned does.
pub(super) async fn run(
    interpreter_factory: InterpreterFactory,
    shim: &str,
    module_src: &str,
    class_name: &str,
    tools: Vec<Arc<dyn Tool>>,
    inputs: Map<String, Value>,
    budget: Option<usize>,
) -> Result<Value> {
    let interpreter = interpreter_factory()?;
    // The generated code names its tools; the host resolves those names, so a copy stays here while
    // the originals go to the sandbox to be callable from it.
    let for_host = tools.clone();
    let (questions, mut asked) = mpsc::unbounded();
    let shim = shim.to_owned();
    let module_src = module_src.to_owned();
    let driver = driver(class_name);

    let (finished, answered) = futures_channel::oneshot::channel();
    std::thread::spawn(move || {
        let outcome = (|| -> Result<Value> {
            let mut registered = tools;
            for name in [CONSTRUCT, CALL] {
                registered.push(Arc::new(Asking::new(name, questions.clone())));
            }
            interpreter.define_tools(&registered)?;
            interpreter.execute(&shim, &Map::new())?;
            interpreter.execute(&module_src, &Map::new())?;
            let mut variables = Map::new();
            variables.insert(INPUTS_VAR.to_owned(), Value::Object(inputs));
            match interpreter.execute(&driver, &variables)? {
                Executed::Printed(value) | Executed::Submitted(value) => Ok(value),
            }
        })();
        let _ = finished.send(outcome);
    });

    let mut built = Built::new(budget, for_host, interpreter_factory);
    while let Some((name, args, reply)) = asked.next().await {
        let answer = match name.as_str() {
            CONSTRUCT => built.construct(&args),
            CALL => built.call(&args).await,
            other => Err(anyhow!("dspy.Flex: the sandbox asked for `{other}`")),
        };
        let _ = reply.send(answer);
    }
    answered
        .await
        .map_err(|_| anyhow!("dspy.Flex: the sandbox thread ended without answering"))?
}

/// The sandbox's JSON back into a prediction — dspy's `_to_prediction`.
///
/// The driver's last expression is `json.dumps(...)`, so what arrives is a *string* holding the
/// generated forward's fields. A run that produced no string never returned a `dspy.Prediction`.
pub(super) fn prediction_of(answered: &Value, signature: &Signature) -> Result<Prediction> {
    let text = answered
        .as_str()
        .filter(|text| !text.is_empty())
        .ok_or_else(|| {
            anyhow!(
                "dspy.Flex: the sandboxed forward returned no serializable result; the generated \
             forward must return a dspy.Prediction (got {answered})"
            )
        })?;
    let fields: Map<String, Value> = serde_json::from_str(text)?;
    let _ = signature;
    Ok(Prediction::new(Example::new(fields), text.to_owned()))
}
