//! The predictors the sandbox asks the host to build, and what it may ask for.
//!
//! Split from the conversation beside it because the two change for different reasons: `bridge.rs`
//! is the crossing — a thread, a channel, and a question answered — and this is *what* is being
//! asked for. A new kind lands here; a change to how the sandbox is driven lands there.

use std::sync::Arc;

use anyhow::{Result, anyhow, bail};
use serde_json::{Map, Value, json};

use crate::example::{Example, Prediction};
use crate::interpreter::InterpreterFactory;
use crate::module::Module;
use crate::predict::code_act::CodeAct;
use crate::predict::program_of_thought::ProgramOfThought;
use crate::predict::rlm::Rlm;
use crate::predict::{ChainOfThought, Predict};
use crate::react::{ReAct, ReActV2, Tool};

use super::bridge::signature_of;

/// The predictors one forward built, in the order the sandbox asked for them.
///
/// Held for the length of the call and never on the `Flex`, as upstream holds them on its
/// `_Invocation`: two forwards of the same program build their own and neither outlives its call.
#[derive(Default)]
pub(super) struct Built {
    predictors: Vec<Bridged>,
    /// How many the sandbox has called, against dspy's `max_predictor_calls`.
    ///
    /// The budget is what stops optimizer-authored code from looping the model: the source is
    /// written by a model and runs unattended, so a `while True` around a predictor is one bad
    /// proposal away. Counted on the *call* rather than the construction, as upstream counts it —
    /// building a hundred predictors costs nothing and calling one a hundred times costs money.
    calls: usize,
    budget: Option<usize>,
    /// The tools the `Flex` was given, which the generated code names rather than passes.
    tools: Vec<Arc<dyn Tool>>,
    interpreter_factory: Option<InterpreterFactory>,
}

/// dspy's `BRIDGEABLE_KINDS`, in upstream's order — what the generated code may build.
const BRIDGEABLE: [&str; 7] = [
    "Predict",
    "ChainOfThought",
    "RLM",
    "CodeAct",
    "ProgramOfThought",
    "ReAct",
    "ReActV2",
];

/// The tools a constructor asked for, resolved from the markers the shim sends.
///
/// A callable cannot cross the JSON boundary, so `tools=[shout]` arrives as
/// `[{"__dspy_tool__": "shout"}]` and the name is looked up among the ones the `Flex` was given.
/// A name that is not there is the generated code inventing a tool, which is worth saying plainly.
fn named_tools(
    kwargs: Option<&Value>,
    available: &[Arc<dyn Tool>],
    attribute: &str,
) -> Result<Vec<Arc<dyn Tool>>> {
    let Some(Value::Array(asked)) = kwargs.and_then(|kwargs| kwargs.get("tools")) else {
        return Ok(Vec::new());
    };
    asked
        .iter()
        .map(|marker| {
            let name = marker
                .get("__dspy_tool__")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("dspy.Flex: `{attribute}` was given a tool with no name"))?;
            available
                .iter()
                .find(|tool| tool.name() == name)
                .cloned()
                .ok_or_else(|| {
                    anyhow!("dspy.Flex: `{attribute}` asked for a tool named `{name}`, which this Flex was not given")
                })
        })
        .collect()
}

/// `ReAct` takes owned boxes where the rest take shared handles.
fn boxed(tools: Vec<Arc<dyn Tool>>) -> Vec<Box<dyn Tool>> {
    tools.into_iter().map(SharedTool::boxed).collect()
}

/// One shared tool, wearing the owned box `ReAct` asks for.
struct SharedTool(Arc<dyn Tool>);

impl SharedTool {
    fn boxed(tool: Arc<dyn Tool>) -> Box<dyn Tool> {
        Box::new(Self(tool))
    }
}

impl Tool for SharedTool {
    fn name(&self) -> &str {
        self.0.name()
    }
    fn description(&self) -> &str {
        self.0.description()
    }
    fn args(&self) -> &Value {
        self.0.args()
    }
    fn call(&self, args: &Value) -> Result<String> {
        self.0.call(args)
    }
    fn call_value(&self, args: &Value) -> Result<Value> {
        self.0.call_value(args)
    }
}

/// One predictor the sandbox asked for. A `ChainOfThought` wraps a `Predict` rather than being one,
/// so the two kinds cannot share a `Vec` without saying which.
enum Bridged {
    Predict(Predict),
    ChainOfThought(ChainOfThought),
    ReAct(ReAct),
    ReActV2(ReActV2),
    Rlm(Rlm),
    CodeAct(CodeAct),
    ProgramOfThought(ProgramOfThought),
}

impl Bridged {
    async fn forward(&self, inputs: Example) -> Result<Prediction> {
        match self {
            Bridged::Predict(predict) => predict.forward(inputs).await,
            Bridged::ChainOfThought(chain) => chain.forward(inputs).await,
            Bridged::ReAct(react) => react.forward(inputs).await,
            Bridged::ReActV2(react) => react.forward(inputs).await,
            Bridged::Rlm(rlm) => rlm.forward(inputs).await,
            Bridged::CodeAct(code) => code.forward(inputs).await,
            Bridged::ProgramOfThought(thought) => thought.forward(inputs).await,
        }
    }
}

impl Built {
    /// What one forward's conversation starts with: its budget, the tools the `Flex` holds, and the
    /// factory a code-executing predictor gets its own sandbox from.
    ///
    /// A constructor rather than five `pub(super)` fields, so the seam between the crossing and what
    /// is being asked for stays one call wide.
    pub(super) fn new(
        budget: Option<usize>,
        tools: Vec<Arc<dyn Tool>>,
        interpreter_factory: InterpreterFactory,
    ) -> Self {
        Self {
            predictors: Vec::new(),
            calls: 0,
            budget,
            tools,
            interpreter_factory: Some(interpreter_factory),
        }
    }

    /// dspy's `_Invocation.construct`: build the predictor the sandbox is about to bind.
    ///
    /// `attr_name` is upstream's handle — the attribute the generated `__init__` assigns to. A
    /// position answers the same question without depending on the name being unique, and the name
    /// travels in the error when a kind has no counterpart yet.
    pub(super) fn construct(&mut self, args: &Value) -> Result<Value> {
        let kind = args.get("kind").and_then(Value::as_str).unwrap_or_default();
        let attribute = args
            .get("attr_name")
            .and_then(Value::as_str)
            .unwrap_or("<unnamed>");
        let signature = signature_of(args.get("signature"))?;
        let asked = named_tools(args.get("kwargs"), &self.tools, attribute)?;
        let interpreter = self
            .interpreter_factory
            .clone()
            .ok_or_else(|| anyhow!("dspy.Flex: no interpreter factory for `{attribute}`"))?;
        let predictor = match kind {
            "Predict" => Bridged::Predict(Predict::from_signature(signature)),
            "ChainOfThought" => Bridged::ChainOfThought(ChainOfThought::from_signature(signature)),
            "ReAct" => Bridged::ReAct(ReAct::new(signature, boxed(asked))),
            "ReActV2" => Bridged::ReActV2(ReActV2::new(signature, boxed(asked))),
            // The code-executing three take the Flex's own factory, so the inner sandbox is the
            // backend chosen for the Flex rather than whatever the default happens to be.
            "RLM" => Bridged::Rlm(Rlm::interpreter_factory(signature, interpreter)),
            "CodeAct" => {
                Bridged::CodeAct(CodeAct::interpreter_factory(signature, asked, interpreter))
            }
            "ProgramOfThought" => Bridged::ProgramOfThought(ProgramOfThought::interpreter_factory(
                signature,
                interpreter,
            )),
            other => bail!(
                "dspy.{other} is not supported inside a sandboxed dspy.Flex yet \
                 (bridgeable: {})",
                BRIDGEABLE.join(", ")
            ),
        };
        self.predictors.push(predictor);
        Ok(json!(self.predictors.len() - 1))
    }

    /// dspy's `_Invocation.call`: run one, and answer with the fields the sandbox reads back.
    pub(super) async fn call(&mut self, args: &Value) -> Result<Value> {
        self.calls += 1;
        if self.budget.is_some_and(|budget| self.calls > budget) {
            let budget = self.budget.unwrap_or_default();
            bail!(
                "Sandboxed dspy.Flex forward exceeded its predictor-call budget ({budget}). \
                 Raise max_predictor_calls if this is expected."
            );
        }
        let handle = args
            .get("handle")
            .and_then(Value::as_u64)
            .ok_or_else(|| anyhow!("dspy.Flex: the sandbox called a predictor with no handle"))?;
        let predictor = self
            .predictors
            .get(handle as usize)
            .ok_or_else(|| anyhow!("dspy.Flex: no predictor with handle {handle}"))?;
        let inputs = match args.get("inputs") {
            Some(Value::Object(fields)) => fields.clone(),
            _ => Map::new(),
        };
        let answered = predictor.forward(Example::new(inputs)).await?;
        Ok(fields_of(&answered))
    }
}

/// dspy's `prediction_to_fields`: what the sandbox reads back off a predictor it called.
///
/// Every field the prediction carries, which is what upstream hands over — narrowing to the
/// signature's declared outputs would drop a `ChainOfThought`'s reasoning, and the generated code
/// is entitled to read it.
fn fields_of(prediction: &Prediction) -> Value {
    Value::Object(
        prediction
            .example
            .fields()
            .map(|(name, value)| (name.to_owned(), value.clone()))
            .collect(),
    )
}
