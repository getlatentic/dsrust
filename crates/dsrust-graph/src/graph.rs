//! The document, as a `Module`.
//!
//! Two methods carry it. `forward` interprets the wiring — the wiring *is* the forward, and no
//! user-written code runs. `named_predictors` hands every node to an optimizer, named after the
//! node so a caller can say which box on the canvas a rewritten instruction belongs to.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use dsrust::anyhow::{Context, Result, anyhow};
use dsrust::lm::DynChatModel;
use dsrust::module::{NamedPredictor, ProgramState, SubmoduleState};
use dsrust::serde_json::Value;
use dsrust::signature::{FieldKind, InField, JsonType, OutField};
use dsrust::{Example, Module, Predict, Prediction, Signature};

use crate::document::{Declared, Field, GraphDocument, Source, Wire};

/// The signature a node asks with, from either spelling.
///
/// A string is parsed, which is what a builder writing `"subject -> angle"` wants. Field nodes are
/// *assembled* — and that is the path a canvas with a node per field has to take, because once a
/// field carries a type the string spelling cannot express, there is no string to parse.
fn declared(declared: &Declared) -> Result<Signature> {
    match declared {
        Declared::Written { signature } => Ok(signature.parse()?),
        Declared::Fields { inputs, outputs } => {
            if outputs.is_empty() {
                return Err(anyhow!(
                    "a node with no output field asks the model for nothing"
                ));
            }
            Ok(Signature {
                instructions: String::new(),
                inputs: inputs.iter().map(as_input).collect(),
                outputs: outputs.iter().map(as_output).collect(),
            })
        }
    }
}

fn as_input(field: &Field) -> InField {
    InField {
        name: field.name.clone(),
        desc: field.description.clone(),
        kind: kind_of(&field.r#type),
        ..Default::default()
    }
}

fn as_output(field: &Field) -> OutField {
    OutField {
        name: field.name.clone(),
        desc: field.description.clone(),
        kind: kind_of(&field.r#type),
        ..Default::default()
    }
}

/// The field's declared type, as the kind a prompt renders.
///
/// The four scalars name themselves. Anything else — `list[str]`, `dict[str, Any]`, a custom type's
/// own name — is a structured field carrying that annotation verbatim, which is exactly how a
/// custom type reaches dspy: the annotation is printed and the value travels as JSON. So a canvas
/// can offer any type it likes without this needing to know them.
fn kind_of(declared: &str) -> FieldKind {
    match declared {
        "str" | "string" => FieldKind::Str,
        "bool" | "boolean" => FieldKind::Bool,
        "int" | "integer" => FieldKind::Int,
        "float" | "number" => FieldKind::Float,
        "reasoning" => FieldKind::Reasoning,
        annotation => FieldKind::Json(JsonType {
            annotation: annotation.to_owned(),
            ..Default::default()
        }),
    }
}

/// One node, ready to run: the module, its id, and the edges feeding it.
///
/// A module rather than a `Predict`, because a node's kind decides what it is —
/// a ReAct runs a tool loop and holds two predictors of its own. Held boxed
/// because `Module` is what every node has in common and the only thing this
/// needs from one: run it, walk its predictors, trace it.
struct Step {
    id: String,
    module: Box<dyn Module>,
    inputs: Vec<Wire>,
}

/// A program whose shape came from a document rather than from a struct.
pub struct Graph {
    steps: Vec<Step>,
    /// What the program answers with: each output field's name, and where its value comes from.
    answers: Vec<Resolved>,
}

/// One answer field, with its source node already resolved to a position.
struct Resolved {
    name: String,
    at: usize,
    field: String,
}

impl Graph {
    /// Build the program the document describes.
    ///
    /// Every node's signature is parsed here, so a document that names a bad one fails at load
    /// rather than at the first run — which is the closest a runtime-shaped program gets to the
    /// build-time check `Predict!("subject -> angle")` would have given it.
    pub fn from_document(document: &GraphDocument, lm: Arc<dyn DynChatModel>) -> Result<Self> {
        let steps = document
            .nodes
            .iter()
            .map(|node| {
                let mut signature = declared(&node.declared).with_context(|| {
                    format!("node {:?} does not declare a usable signature", node.id)
                })?;
                if let Some(instructions) = &node.instructions {
                    signature = signature.with_instructions(instructions);
                }
                Ok(Step {
                    id: node.id.clone(),
                    module: Box::new(Predict::from_signature(signature).set_lm(Arc::clone(&lm))),
                    inputs: node.inputs.clone(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        if document.answers.is_empty() {
            return Err(anyhow!("the document names no output field to answer with"));
        }
        let answers = document
            .answers
            .iter()
            .map(|answer| {
                Ok(Resolved {
                    name: answer.name.clone(),
                    at: steps
                        .iter()
                        .position(|step| step.id == answer.node)
                        .with_context(|| {
                            format!("no node named {:?} to answer with", answer.node)
                        })?,
                    field: answer.field.clone(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { steps, answers })
    }

    /// Every node the optimizer will reach, before handing it one.
    ///
    /// The failure this exists for is silent: a program whose `named_predictors` misses a node is
    /// one an optimizer walks past, and `compile` still returns `Ok`. The run is green, the diff is
    /// empty, and nothing says which. Calling this first turns that into an error at the moment it
    /// is still cheap to fix.
    ///
    /// A caller showing an instruction diff wants the second half too: an optimization that walked
    /// every node and *still* changed nothing is a real outcome worth showing, and it looks
    /// identical to this bug unless the walk was checked.
    pub fn walk_covers_every_node(&mut self) -> Result<()> {
        let declared: Vec<String> = self.steps.iter().map(|step| step.id.clone()).collect();
        let walked: Vec<String> = self
            .named_predictors()
            .into_iter()
            .map(|found| found.name)
            .collect();
        let missed: Vec<&String> = declared.iter().filter(|id| !walked.contains(id)).collect();
        match missed.is_empty() {
            true => Ok(()),
            false => Err(anyhow!(
                "the optimizer would walk {} of {} nodes, missing {missed:?} — a compile would \
                 report success having rewritten nothing",
                walked.len(),
                declared.len()
            )),
        }
    }

    /// Which node an edge names, as a position — resolved once here so `forward` cannot be handed
    /// an edge pointing at a node that does not exist.
    fn position_of(&self, id: &str) -> Result<usize> {
        self.steps
            .iter()
            .position(|step| step.id == id)
            .ok_or_else(|| {
                anyhow!("an edge names node {id:?}, which the document does not declare")
            })
    }

    /// Whether a saved state describes a differently-shaped program than this one.
    ///
    /// dspy restores a signature by zipping the saved fields onto the live ones and stopping at
    /// the shorter — `zip(..., strict=False)`, explicitly — so a program that has gained a field
    /// restores what it can. That is right for dspy, where a program's shape is its source code
    /// and cannot change under a saved state without a recompile.
    ///
    /// A node here can. It keeps its id while changing kind, and a `Predict` that becomes a
    /// `ChainOfThought` grows a leading `reasoning` output — so every saved field lands one
    /// position late, the reasoning renders under the next field's prefix, and the last field
    /// falls back to an inferred default. The program that runs is one nobody wrote, and nothing
    /// says so.
    ///
    /// Answers the complaint rather than refusing outright, because what to do about it is the
    /// caller's: a program is still runnable untuned, and saying which node disagreed is more use
    /// than a failed load.
    pub fn foreign_state(&mut self, state: &ProgramState) -> Option<String> {
        for predictor in self.named_predictors() {
            let Some(SubmoduleState::Predictor(saved)) = state.state(&predictor.name) else {
                continue;
            };
            let live = predictor.signature.inputs.len() + predictor.signature.outputs.len();
            if saved.signature.fields.len() != live {
                return Some(format!(
                    "node {:?} has {live} fields and the state has {}",
                    predictor.name,
                    saved.signature.fields.len()
                ));
            }
        }
        None
    }

    /// One node's inputs, read from the program's own inputs or from an earlier node's answer.
    fn feed(&self, step: &Step, inputs: &Example, produced: &[Prediction]) -> Result<Example> {
        let mut fed = Example::default();
        for wire in &step.inputs {
            let value = match &wire.source {
                Source::Input { field } => inputs.get(field).cloned(),
                Source::Node { node, field } => {
                    let at = self.position_of(node)?;
                    produced
                        .get(at)
                        .and_then(|answer| answer.get(field))
                        .cloned()
                }
            };
            fed.set(&wire.name, value.unwrap_or(Value::Null));
        }
        Ok(fed)
    }

    /// The answer is the output layer, not one node: each field names itself and takes its value
    /// from one field of one module, so a program with several outputs answers with all of them
    /// under their own names.
    fn answer(&self, produced: &[Prediction]) -> Result<Prediction> {
        let mut answered = Example::default();
        for answer in &self.answers {
            let value = produced
                .get(answer.at)
                .and_then(|step| step.get(&answer.field))
                .cloned()
                .ok_or_else(|| {
                    anyhow!(
                        "node {:?} produced no field {:?} for output {:?}",
                        self.steps[answer.at].id,
                        answer.field,
                        answer.name
                    )
                })?;
            answered.set(&answer.name, value);
        }
        Ok(Prediction::new(answered, ""))
    }
}

impl Module for Graph {
    /// The wiring is the forward: each node's inputs are read from the program's own inputs or
    /// from an earlier node's answer, in declaration order.
    ///
    /// **The body is wrapped in the observability point `#[derive(Module)]` writes.** That derive
    /// is dspy's `Module.__call__` decorator: it opens `on_module_start` before the body and closes
    /// `on_module_end` after it. A hand-written `forward` *is* that entry, so leaving it out means
    /// the outermost program of a run emits nothing — the inner `Predict`s still report, so a
    /// listener sees steps happening inside a module that never started.
    ///
    /// Declaration order is the execution order, which is what makes this an interpreter rather
    /// than a scheduler. A document that wires a node from a later one is a document the builder
    /// should have rejected; here it reads as a missing value rather than a deadlock.
    fn forward<'a>(
        &'a self,
        inputs: Example,
    ) -> Pin<Box<dyn Future<Output = Result<Prediction>> + Send + 'a>> {
        let watch = dsrust::observe::module_shown("Graph", &inputs, Module::callbacks(self));
        Box::pin(dsrust::observe::watching(watch, async move {
            let mut produced: Vec<Prediction> = Vec::new();
            for step in &self.steps {
                let fed = self.feed(step, &inputs, &produced)?;
                produced.push(step.module.forward(fed).await?);
            }
            self.answer(&produced)
        }))
    }

    /// The half `#[derive(Module)]` would have written, and the half a hand-written module
    /// forgets.
    ///
    /// Without it the trait's default answers with nothing: an optimizer walks no predictors,
    /// rewrites none, and reports success. Naming each after its node is what lets a caller show a
    /// rewritten instruction against the box it belongs to.
    fn named_predictors(&mut self) -> Vec<NamedPredictor<'_>> {
        self.steps
            .iter_mut()
            .flat_map(|step| {
                let id = step.id.clone();
                step.module
                    .named_predictors()
                    .into_iter()
                    .map(move |mut found| {
                        // A leaf predictor names itself `self`, and the node id
                        // stands for it whole. A module holding several — a
                        // ReAct's `react` and `extract.predict` — keeps each
                        // under the node, the way dspy paths a submodule.
                        found.name = match found.name.as_str() {
                            "self" => id.clone(),
                            inner => format!("{id}.{inner}"),
                        };
                        found
                    })
            })
            .collect()
    }
}
