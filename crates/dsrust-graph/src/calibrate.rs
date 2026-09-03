//! Calibrate's own document, read into a program.
//!
//! The shape is theirs, taken from their own `calibrate-codegen` fixtures and kept here as
//! `tests/fixtures/multi_module_graph.json`. Three things about it decide the translation:
//!
//! * **Edges are port-to-port, and a field can be renamed across one.** `predict.field_out` carries
//!   `fieldName: "answer"` into `cot.field_in` as `fieldName: "context"`. The downstream input is
//!   named by the *receiving* end, and the value is fetched by the *sending* end's name.
//! * **A module's signature comes from its edges**, not from a string or from field nodes of its
//!   own: inputs are the incoming data edges' `to.fieldName`, outputs the outgoing edges'
//!   `from.fieldName`. So a module with no outgoing edge declares no outputs, and is a document bug
//!   rather than a program.
//! * **The answer is the OutputField layer.** Each `OutputField` node names one field of the
//!   program's answer and is fed by one module field.
//!
//! `edgeKind: "control"` orders execution and carries no data, so it is read for nothing here —
//! order comes from the node list, which Calibrate emits topologically.

use std::collections::BTreeMap;

use dsrust::anyhow::{Context, Result, anyhow};
use dsrust::serde_json::Value;
use serde::{Deserialize, Serialize};

use crate::document::{Answer, Declared, Field, GraphDocument, Node, Source, Wire};

/// The node kinds that become a predictor.
const MODULES: &[&str] = &["Predict", "ChainOfThought", "ReAct", "CustomModule"];

/// The kinds that are not predictors and are not meant to be: the program's edges, and the
/// ordering marker.
///
/// Named rather than assumed, because a kind in neither list is one this builder does not
/// understand — and quietly leaving it out builds a *different program* than the document
/// describes. See [`CalibrateGraph::known_kinds`].
const BOUNDARY: &[&str] = &["Start", "InputField", "OutputField", "Tool"];

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CalibrateGraph {
    #[serde(default)]
    pub nodes: Vec<GraphNode>,
    #[serde(default)]
    pub edges: Vec<GraphEdge>,
    /// The per-module signature, whose `docstring` is the instruction an optimizer rewrites.
    #[serde(default)]
    pub signatures: Vec<SignatureSpec>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphNode {
    pub id: String,
    pub kind: String,
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub config: BTreeMap<String, Value>,
    #[serde(default)]
    pub signature_id: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphEdge {
    #[serde(default)]
    pub edge_kind: String,
    pub from: EdgeEndpoint,
    pub to: EdgeEndpoint,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EdgeEndpoint {
    pub node_id: String,
    #[serde(default)]
    pub port_id: String,
    /// Absent at a boundary port, where the field's name is the field node's own.
    #[serde(default)]
    pub field_name: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SignatureSpec {
    pub id: String,
    #[serde(default)]
    pub docstring: String,
}

impl CalibrateGraph {
    pub fn from_json(text: &str) -> Result<Self> {
        Ok(dsrust::serde_json::from_str(text)?)
    }

    /// The document, as a program.
    pub fn to_document(&self) -> Result<GraphDocument> {
        self.known_kinds()?;
        let nodes = self
            .nodes
            .iter()
            .filter(|node| MODULES.contains(&node.kind.as_str()))
            .map(|node| self.module(node))
            .collect::<Result<Vec<_>>>()?;
        Ok(GraphDocument {
            nodes,
            answers: self.answers()?,
        })
    }

    /// Every node kind is one this builder knows.
    ///
    /// Filtering to the module kinds and moving on is what a builder naturally does, and it is
    /// wrong: a kind added to Calibrate and not here is silently dropped, and what gets built is a
    /// program with fewer steps than the document has — running less, and optimizing less, with
    /// nothing anywhere saying so. Measured on this crate's own fixture: renaming one module's kind
    /// built a program of one node whose answer pointed at the node that had been dropped.
    ///
    /// So an unknown kind stops the load. A new kind is a change to this list and a decision about
    /// what it means, which is the point.
    fn known_kinds(&self) -> Result<()> {
        let unknown: Vec<String> = self
            .nodes
            .iter()
            .filter(|node| {
                !MODULES.contains(&node.kind.as_str()) && !BOUNDARY.contains(&node.kind.as_str())
            })
            .map(|node| format!("{} ({})", node.id, node.kind))
            .collect();
        match unknown.is_empty() {
            true => Ok(()),
            false => Err(anyhow!(
                "this builder does not know the kind of {unknown:?} — building without them would \
                 be a program the document does not describe"
            )),
        }
    }

    /// One module: the fields its edges give it, the edges that feed them, and its instructions.
    fn module(&self, node: &GraphNode) -> Result<Node> {
        // A kind this builder cannot honour is refused, never approximated.
        // Both of these carry something the translation drops — ReAct its tools
        // and iteration limit, CustomModule its Python body — and a node built
        // as a plain Predict runs a different program than the canvas draws and
        // the exported Python emits, without saying so. That silence is the
        // same defect ChainOfThought had here until it grew its reasoning
        // field, and it is worse for these two because there is no field to add.
        match node.kind.as_str() {
            "ReAct" => {
                return Err(anyhow!(
                    "node {:?} is a ReAct: it calls tools in a loop, and this builder models no \
                     tools — built here it would answer without ever calling one. Switch this app \
                     to the Python runtime in Settings, or make the node a Predict or \
                     ChainOfThought.",
                    node.id
                ));
            }
            "CustomModule" => {
                return Err(anyhow!(
                    "node {:?} is a CustomModule: its body is Python, which this builder cannot \
                     run — built here that code would be ignored rather than executed. Switch \
                     this app to the Python runtime in Settings.",
                    node.id
                ));
            }
            _ => {}
        }
        let incoming: Vec<&GraphEdge> = self
            .data_edges()
            .filter(|e| e.to.node_id == node.id)
            .collect();
        let outgoing: Vec<&GraphEdge> = self
            .data_edges()
            .filter(|e| e.from.node_id == node.id)
            .collect();
        if outgoing.is_empty() {
            return Err(anyhow!(
                "module {:?} has no outgoing data edge, so it declares no output field",
                node.id
            ));
        }

        let inputs = incoming
            .iter()
            .map(|edge| {
                Ok(Field {
                    name: self.receiving_name(edge)?,
                    // A module-to-module field carries a name and a description across the edge and
                    // no type; only a boundary field node declares one.
                    r#type: self.type_at(&edge.from).unwrap_or_else(|| "str".to_owned()),
                    description: edge.to.description.clone().unwrap_or_default(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut outputs = outgoing
            .iter()
            .map(|edge| {
                Ok(Field {
                    name: self.sending_name(edge)?,
                    r#type: self.type_at(&edge.to).unwrap_or_else(|| "str".to_owned()),
                    description: String::new(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        // One output feeding several consumers is one field, not one per consumer. An edge is a
        // *reader* of a field, and the count of readers is not part of what the module produces —
        // declaring it per edge made the signature `-> answer, answer` and the adapter then
        // demanded the model answer `answer` twice, failing a reply that was correct.
        //
        // Inputs are deliberately not deduped: two edges feeding one input name is a document
        // saying two different things fill one slot, which is better loud.
        let mut declared = std::collections::BTreeSet::new();
        outputs.retain(|field| declared.insert(field.name.clone()));
        // dspy's ChainOfThought *is* the same signature with a leading `reasoning`
        // output — there is no separate module, only that field. Reading the kind
        // is what makes the difference: without it a node the canvas draws as a
        // ChainOfThought, and the exported Python emits as `dspy.ChainOfThought`,
        // ran here as a plain Predict with no reasoning step at all.
        if node.kind == "ChainOfThought" && !declared.contains("reasoning") {
            outputs.insert(
                0,
                Field {
                    name: "reasoning".to_owned(),
                    // `str`, which is what dspy's own ChainOfThought declares
                    // its rationale as. dspy 3.3's `Reasoning` type renders the
                    // same *except* that the output-requirement hint fires —
                    // "(must be formatted as a valid Python str)" — because its
                    // annotation is not literally `str`. Ten tokens a call, and
                    // worse: told the field must be a valid Python str, the
                    // model quotes, so exact match then fails on `"positive"`.
                    // Every score taken through this path carried that handicap.
                    r#type: "str".to_owned(),
                    description: String::new(),
                },
            );
        }

        Ok(Node {
            id: node.id.clone(),
            declared: Declared::Fields { inputs, outputs },
            instructions: self.instructions(node),
            inputs: incoming
                .iter()
                .map(|edge| {
                    Ok(Wire {
                        name: self.receiving_name(edge)?,
                        source: match self.kind_of(&edge.from.node_id).as_deref() {
                            // A boundary: the program's own input, named by the field node.
                            Some("InputField") => Source::Input {
                                field: self.field_name_of(&edge.from.node_id)?,
                            },
                            // A module: fetched by the *sending* end's name, which the rename
                            // across the edge makes different from the receiving one.
                            _ => Source::Node {
                                node: edge.from.node_id.clone(),
                                field: self.sending_name(edge)?,
                            },
                        },
                    })
                })
                .collect::<Result<Vec<_>>>()?,
        })
    }

    /// Every `OutputField` node, and the module field feeding it.
    fn answers(&self) -> Result<Vec<Answer>> {
        self.nodes
            .iter()
            .filter(|node| node.kind == "OutputField")
            .map(|node| {
                let feeding = self
                    .data_edges()
                    .find(|edge| edge.to.node_id == node.id)
                    .with_context(|| format!("output field {:?} is fed by nothing", node.id))?;
                Ok(Answer {
                    name: self.field_name_of(&node.id)?,
                    node: feeding.from.node_id.clone(),
                    field: self.sending_name(feeding)?,
                })
            })
            .collect()
    }

    fn data_edges(&self) -> impl Iterator<Item = &GraphEdge> {
        // Anything not explicitly `control` carries data; an older document may omit the kind.
        self.edges.iter().filter(|edge| edge.edge_kind != "control")
    }

    /// A module input's name: the receiving edge's own `fieldName`, else the `InputField` node
    /// feeding it.
    ///
    /// **Edge-wins**, and deliberately: a field chained from another module is named only on the
    /// edge, so the edge has to be able to name it.
    fn receiving_name(&self, edge: &GraphEdge) -> Result<String> {
        if let Some(name) = &edge.to.field_name {
            return Ok(name.clone());
        }
        self.boundary_name(edge, [&edge.to, &edge.from])
    }

    /// A module output's name: the `OutputField` node it feeds, else the sending edge's own
    /// `fieldName`.
    ///
    /// **Node-wins, which is the opposite precedence to an input's** — Calibrate's own resolver
    /// does this so that renaming an output field propagates immediately rather than waiting for
    /// the edge to catch up. A stale `from.fieldName` beside a renamed `OutputField.config.name`
    /// is exactly the disagreement, and taking the edge's word for it would build a program
    /// answering under the old name.
    ///
    /// Between two modules there is no node to ask, so the edge names it or nothing does.
    fn sending_name(&self, edge: &GraphEdge) -> Result<String> {
        if self.kind_of(&edge.to.node_id).as_deref() == Some("OutputField") {
            return self.field_name_of(&edge.to.node_id);
        }
        if let Some(name) = &edge.from.field_name {
            return Ok(name.clone());
        }
        self.boundary_name(edge, [&edge.from, &edge.to])
    }

    /// The name a boundary field node gives, trying each end in turn.
    ///
    /// `seed_field_node_graph.json` names its fields at neither end of an edge, so the name comes
    /// from the `InputField` and `OutputField` nodes the edge runs between. Both spellings stay
    /// valid — `EdgeEndpoint.fieldName` is optional by design, absent for older edges and for
    /// boundary ports, which carry one field per port by construction.
    fn boundary_name(&self, edge: &GraphEdge, ends: [&EdgeEndpoint; 2]) -> Result<String> {
        for endpoint in ends {
            if self.is_field_node(&endpoint.node_id) {
                return self.field_name_of(&endpoint.node_id);
            }
        }
        Err(anyhow!(
            "the edge from {:?} to {:?} names its field at neither end, and runs between two \
             modules — so there is no field node to take the name from",
            edge.from.node_id,
            edge.to.node_id
        ))
    }

    fn is_field_node(&self, id: &str) -> bool {
        matches!(
            self.kind_of(id).as_deref(),
            Some("InputField") | Some("OutputField")
        )
    }

    fn node(&self, id: &str) -> Result<&GraphNode> {
        self.nodes
            .iter()
            .find(|node| node.id == id)
            .with_context(|| format!("an edge names node {id:?}, which the document does not have"))
    }

    fn kind_of(&self, id: &str) -> Option<String> {
        self.nodes
            .iter()
            .find(|node| node.id == id)
            .map(|node| node.kind.clone())
    }

    /// A field node's own name, from its config.
    fn field_name_of(&self, id: &str) -> Result<String> {
        self.node(id)?
            .config
            .get("name")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .with_context(|| format!("field node {id:?} declares no name"))
    }

    /// The declared type at a boundary, if this endpoint is one. A module-to-module field has none.
    fn type_at(&self, endpoint: &EdgeEndpoint) -> Option<String> {
        let node = self.nodes.iter().find(|node| node.id == endpoint.node_id)?;
        match node.kind.as_str() {
            "InputField" | "OutputField" => node
                .config
                .get("type")
                .and_then(Value::as_str)
                .map(str::to_owned),
            _ => None,
        }
    }

    /// The module's instructions: its signature's docstring.
    ///
    /// Absent or empty falls back to a generic line naming the node, which is what Calibrate's
    /// current Python emitter synthesizes for every module. Reading the real docstring first is
    /// the more faithful of the two, and the fallback keeps a document without one running.
    fn instructions(&self, node: &GraphNode) -> Option<String> {
        let written = node.signature_id.as_ref().and_then(|id| {
            self.signatures
                .iter()
                .find(|spec| &spec.id == id)
                .filter(|spec| !spec.docstring.trim().is_empty())
                .map(|spec| spec.docstring.clone())
        });
        Some(written.unwrap_or_else(|| {
            let label = match node.label.is_empty() {
                true => node.id.clone(),
                false => node.label.clone(),
            };
            format!("{} signature for {label} (dspy.{}).", node.kind, node.kind)
        }))
    }
}
