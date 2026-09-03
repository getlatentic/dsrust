//! The document a visual builder saves, and the shape a program is read from.
//!
//! Deliberately small: enough structure to be a real graph — several nodes, edges between them,
//! edges from the program's own inputs — and nothing that belongs to any particular editor. A
//! builder with a richer document maps onto this by ignoring the rest.

use dsrust::serde_json::Value;
use serde::{Deserialize, Serialize};

/// Where one input of one node comes from. This is the edge a user draws.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "from", rename_all = "snake_case")]
pub enum Source {
    /// A field of the program's own inputs — an edge from the canvas border.
    Input { field: String },
    /// A field of an earlier node's output — an edge between two boxes.
    Node { node: String, field: String },
}

/// How a node states what it asks for.
///
/// A builder that writes signatures as strings uses [`Written`](Declared::Written). One whose
/// canvas has a node *per field* — with a type on each, and edges into them — uses
/// [`Fields`](Declared::Fields), which is the shape that cannot be spelled as a string at all once
/// a field carries a custom type.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Declared {
    /// `"subject -> angle"`, exactly as dsrust parses it.
    Written { signature: String },
    /// The fields themselves, which is what a field-node canvas has.
    Fields {
        inputs: Vec<Field>,
        outputs: Vec<Field>,
    },
}

/// One field node: its name, what type it carries, and the prose shown beside it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Field {
    pub name: String,
    /// `str`, `bool`, `int`, `float`, `reasoning`, or anything else — which is read as a
    /// structured field carrying that annotation, the way a custom type reaches dspy.
    #[serde(default = "str_type")]
    pub r#type: String,
    /// The description the prompt carries for this field.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
}

fn str_type() -> String {
    "str".to_owned()
}

/// One box on the canvas: what it asks for, and what feeds each of its inputs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Node {
    pub id: String,
    #[serde(flatten)]
    pub declared: Declared,
    /// The instructions the node starts with. An optimizer rewrites these, which is the whole
    /// point of compiling a graph rather than only running it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    /// One entry per input the signature declares.
    pub inputs: Vec<Wire>,
}

/// One input, and the edge that feeds it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Wire {
    pub name: String,
    #[serde(flatten)]
    pub source: Source,
}

/// One field of the program's own answer.
///
/// A canvas's outputs are a *layer*, not a node: each output field names itself, and takes its
/// value from one field of one module. A program with a single output is the one-entry case.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Answer {
    /// The name this field carries in the program's answer.
    pub name: String,
    /// The node that produced it.
    pub node: String,
    /// Which of that node's fields.
    pub field: String,
}

/// The whole canvas.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphDocument {
    pub nodes: Vec<Node>,
    /// What the program answers with — one entry per output field.
    pub answers: Vec<Answer>,
}

impl GraphDocument {
    pub fn from_json(text: &str) -> dsrust::anyhow::Result<Self> {
        Ok(dsrust::serde_json::from_str(text)?)
    }

    pub fn to_json(&self) -> String {
        dsrust::serde_json::to_string_pretty(self).unwrap_or_default()
    }

    /// A two-node graph: plan an angle from a subject, then write from the angle.
    pub fn example() -> Self {
        Self {
            nodes: vec![
                Node {
                    id: "plan".to_owned(),
                    declared: Declared::Written {
                        signature: "subject -> angle".to_owned(),
                    },
                    instructions: Some("Pick one angle on the subject.".to_owned()),
                    inputs: vec![Wire {
                        name: "subject".to_owned(),
                        source: Source::Input {
                            field: "subject".to_owned(),
                        },
                    }],
                },
                Node {
                    id: "write".to_owned(),
                    declared: Declared::Fields {
                        inputs: vec![Field {
                            name: "angle".to_owned(),
                            r#type: "str".to_owned(),
                            description: "The angle to write on.".to_owned(),
                        }],
                        outputs: vec![Field {
                            name: "haiku".to_owned(),
                            r#type: "str".to_owned(),
                            description: String::new(),
                        }],
                    },
                    instructions: Some("Write a haiku on that angle.".to_owned()),
                    inputs: vec![Wire {
                        name: "angle".to_owned(),
                        source: Source::Node {
                            node: "plan".to_owned(),
                            field: "angle".to_owned(),
                        },
                    }],
                },
            ],
            answers: vec![Answer {
                name: "haiku".to_owned(),
                node: "write".to_owned(),
                field: "haiku".to_owned(),
            }],
        }
    }
}

/// Values a node produced, addressed the way an edge addresses them.
pub type Produced = Vec<(String, Value)>;
