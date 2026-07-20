//! dspy `baml_adapter._render_type_str`: the compact notation a BAML prompt states a type in.
//!
//! A JSON schema states a type in the vocabulary a validator reads. This states the same type in
//! one a reader follows: braces, one member per line, and whatever the type says about itself as
//! comments above the member it belongs to. Upstream renders it straight off pydantic, so the
//! facts it reads — each member's name, description and alias, and every model's docstring —
//! reach the crate as the reflection on the field's kind, and only the reading of them is here.

use anyhow::{Result, anyhow};
use serde::Deserialize;
use serde_json::Value;

use crate::adapter::python_json::format_field_value;
use crate::signature::{FieldKind, OutField};

/// Python's `#` rather than another language's `//`. Upstream notes that a model follows the
/// comment marker of the language it is being asked to think in.
const COMMENT: &str = "#";

/// One level of nesting, matching upstream's `INDENTATION`.
const INDENT: &str = "  ";

/// Upstream's refusal, which its own tests match on.
const RECURSIVE: &str =
    "BAMLAdapter cannot handle recursive pydantic models, please use a different adapter.";

/// A type as Python reflected it: the annotation itself, and every model it names.
///
/// The models are a table referred to by index rather than a tree, because a type may name one
/// twice or name itself. Whether that is renderable is decided by the walk below, so what
/// crosses is the graph as it really is.
#[derive(Deserialize)]
struct Reflection {
    #[serde(rename = "type")]
    declared: Node,
    models: Vec<Model>,
}

/// One pydantic model: what it says about itself, and the members it declares in order.
#[derive(Deserialize)]
struct Model {
    doc: Option<String>,
    fields: Vec<Field>,
}

/// One member of a model: how it is named here, what it says about itself, the name it is keyed
/// by elsewhere, and its own type.
#[derive(Deserialize)]
struct Field {
    name: String,
    desc: Option<String>,
    alias: Option<String>,
    #[serde(rename = "type")]
    declared: Node,
}

/// One node of an annotation — what upstream's renderer branches on, with nothing yet rendered.
#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Node {
    Str,
    Int,
    Float,
    Bool,
    Model {
        model: usize,
    },
    /// `of` holds the arms that are not `None`, and `optional` says whether one of them was.
    Union {
        of: Vec<Node>,
        optional: bool,
    },
    Literal {
        members: Vec<Value>,
    },
    List {
        of: Box<Node>,
    },
    Dict {
        key: Box<Node>,
        value: Box<Node>,
    },
    /// A type the reflection does not take apart, under the name Python prints for it.
    Named {
        name: String,
    },
}

/// The notation an output field's type is stated in.
///
/// A closed set is the field's type wherever it has one, and upstream spells it as its members.
/// The scalars name themselves. Everything else states the structure Python reflected, or —
/// where nothing reflected it, which is every signature declared in Rust — the type's own name.
pub(super) fn output_type(field: &OutField) -> Result<String> {
    if let Some(values) = &field.values {
        return Ok(joined(
            values.iter().map(|value| quoted(&value.wire_form())),
        ));
    }
    match &field.kind {
        FieldKind::Str => Ok("string".to_owned()),
        FieldKind::Int => Ok("int".to_owned()),
        FieldKind::Float => Ok("float".to_owned()),
        FieldKind::Bool => Ok("boolean".to_owned()),
        FieldKind::Json(json) => match &json.reflection {
            Some(reflection) => reflected(reflection)?.render(),
            None => Ok(json.annotation.clone()),
        },
    }
}

/// Whether a field's declared type is a record — a model with members of its own — rather than a
/// container, a scalar, or a type nothing reflected.
pub(super) fn is_record(kind: &FieldKind) -> bool {
    let FieldKind::Json(json) = kind else {
        return false;
    };
    json.reflection
        .as_ref()
        .and_then(|reflection| reflected(reflection).ok())
        .is_some_and(|reflection| matches!(reflection.declared, Node::Model { .. }))
}

fn reflected(reflection: &Value) -> Result<Reflection> {
    Reflection::deserialize(reflection).map_err(|error| anyhow!("bad type reflection: {error}"))
}

impl Reflection {
    /// The whole annotation, at the depth an output field's own line starts from.
    fn render(&self) -> Result<String> {
        self.node(&self.declared, 0, &mut Vec::new())
    }

    /// One node. `seen` carries the models already opened, which is what makes a model that
    /// reaches itself an error rather than an endless walk.
    fn node(&self, node: &Node, indent: usize, seen: &mut Vec<usize>) -> Result<String> {
        match node {
            Node::Str => Ok("string".to_owned()),
            Node::Int => Ok("int".to_owned()),
            Node::Float => Ok("float".to_owned()),
            Node::Bool => Ok("boolean".to_owned()),
            Node::Named { name } => Ok(name.clone()),
            Node::Model { model } => self.model(*model, indent, seen),
            Node::Literal { members } => Ok(joined(
                members
                    .iter()
                    .map(|member| quoted(&format_field_value(member))),
            )),
            Node::Union { of, optional } => {
                let arms = of
                    .iter()
                    .map(|arm| self.node(arm, indent, seen))
                    .collect::<Result<Vec<_>>>()?;
                match optional {
                    true => Ok(format!("{} or null", joined(arms))),
                    false => Ok(joined(arms)),
                }
            }
            Node::List { of } => self.list(of, indent, seen),
            Node::Dict { key, value } => Ok(format!(
                "dict[{}, {}]",
                self.node(key, indent, seen)?,
                self.node(value, indent, seen)?
            )),
        }
    }

    /// A list of models is written as its element's own schema between brackets, so the model
    /// reads the shape of one entry rather than a type name it has to expand for itself. Every
    /// other element type takes the `[]` suffix.
    fn list(&self, element: &Node, indent: usize, seen: &mut Vec<usize>) -> Result<String> {
        let Node::Model { model } = element else {
            return Ok(format!("{}[]", self.node(element, indent, seen)?));
        };
        let entry = self.model(*model, indent + 1, seen)?;
        Ok(format!("[\n{entry}\n{}]", INDENT.repeat(indent)))
    }

    /// dspy `_build_simplified_schema`: a model's members between braces, each under whatever
    /// comment it earns.
    ///
    /// `seen` is never unwound, which is upstream's rule and not an oversight in it: a model
    /// reached twice anywhere in one annotation is refused, not only one that reaches itself.
    fn model(&self, index: usize, indent: usize, seen: &mut Vec<usize>) -> Result<String> {
        let model = self.models.get(index).ok_or_else(|| {
            anyhow!("type reflection names model {index}, which it does not carry")
        })?;
        if seen.contains(&index) {
            return Err(anyhow!(RECURSIVE));
        }
        seen.push(index);
        let current = INDENT.repeat(indent);
        let next = INDENT.repeat(indent + 1);
        // Only the outermost model states its docstring here. A nested one states it above the
        // member that carries it, where the type is named and the braces alone would not say so.
        let mut lines = match indent {
            0 => comment(model.doc.as_deref(), &current),
            _ => Vec::new(),
        };
        lines.push(format!("{current}{{"));
        if model.fields.is_empty() {
            lines.push(format!("{next}{COMMENT} No fields defined"));
        }
        for field in &model.fields {
            lines.extend(self.field_comments(field, &next));
            let declared = self.node(&field.declared, indent + 1, seen)?;
            lines.push(format!("{next}{}: {declared},", field.name));
        }
        lines.push(format!("{current}}}"));
        Ok(lines.join("\n"))
    }

    /// What a member says before its own line: its description, or — having none — the name it
    /// is keyed by elsewhere, so a reply keyed that way is still recognisable. Then the
    /// docstring of the model it carries, which that model's own braces have nowhere to put.
    fn field_comments(&self, field: &Field, indent: &str) -> Vec<String> {
        let alias = stated(field.alias.as_deref()).filter(|alias| *alias != field.name);
        let own = match (stated(field.desc.as_deref()), alias) {
            (Some(desc), _) => Some(format!("{indent}{COMMENT} {desc}")),
            (None, Some(alias)) => Some(format!("{indent}{COMMENT} alias: {alias}")),
            (None, None) => None,
        };
        let carried = self
            .carried(&field.declared)
            .and_then(|model| model.doc.as_deref());
        own.into_iter().chain(comment(carried, indent)).collect()
    }

    /// The model a member's own annotation is, once an optional wrapper around it is unwrapped.
    /// Upstream looks exactly one level deep, so a model inside a list or a wider union states
    /// nothing above the member.
    fn carried(&self, node: &Node) -> Option<&Model> {
        let node = match node {
            Node::Union { of, .. } if of.len() == 1 => &of[0],
            node => node,
        };
        match node {
            Node::Model { model } => self.models.get(*model),
            _ => None,
        }
    }
}

/// A docstring as comment lines. A blank line carries nothing and is dropped, and the rest shed
/// the indentation their source file gave them.
fn comment(doc: Option<&str>, indent: &str) -> Vec<String> {
    doc.into_iter()
        .flat_map(|doc| doc.trim().lines().map(str::trim).collect::<Vec<_>>())
        .filter(|line| !line.is_empty())
        .map(|line| format!("{indent}{COMMENT} {line}"))
        .collect()
}

/// A member of a closed set, which upstream quotes whatever Python type it really is.
fn quoted(member: &str) -> String {
    format!("\"{member}\"")
}

/// The alternatives of a union or a closed set, in the word a reader parses more readily than
/// the pipe a type checker would use.
fn joined(alternatives: impl IntoIterator<Item = String>) -> String {
    alternatives.into_iter().collect::<Vec<_>>().join(" or ")
}

/// The text where there is any, since upstream treats an empty description as no description.
fn stated(text: Option<&str>) -> Option<&str> {
    text.filter(|text| !text.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signature::JsonType;
    use serde_json::json;

    /// An output field carrying exactly the reflection `bridge/python/rust_adapter.py` sends.
    fn reflected_output(reflection: Value) -> OutField {
        OutField {
            name: "output".into(),
            desc: String::new(),
            kind: FieldKind::Json(JsonType {
                annotation: "Reflected".into(),
                descriptions: Vec::new(),
                reflection: Some(reflection),
            }),
            values: None,
            schema: None,
        }
    }

    fn model(doc: Value, fields: Value) -> Value {
        json!({ "doc": doc, "fields": fields })
    }

    fn field(name: &str, declared: Value) -> Value {
        json!({ "name": name, "desc": null, "alias": null, "type": declared })
    }

    /// `PatientAddress` from upstream's test file, as Python reflects it.
    fn address() -> Value {
        model(
            json!("Patient Address model docstring"),
            json!([
                field("street", json!({ "kind": "str" })),
                field("city", json!({ "kind": "str" })),
                field(
                    "country",
                    json!({ "kind": "literal", "members": ["US", "CA"] })
                ),
            ]),
        )
    }

    /// `PatientDetails`, whose address is optional and whose name has a description.
    fn details(address_index: usize) -> Value {
        model(
            json!(
                "\n    Patient Details model docstring\n    Multiline docstring support test\n    "
            ),
            json!([
                { "name": "name", "desc": "Full name of the patient", "alias": null,
                  "type": { "kind": "str" } },
                field("age", json!({ "kind": "int" })),
                field(
                    "address",
                    json!({
                        "kind": "union",
                        "of": [{ "kind": "model", "model": address_index }],
                        "optional": true,
                    })
                ),
            ]),
        )
    }

    fn patient_details() -> Value {
        json!({ "type": { "kind": "model", "model": 0 }, "models": [details(1), address()] })
    }

    /// The bytes `dspy.adapters.baml_adapter.BAMLAdapter().format_field_structure` writes for a
    /// `PatientDetails` output, with the surrounding section removed.
    #[test]
    fn a_model_states_its_docstring_then_each_member_under_its_own_comment() {
        assert_eq!(
            output_type(&reflected_output(patient_details())).expect("renders"),
            "# Patient Details model docstring\n\
             # Multiline docstring support test\n\
             {\n\
             \x20 # Full name of the patient\n\
             \x20 name: string,\n\
             \x20 age: int,\n\
             \x20 # Patient Address model docstring\n\
             \x20 address:   {\n\
             \x20   street: string,\n\
             \x20   city: string,\n\
             \x20   country: \"US\" or \"CA\",\n\
             \x20 } or null,\n\
             }"
        );
    }

    /// Upstream's `ComplexNestedModel`: a member that is a model, a list, and a mapping.
    #[test]
    fn nesting_indents_a_member_model_and_names_a_containers_contents() {
        let complex = model(
            json!("Complex model docstring"),
            json!([
                { "name": "id", "desc": "Unique identifier", "alias": null,
                  "type": { "kind": "int" } },
                field("details", json!({ "kind": "model", "model": 1 })),
                field("tags", json!({ "kind": "list", "of": { "kind": "str" } })),
                field(
                    "metadata",
                    json!({ "kind": "dict", "key": { "kind": "str" }, "value": { "kind": "str" } })
                ),
            ]),
        );
        let reflection = json!({ "type": { "kind": "model", "model": 0 },
                    "models": [complex, details(2), address()] });
        assert_eq!(
            output_type(&reflected_output(reflection)).expect("renders"),
            "# Complex model docstring\n\
             {\n\
             \x20 # Unique identifier\n\
             \x20 id: int,\n\
             \x20 # Patient Details model docstring\n\
             \x20 # Multiline docstring support test\n\
             \x20 details:   {\n\
             \x20   # Full name of the patient\n\
             \x20   name: string,\n\
             \x20   age: int,\n\
             \x20   # Patient Address model docstring\n\
             \x20   address:     {\n\
             \x20     street: string,\n\
             \x20     city: string,\n\
             \x20     country: \"US\" or \"CA\",\n\
             \x20   } or null,\n\
             \x20 },\n\
             \x20 tags: string[],\n\
             \x20 metadata: dict[string, string],\n\
             }"
        );
    }

    /// Upstream's `ModelWithLists`: a list of models opens brackets around the entry's schema,
    /// where a list of scalars only suffixes the element type.
    #[test]
    fn a_list_of_models_brackets_one_entrys_schema() {
        let with_lists = model(
            json!(null),
            json!([
                { "name": "items", "desc": "List of patient addresses", "alias": null,
                  "type": { "kind": "list", "of": { "kind": "model", "model": 1 } } },
                field("scores", json!({ "kind": "list", "of": { "kind": "float" } })),
            ]),
        );
        let reflection = json!({ "type": { "kind": "model", "model": 0 },
                                 "models": [with_lists, address()] });
        assert_eq!(
            output_type(&reflected_output(reflection)).expect("renders"),
            "{\n\
             \x20 # List of patient addresses\n\
             \x20 items: [\n\
             \x20   {\n\
             \x20     street: string,\n\
             \x20     city: string,\n\
             \x20     country: \"US\" or \"CA\",\n\
             \x20   }\n\
             \x20 ],\n\
             \x20 scores: float[],\n\
             }"
        );
    }

    /// Upstream's `ModelWithAliasNoDescription`: the alias stands in for a description, and a
    /// member with neither says nothing at all.
    #[test]
    fn an_alias_is_stated_only_where_the_member_describes_itself_no_other_way() {
        let aliased = model(
            json!(null),
            json!([
                { "name": "internal_field", "desc": null, "alias": "public_name",
                  "type": { "kind": "str" } },
                field("regular_field", json!({ "kind": "int" })),
                { "name": "field_with_description", "desc": "This field has a description",
                  "alias": "desc_field", "type": { "kind": "str" } },
            ]),
        );
        let reflection = json!({ "type": { "kind": "model", "model": 0 }, "models": [aliased] });
        assert_eq!(
            output_type(&reflected_output(reflection)).expect("renders"),
            "{\n\
             \x20 # alias: public_name\n\
             \x20 internal_field: string,\n\
             \x20 regular_field: int,\n\
             \x20 # This field has a description\n\
             \x20 field_with_description: string,\n\
             }"
        );
    }

    /// An alias that only repeats the member's own name tells the model nothing it cannot see.
    #[test]
    fn an_alias_equal_to_the_member_name_states_nothing() {
        let same = model(
            json!(null),
            json!([{ "name": "field", "desc": null, "alias": "field", "type": { "kind": "str" } }]),
        );
        let reflection = json!({ "type": { "kind": "model", "model": 0 }, "models": [same] });
        assert_eq!(
            output_type(&reflected_output(reflection)).expect("renders"),
            "{\n  field: string,\n}"
        );
    }

    /// Upstream's `ProblematicModel`: a type pydantic cannot describe still has a name, and
    /// upstream prints it rather than leaving the member blank.
    #[test]
    fn a_type_with_no_structure_is_stated_by_its_name() {
        let problematic = model(
            json!(null),
            json!([field("field", json!({ "kind": "named", "name": "object" }))]),
        );
        let reflection =
            json!({ "type": { "kind": "model", "model": 0 }, "models": [problematic] });
        assert_eq!(
            output_type(&reflected_output(reflection)).expect("renders"),
            "{\n  field: object,\n}"
        );
    }

    #[test]
    fn a_model_with_no_members_says_so() {
        let empty = json!({ "type": { "kind": "model", "model": 0 },
                            "models": [model(json!(null), json!([]))] });
        assert_eq!(
            output_type(&reflected_output(empty)).expect("renders"),
            "{\n  # No fields defined\n}"
        );
    }

    /// Upstream's `CircularModel`, whose refusal its own test matches on.
    #[test]
    fn a_model_that_reaches_itself_is_refused() {
        let circular = model(
            json!(null),
            json!([
                field("name", json!({ "kind": "str" })),
                field("field", json!({ "kind": "model", "model": 0 })),
            ]),
        );
        let reflection = json!({ "type": { "kind": "model", "model": 0 }, "models": [circular] });
        let error = output_type(&reflected_output(reflection)).expect_err("refuses");
        assert!(
            error
                .to_string()
                .contains("BAMLAdapter cannot handle recursive pydantic models"),
            "got: {error}"
        );
    }

    /// Upstream tracks the models it has opened across the whole annotation and never unwinds,
    /// so two members of one type are refused as readily as a model that contains itself.
    #[test]
    fn one_model_reached_by_two_members_is_refused_the_same_way() {
        let pair = model(
            json!(null),
            json!([
                field("home", json!({ "kind": "model", "model": 1 })),
                field("work", json!({ "kind": "model", "model": 1 })),
            ]),
        );
        let reflection =
            json!({ "type": { "kind": "model", "model": 0 }, "models": [pair, address()] });
        assert!(output_type(&reflected_output(reflection)).is_err());
    }

    #[test]
    fn a_union_names_every_arm_and_says_when_one_was_none() {
        let both = json!({
            "type": {
                "kind": "union",
                "of": [{ "kind": "str" }, { "kind": "int" }],
                "optional": true,
            },
            "models": [],
        });
        assert_eq!(
            output_type(&reflected_output(both)).expect("renders"),
            "string or int or null"
        );
    }

    /// A closed set is quoted whatever Python type its members are, and each is spelled the way
    /// Python's own `str` would spell it — `True`, never `true`.
    #[test]
    fn closed_set_members_are_quoted_in_pythons_spelling() {
        let mixed = json!({
            "type": { "kind": "literal", "members": [1, true, "text"] },
            "models": [],
        });
        assert_eq!(
            output_type(&reflected_output(mixed)).expect("renders"),
            "\"1\" or \"True\" or \"text\""
        );
    }

    #[test]
    fn a_scalar_field_names_itself_and_a_closed_set_replaces_the_kind() {
        let scalar = |kind: FieldKind| OutField {
            name: "output".into(),
            desc: String::new(),
            kind,
            values: None,
            schema: None,
        };
        for (kind, expected) in [
            (FieldKind::Str, "string"),
            (FieldKind::Int, "int"),
            (FieldKind::Float, "float"),
            (FieldKind::Bool, "boolean"),
        ] {
            assert_eq!(output_type(&scalar(kind)).expect("renders"), expected);
        }
        let mut closed = scalar(FieldKind::Str);
        closed.values = Some(vec!["red".into(), "blue".into()]);
        assert_eq!(
            output_type(&closed).expect("renders"),
            "\"red\" or \"blue\""
        );
    }

    /// Every signature declared in Rust reaches here without a reflection, and a type name is
    /// what upstream falls back to for a type it cannot take apart either.
    #[test]
    fn a_type_nothing_reflected_is_stated_by_its_annotation() {
        let opaque = OutField {
            name: "output".into(),
            desc: String::new(),
            kind: FieldKind::Json(JsonType::plain("list[Idea]")),
            values: None,
            schema: None,
        };
        assert_eq!(output_type(&opaque).expect("renders"), "list[Idea]");
        assert!(!is_record(&opaque.kind));
    }

    #[test]
    fn only_a_model_counts_as_a_record() {
        let kind = |reflection: Value| {
            FieldKind::Json(JsonType {
                annotation: "Reflected".into(),
                descriptions: Vec::new(),
                reflection: Some(reflection),
            })
        };
        assert!(is_record(&kind(patient_details())));
        assert!(!is_record(&kind(json!({
            "type": { "kind": "list", "of": { "kind": "model", "model": 0 } },
            "models": [address()],
        }))));
        assert!(!is_record(&FieldKind::Str));
    }
}
