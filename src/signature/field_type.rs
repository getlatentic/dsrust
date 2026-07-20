//! The type a signature field carries: the kinds themselves, the closed sets that narrow one,
//! how each is spelled for the model, and how a reply value is read back into it.

use anyhow::{Result, anyhow};
use serde_json::{Value, json};

/// The wire type of a field. It decides the schema type, the annotation prompts carry next
/// to the field name, and how a reply value coerces before validation. `Json` covers every
/// non-scalar Rust type — `Vec<String>`, user structs, `Vec<Struct>` — carried as JSON.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FieldKind {
    Str,
    Bool,
    Int,
    Float,
    /// The scalar kinds name themselves in Python; a non-scalar does not, so the Python type
    /// dspy would print travels with the variant — `dict[str, Any]`, `list[str]`.
    Json(JsonType),
}

/// A non-scalar field's Python type, and the prose any custom type in it contributes.
///
/// dspy reads a description off the *annotation*, not the field: `Type.description()` belongs to
/// `dspy.Code` itself, and every field annotated with it earns the same line. An annotation can
/// name more than one custom type, so the descriptions are a list in the order dspy extracts
/// them.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct JsonType {
    /// The annotation prompts carry — `dict[str, Any]`, `Citations`, `list[str]`.
    pub annotation: String,
    /// Each custom type the annotation names, as its printed name and its description. Empty
    /// for a plain structure, which contributes no prose.
    pub descriptions: Vec<(String, String)>,
}

impl JsonType {
    /// A structure whose annotation carries no custom-type prose.
    pub fn plain(annotation: impl Into<String>) -> Self {
        Self {
            annotation: annotation.into(),
            descriptions: Vec::new(),
        }
    }
}

impl FieldKind {
    /// A non-scalar whose Python type this crate cannot name. The derive maps every such Rust
    /// type here, so prompts print `json` where dspy would print `list[Idea]`.
    pub fn opaque_json() -> Self {
        FieldKind::Json(JsonType::plain("json"))
    }

    /// The JSON-schema type name for scalar kinds; a `Json` field has no single type name —
    /// it carries its full nested schema on the [`OutField`](super::OutField) instead.
    pub fn schema_type(&self) -> Option<&'static str> {
        match self {
            FieldKind::Str => Some("string"),
            FieldKind::Bool => Some("boolean"),
            FieldKind::Int => Some("integer"),
            FieldKind::Float => Some("number"),
            FieldKind::Json(_) => None,
        }
    }

    /// The name the model is shown for this field's type. dspy prints Python's own type names
    /// through `get_annotation_name`, so a model tuned on DSPy prompts sees `int`, not the
    /// JSON Schema spelling `integer`. The conformance fixtures pin this.
    pub fn annotation(&self) -> &str {
        match self {
            FieldKind::Str => "str",
            FieldKind::Bool => "bool",
            FieldKind::Int => "int",
            FieldKind::Float => "float",
            FieldKind::Json(json) => &json.annotation,
        }
    }
}

/// One member of a closed set. Python's `Literal` admits strings, integers and booleans and
/// spells each differently, so a member keeps its type rather than flattening to text: that
/// would print `Literal[True]` as `Literal['True']` and put quotes round every number.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LiteralValue {
    Str(String),
    Int(i64),
    Bool(bool),
}

impl LiteralValue {
    /// The member as it appears inside `Literal[...]`: dspy's
    /// `_quoted_string_for_literal_type_annotation` for a string, Python's own spelling for
    /// the rest — `True`, never `true`.
    fn annotation(&self) -> String {
        match self {
            LiteralValue::Str(text) => quoted_member(text),
            LiteralValue::Int(number) => number.to_string(),
            LiteralValue::Bool(true) => "True".to_owned(),
            LiteralValue::Bool(false) => "False".to_owned(),
        }
    }

    /// The text a marker-path reply carries for this member. Every value crosses that path as
    /// text, so this is what a closed set is checked against and what prompts list.
    pub(super) fn wire_form(&self) -> String {
        match self {
            LiteralValue::Str(text) => text.clone(),
            typed => typed.annotation(),
        }
    }

    pub(super) fn to_json(&self) -> Value {
        match self {
            LiteralValue::Str(text) => json!(text),
            LiteralValue::Int(number) => json!(number),
            LiteralValue::Bool(flag) => json!(flag),
        }
    }

    pub(super) fn schema_type(&self) -> &'static str {
        match self {
            LiteralValue::Str(_) => "string",
            LiteralValue::Int(_) => "integer",
            LiteralValue::Bool(_) => "boolean",
        }
    }
}

impl From<&str> for LiteralValue {
    fn from(text: &str) -> Self {
        LiteralValue::Str(text.to_owned())
    }
}

impl From<String> for LiteralValue {
    fn from(text: String) -> Self {
        LiteralValue::Str(text)
    }
}

/// dspy's `get_annotation_name` over a `Literal`: the members, spelled the way Python spells
/// them, inside `Literal[...]`.
fn literal_annotation(values: &[LiteralValue]) -> String {
    let members: Vec<String> = values.iter().map(LiteralValue::annotation).collect();
    format!("Literal[{}]", members.join(", "))
}

/// The members as a reply would spell them, for prompts and for validation feedback.
pub(crate) fn wire_forms(values: &[LiteralValue], separator: &str) -> String {
    let forms: Vec<String> = values.iter().map(LiteralValue::wire_form).collect();
    forms.join(separator)
}

/// A closed set is the field's type where there is one: dspy spells it `Literal['a', 'b']`
/// and prints that as the annotation, rather than a note sitting beside the kind.
pub(super) fn annotation_of(values: Option<&Vec<LiteralValue>>, kind: &FieldKind) -> String {
    match values {
        Some(values) => literal_annotation(values),
        None => kind.annotation().to_owned(),
    }
}

/// dspy's `_quoted_string_for_literal_type_annotation`: single quotes, unless the value holds
/// one and no double quote, in which case double quotes avoid the escape. A value carrying
/// both styles escapes its single quotes.
fn quoted_member(value: &str) -> String {
    match (value.contains('\''), value.contains('"')) {
        (true, false) => format!("\"{value}\""),
        (true, true) => format!("'{}'", value.replace('\'', "\\'")),
        _ => format!("'{value}'"),
    }
}

pub(super) fn coerce_value(kind: &FieldKind, name: &str, value: &mut Value) -> Result<()> {
    match kind {
        FieldKind::Str => Ok(()),
        FieldKind::Bool => coerce_bool(name, value),
        FieldKind::Int => coerce_int(name, value),
        FieldKind::Float => coerce_float(name, value),
        FieldKind::Json(_) => coerce_json(name, value),
    }
}

/// Either case of the two keywords, because the crate asks the model for a bool in Python's
/// spelling and renders one the same way; a reply that echoes what it was shown has to parse.
fn coerce_bool(name: &str, value: &mut Value) -> Result<()> {
    if value.is_boolean() {
        return Ok(());
    }
    let parsed = value
        .as_str()
        .and_then(|text| match text.trim().to_ascii_lowercase().as_str() {
            "true" => Some(true),
            "false" => Some(false),
            _ => None,
        });
    match parsed {
        Some(parsed) => {
            *value = Value::Bool(parsed);
            Ok(())
        }
        None => Err(anyhow!("{name} must be true or false, got {value}")),
    }
}

fn coerce_int(name: &str, value: &mut Value) -> Result<()> {
    if value.as_i64().is_some() {
        return Ok(());
    }
    if let Some(text) = value.as_str()
        && let Ok(parsed) = text.trim().parse::<i64>()
    {
        *value = Value::from(parsed);
        return Ok(());
    }
    Err(anyhow!("{name} must be an integer, got {value}"))
}

/// A native value of any non-string shape passes through; a string parses as JSON, with a
/// surrounding code fence tolerated because marker-path models like to wrap JSON in one.
fn coerce_json(name: &str, value: &mut Value) -> Result<()> {
    let Some(text) = value.as_str() else {
        return Ok(());
    };
    match serde_json::from_str(strip_code_fence(text)) {
        Ok(parsed) => {
            *value = parsed;
            Ok(())
        }
        Err(error) => Err(anyhow!("{name} must be valid JSON: {error}")),
    }
}

/// The content of a ```json ... ``` (or bare ```) block, when the whole text is one fence.
fn strip_code_fence(text: &str) -> &str {
    let trimmed = text.trim();
    let Some(fenced) = trimmed
        .strip_prefix("```")
        .and_then(|rest| rest.strip_suffix("```"))
    else {
        return trimmed;
    };
    fenced.strip_prefix("json").unwrap_or(fenced).trim()
}

/// Any native JSON number counts as a float; JSON cannot spell a non-finite one, so only
/// the string form needs the finiteness check.
fn coerce_float(name: &str, value: &mut Value) -> Result<()> {
    if value.is_number() {
        return Ok(());
    }
    if let Some(text) = value.as_str()
        && let Ok(parsed) = text.trim().parse::<f64>()
        && parsed.is_finite()
    {
        *value = Value::from(parsed);
        return Ok(());
    }
    Err(anyhow!("{name} must be a number, got {value}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The expected strings are upstream's own, from
    /// `test_chat_adapter_quotes_literals_as_expected`: dspy quotes a member with double
    /// quotes only to avoid escaping a single quote it contains.
    #[test]
    fn a_closed_set_quotes_its_members_the_way_dspy_does() {
        for (values, expected) in [
            (
                vec!["one", "two", "three\""],
                "Literal['one', 'two', 'three\"']",
            ),
            (
                vec!["she's here", "okay", "test"],
                "Literal[\"she's here\", 'okay', 'test']",
            ),
            (
                vec!["both\"and'", "another"],
                "Literal['both\"and\\'', 'another']",
            ),
            (vec!["foo", "bar"], "Literal['foo', 'bar']"),
        ] {
            let owned: Vec<LiteralValue> = values.iter().map(|value| (*value).into()).collect();
            assert_eq!(literal_annotation(&owned), expected);
        }
    }

    /// Upstream's fifth case in the same test: a `Literal` may mix types, and Python spells a
    /// bool `True`, not `true`, and a number without quotes.
    #[test]
    fn a_closed_set_spells_non_string_members_the_way_python_does() {
        for (values, expected) in [
            (
                vec![LiteralValue::Int(1), "bar".into()],
                "Literal[1, 'bar']",
            ),
            (
                vec![LiteralValue::Bool(true), LiteralValue::Int(3), "foo".into()],
                "Literal[True, 3, 'foo']",
            ),
            (vec![LiteralValue::Bool(false)], "Literal[False]"),
        ] {
            assert_eq!(literal_annotation(&values), expected);
        }
    }

    /// A reply carries every member as text, so the closed set is checked against that
    /// spelling rather than the annotation's — `3`, not `'3'`.
    #[test]
    fn a_closed_set_checks_replies_against_the_spelling_they_arrive_in() {
        let values = vec![LiteralValue::Int(3), LiteralValue::Bool(true), "foo".into()];
        assert_eq!(wire_forms(&values, ", "), "3, True, foo");
    }
}
