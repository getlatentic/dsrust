//! The type a signature field carries: the kinds themselves, the closed sets that narrow one,
//! how each is spelled for the model, and how a reply value is read back into it.

use anyhow::{Result, anyhow};
use serde_json::{Value, json};

/// The wire type of a field. It decides the schema type, the annotation prompts carry next
/// to the field name, and how a reply value coerces before validation. `Json` covers every
/// non-scalar Rust type — `Vec<String>`, user structs, `Vec<Struct>` — carried as JSON.
///
/// `Str` is the default because an undeclared dspy field is a `str`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum FieldKind {
    #[default]
    Str,
    Bool,
    Int,
    Float,
    /// The scalar kinds name themselves in Python; a non-scalar does not, so the Python type
    /// dspy would print travels with the variant — `dict[str, Any]`, `list[str]`.
    Json(JsonType),
    /// A field whose value is one of a named type's members — Python's `enum.Enum`.
    ///
    /// Distinct from a closed set spelled `Literal[...]`: dspy prints the type's own name as the
    /// annotation and asks the model to produce one of the members' *values*, where a `Literal`
    /// prints its members and asks for an exact match on the spelling.
    Enum(String),
    /// dspy 3.3's `Reasoning`: a str-like custom type. It renders and coerces exactly as `Str`
    /// (`get_annotation_name` returns "str", its `format` yields the raw content, so it carries no
    /// schema note), but its annotation *is not* the `str` type, so the output-requirement hint
    /// still fires — `(must be formatted as a valid Python str)`. That one difference is why it is
    /// a kind of its own rather than a `Str`.
    Reasoning,
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
    /// Each custom type the annotation names. Empty for a plain structure, which says nothing
    /// about itself.
    pub descriptions: Vec<TypeDescription>,
    /// The annotation's own structure, as Python reflected it: the members each model declares,
    /// their names, descriptions and aliases, and every model's docstring.
    ///
    /// A JSON schema is the same type through a lossy lens — it keys a property by the alias or
    /// by the name but never carries both, and has no spelling at all for `object`, a dict's key
    /// type or a `datetime` — so an adapter that states the declared type itself rather than a
    /// schema of it reads this. Absent wherever nothing reflected the type, which is every
    /// signature declared in Rust.
    pub reflection: Option<Value>,
}

/// What one custom type says about itself on its field's prompt line.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TypeDescription {
    /// The name dspy prints for the type — `Code`, `Citations`.
    pub name: String,
    /// The prose the type states about itself.
    pub text: String,
    /// Whether this prose already says what a JSON schema would, so the field states its
    /// contract once instead of twice.
    ///
    /// dspy sets this for `dspy.Code` alone, whose description spells out the markdown block it
    /// expects while its schema block runs to hundreds of characters saying the same thing. The
    /// property is the type's, not the field's: every field annotated with such a type drops the
    /// schema note, and none annotated with any other type does.
    pub replaces_schema: bool,
}

impl JsonType {
    /// A structure whose annotation carries no custom-type prose.
    pub fn plain(annotation: impl Into<String>) -> Self {
        Self {
            annotation: annotation.into(),
            descriptions: Vec::new(),
            reflection: None,
        }
    }

    /// The same, carrying the structure of the declared type.
    ///
    /// What an adapter that *states* a type rather than a schema of it needs. The derive fills
    /// this from `schemars`, which is the only description of a Rust type available to it.
    pub fn reflected(annotation: impl Into<String>, reflection: Value) -> Self {
        Self {
            annotation: annotation.into(),
            descriptions: Vec::new(),
            reflection: Some(reflection),
        }
    }
}

impl FieldKind {
    /// A non-scalar whose Python type this crate cannot name, and whose structure nothing
    /// described. `FieldKind::reflected_json` is what the derive reaches for instead wherever a
    /// `schemars` schema exists, which is every field it generates.
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
            // dspy schemas an enum by its members, which the field carries beside this.
            FieldKind::Enum(_) => Some("string"),
            FieldKind::Reasoning => Some("string"),
        }
    }

    /// Whether this is the plain `str` annotation, which is what decides the output-requirement
    /// hint: dspy asks `annotation is not str`, so every other kind — including one that *prints*
    /// `str`, as [`FieldKind::Reasoning`] does — earns the hint. A closed set is checked beside
    /// this, since `Literal[...]` is not `str` either.
    pub(crate) fn is_plain_str(&self) -> bool {
        matches!(self, FieldKind::Str)
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
            FieldKind::Enum(name) => name,
            // dspy's `get_annotation_name` returns "str" for `Reasoning`, keeping the old
            // `ChainOfThought` wording where the reasoning field read as a plain string.
            FieldKind::Reasoning => "str",
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
    /// A member Python prints as itself rather than as a literal — an enum member, whose `str`
    /// is `Colour.RED`. It is not a string: quoting it would tell the model to answer
    /// `'Colour.RED'` where dspy asks for `Colour.RED`.
    Bare(String),
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
            LiteralValue::Bare(text) => text.clone(),
        }
    }

    /// The text a marker-path reply carries for this member. Every value crosses that path as
    /// text, so this is what a closed set is checked against and what prompts list.
    pub(crate) fn wire_form(&self) -> String {
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
            // A reply names the member, so that name is the value that crosses.
            LiteralValue::Bare(text) => json!(text),
        }
    }

    pub(super) fn schema_type(&self) -> &'static str {
        match self {
            LiteralValue::Str(_) => "string",
            LiteralValue::Int(_) => "integer",
            LiteralValue::Bool(_) => "boolean",
            LiteralValue::Bare(_) => "string",
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
    match (kind, values) {
        // An enum's members are its closed set, but dspy prints the type that named them
        // rather than the members — `Status`, not `Literal['active', 'done']`.
        (FieldKind::Enum(name), _) => name.clone(),
        (_, Some(values)) => literal_annotation(values),
        (_, None) => kind.annotation().to_owned(),
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

pub(crate) fn coerce_value(kind: &FieldKind, name: &str, value: &mut Value) -> Result<()> {
    match kind {
        // `Reasoning` carries its content as text, exactly as a `Str` does.
        FieldKind::Str | FieldKind::Reasoning => Ok(()),
        FieldKind::Bool => coerce_bool(name, value),
        FieldKind::Int => coerce_int(name, value),
        FieldKind::Float => coerce_float(name, value),
        FieldKind::Json(_) => coerce_json(name, value),
        // A member reaches the marker path as the text of its value, which is what the model
        // was asked for; naming the member it belongs to is the declared type's job.
        FieldKind::Enum(_) => Ok(()),
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

    #[test]
    fn a_member_python_prints_as_itself_is_not_quoted() {
        // `Literal[Colour.RED]` reaches the model as `Colour.RED`. Quoting it would ask for
        // `'Colour.RED'`, which is a different answer.
        let member = LiteralValue::Bare("Colour.RED".to_owned());
        assert_eq!(member.annotation(), "Colour.RED");
        assert_eq!(member.wire_form(), "Colour.RED");
        // A plain string in the same position keeps its quotes, which is the distinction.
        assert_eq!(LiteralValue::Str("red".to_owned()).annotation(), "'red'");
    }

    #[test]
    fn an_enum_prints_its_type_and_asks_for_a_member_value() {
        // dspy names the type in the annotation and lists the members' values in the note,
        // where a `Literal` prints the members themselves and demands an exact match.
        let kind = FieldKind::Enum("Status".to_owned());
        let members = vec![
            LiteralValue::Str("active".to_owned()),
            LiteralValue::Str("done".to_owned()),
        ];
        assert_eq!(annotation_of(Some(&members), &kind), "Status");
        assert_eq!(
            annotation_of(Some(&members), &FieldKind::Str),
            "Literal['active', 'done']"
        );
    }
}
