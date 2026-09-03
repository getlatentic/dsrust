//! The JSON schema a Python type annotation stands for, for signatures written as a string.
//!
//! A declared struct hands its schema to schemars. A string signature has no Rust type at all —
//! `"claim, notes -> new_notes: list[str]"` carries the *annotation*, and dspy resolves it to a
//! real Python type and asks pydantic:
//!
//! ```python
//! # signatures/signature.py, _parse_type_node
//! annotation = eval(node, {"typing": typing, **typing.__dict__, **builtins.__dict__})
//! ```
//!
//! So a `list[str]` output prints `{"type": "array", "items": {"type": "string"}}` in its note.
//! Without this it printed nothing: the same field declared as a Rust `Vec<String>` carried its
//! schema and the string spelling did not, which is two prompts for one program.
//!
//! Only the annotations the string form can actually spell are resolved. A name this does not know
//! is a caller's own type, which a string signature cannot name anyway.

use anyhow::{Result, anyhow};
use serde_json::{Map, Value, json};

use super::field_type::JsonType;
use super::{FieldKind, LiteralValue};

/// An annotation upstream would have resolved, or the reason it would not have.
///
/// dspy evaluates the annotation against builtins and `typing`, so a name outside those is a
/// `ValueError: Unknown name: …` — even `Image`, which is dspy's own type. A Rust program has no
/// namespace to register into, so that set is closed here and an unknown name is a mistake rather
/// than a type this port has not learned yet.
///
/// Arity is upstream's too: `dict[str]` is `TypeError: Expected two type arguments`.
pub(crate) fn refuse_unless_resolvable(annotation: &str, field: &str) -> Result<()> {
    if resolvable(annotation) {
        return Ok(());
    }
    Err(anyhow!(
        "Invalid signature format: field '{field}' is annotated '{annotation}', which is not a \
         type. A signature string names Python's own types — `str`, `int`, `list[str]`, \
         `dict[str, int]`, `Optional[T]`, `Literal[...]` — and a type of your own is declared with \
         `#[derive(Signature)]` instead."
    ))
}

/// Whether upstream would have resolved this annotation to a type.
///
/// Distinct from [`schema_for`], which answers a narrower question: what schema to *print*. One of
/// dspy's own types resolves and has no schema this can build from a string, and it can sit
/// anywhere a type can — `list[dspy.Tool]` is what the tools guide writes. Answering resolvability
/// with `schema_for` refused every one of those, which is the shape of an over-tight check: it was
/// right about the leaf and wrong about the tree.
pub(crate) fn resolvable(annotation: &str) -> bool {
    let annotation = annotation.trim();
    if custom_type(annotation).is_some() || scalar(annotation).is_some() {
        return true;
    }
    if let Some(parts) = union_parts(annotation) {
        return parts.iter().all(|part| resolvable(part));
    }
    match split_generic(annotation) {
        // A `Literal`'s parameters are values rather than types, so they are not resolved — only
        // that there is at least one of them.
        Some(("Literal" | "typing.Literal", inside)) => super::parse::split_top_level(inside, ',')
            .iter()
            .any(|member| !member.trim().is_empty()),
        Some((name, inside)) => {
            let name = head_of(name);
            let parameters = super::parse::split_top_level(inside, ',');
            let arity = matches!(
                (name, parameters.len()),
                (
                    "list" | "List" | "set" | "Set" | "frozenset" | "Optional",
                    1
                ) | ("dict" | "Dict", 2)
            ) || matches!(name, "tuple" | "Tuple" | "Union" if !parameters.is_empty());
            arity && parameters.iter().all(|part| resolvable(part))
        }
        None => false,
    }
}

/// The schema for one annotation, or nothing when it names no type this can build.
pub(crate) fn schema_for(annotation: &str) -> Option<Value> {
    Some(super::pydantic::as_dspy_prints_it(shaped(annotation)?))
}

fn shaped(annotation: &str) -> Option<Value> {
    let annotation = annotation.trim();
    if let Some(arms) = union_arms(annotation) {
        return Some(json!({ "anyOf": arms }));
    }
    match scalar(annotation) {
        Some(scalar) => Some(scalar),
        None => container(annotation),
    }
}

/// dspy's own types, which a signature string names through the module — `dspy.History`, not
/// `History`. Measured: the bare name is `ValueError: Unknown name: History` and the dotted one
/// resolves, because upstream evaluates the annotation with `dspy` in scope and nothing else of
/// its own.
///
/// They carry no schema. dspy renders each by its class name and the type states its own contract,
/// which is why `Some(None)` is the answer rather than a schema of one.
const CUSTOM: [&str; 10] = [
    "Audio",
    "Code",
    "File",
    "History",
    "Image",
    "Reasoning",
    "Tool",
    "ToolCallResults",
    "ToolCalls",
    "Type",
];

/// The field one of dspy's own types becomes, description and all.
///
/// The rendering machinery has always been there — a `JsonType` carries `descriptions`, and the
/// prompt writes `Type description of X: …` from them — and `#[derive(Signature)]` fills it in by
/// asking the Rust type through [`TypeProbe`](crate::__macro_support::TypeProbe). The *string*
/// form could not: it has a name and no type, and a name cannot be asked anything.
///
/// So the name is dispatched to the type here, once. Exhaustive over [`CUSTOM`] by construction —
/// the match has no fallback arm that could silently swallow a new entry — and `CUSTOM` is held
/// against dspy's own roster of `Type` subclasses by the examples suite.
///
/// `Reasoning` is the odd one and upstream's own oddity: it is a str-like type whose
/// `get_annotation_name` answers `str`, so a field of it renders exactly as a plain string does.
pub(crate) fn custom_field(name: &str) -> (FieldKind, Option<Vec<LiteralValue>>) {
    use crate::adapter::Type;

    let field = |descriptions: Vec<crate::signature::TypeDescription>| {
        (
            FieldKind::Json(JsonType {
                annotation: name.to_owned(),
                descriptions,
                ..Default::default()
            }),
            None,
        )
    };
    let from_type = |description: Option<crate::signature::TypeDescription>| {
        field(description.into_iter().collect())
    };
    match name {
        // Upstream renders this as `str`, so it is one here.
        "Reasoning" => (FieldKind::Reasoning, None),
        "Code" => from_type(<crate::Code as Type>::description()),
        // The one that is not a `Type`: its description is an inherent method.
        "ToolCalls" => field(vec![crate::signature::TypeDescription {
            name: "ToolCalls".to_owned(),
            text: crate::ToolCalls::description().to_owned(),
            // dspy prints this type's schema *as well as* its prose, unlike `Code`.
            replaces_schema: false,
        }]),
        "Audio" => from_type(<crate::Audio as Type>::description()),
        "Image" => from_type(<crate::Image as Type>::description()),
        "File" => from_type(<crate::File as Type>::description()),
        // Every other name `CUSTOM` holds. Listed rather than defaulted, so adding one without
        // deciding what it renders is a compile error rather than an empty field.
        "History" | "Tool" | "Type" | "ToolCallResults" => field(Vec::new()),
        other => unreachable!("{other} is in CUSTOM and has no rendering"),
    }
}

/// The schema one of dspy's own types prints in a field's note, when it prints one.
///
/// Separate from [`custom_field`] because the schema lives on the *field* rather than on its kind,
/// and because the two answers differ per type: `Code` states its contract in prose and prints no
/// schema, `ToolCalls` prints both.
pub(crate) fn custom_schema(name: &str) -> Option<Value> {
    match name {
        "ToolCalls" => Some(crate::ToolCalls::output_schema()),
        "Audio" => Some(crate::Audio::output_schema()),
        "File" => Some(crate::File::output_schema()),
        "Image" => Some(crate::Image::output_schema()),
        // `Code` states its contract in prose instead, and the rest print none.
        _ => None,
    }
}

/// The bare name of a `dspy.X` annotation, when it is one.
pub(crate) fn custom_type(annotation: &str) -> Option<&str> {
    let name = annotation.trim().strip_prefix("dspy.")?;
    CUSTOM.contains(&name).then_some(name)
}

fn scalar(annotation: &str) -> Option<Value> {
    match annotation.strip_prefix("typing.").unwrap_or(annotation) {
        "str" => Some(json!({ "type": "string" })),
        "int" => Some(json!({ "type": "integer" })),
        "float" => Some(json!({ "type": "number" })),
        "bool" => Some(json!({ "type": "boolean" })),
        "bytes" => Some(json!({ "type": "string", "format": "binary" })),
        // pydantic states no constraint at all for `Any`. It writes the bare `true` in exactly one
        // position — a mapping's value — and the empty schema everywhere else, which [`container`]
        // is where that is decided.
        "Any" | "typing.Any" => Some(json!({})),
        "None" | "NoneType" => Some(json!({ "type": "null" })),
        // A container named without its parameters: upstream resolves the builtin and pydantic
        // schemas it with nothing said about what it holds.
        "list" | "List" | "tuple" | "Tuple" => Some(json!({ "type": "array", "items": {} })),
        "set" | "Set" | "frozenset" => {
            Some(json!({ "type": "array", "items": {}, "uniqueItems": true }))
        }
        "dict" | "Dict" => Some(json!({ "type": "object", "additionalProperties": true })),
        _ => None,
    }
}

/// The head of a printed generic, with the spellings dspy prints for one type folded together:
/// the `typing.` qualifier dropped, and `UnionType` — what `get_annotation_name` prints for a PEP
/// 604 `A | B`, since that is the name of its `types.UnionType` — read as the `Union` it is.
pub(super) fn head_of(name: &str) -> &str {
    match name.strip_prefix("typing.").unwrap_or(name) {
        "UnionType" => "Union",
        other => other,
    }
}

fn container(annotation: &str) -> Option<Value> {
    let (name, inside) = split_generic(annotation)?;
    let name = head_of(name);
    let parameters = super::parse::split_top_level(inside, ',');
    let resolved: Option<Vec<Value>> = parameters.iter().map(|part| shaped(part)).collect();
    let resolved = resolved?;
    match (name, resolved.as_slice()) {
        ("list" | "List", [items]) => Some(json!({ "type": "array", "items": items })),
        // A set is an array upstream too, distinguished only by the uniqueness keyword.
        ("set" | "Set" | "frozenset", [items]) => {
            Some(json!({ "type": "array", "items": items, "uniqueItems": true }))
        }
        // Only the value type reaches the schema: JSON object keys are strings already.
        // Upstream states the arity: `dict[str]` is a `TypeError`, not a mapping of unknowns.
        ("dict" | "Dict", [_key, value]) => {
            // An unconstrained value is the one position where pydantic writes `true` rather than
            // the empty schema it writes for `Any` everywhere else.
            let value = match value.as_object().is_some_and(Map::is_empty) {
                true => Value::Bool(true),
                false => (*value).clone(),
            };
            Some(json!({ "type": "object", "additionalProperties": value }))
        }
        ("tuple" | "Tuple", items) => Some(json!({
            "type": "array",
            "maxItems": items.len(),
            "minItems": items.len(),
            "prefixItems": items,
        })),
        ("Optional", [inner]) => Some(json!({ "anyOf": [inner, { "type": "null" }] })),
        ("Union", arms) => Some(json!({ "anyOf": arms })),
        _ => None,
    }
}

/// The arms of a `T | None` union, which is the other spelling of `Optional[T]`.
fn union_arms(annotation: &str) -> Option<Vec<Value>> {
    union_parts(annotation)?
        .iter()
        .map(|part| shaped(part))
        .collect()
}

/// The `|`-separated arms of a union, or nothing when there is only one.
fn union_parts(annotation: &str) -> Option<Vec<&str>> {
    let parts = super::parse::split_top_level(annotation, '|');
    (parts.len() >= 2).then_some(parts)
}

/// The arms of a union however it was spelled — `T | None`, `Optional[T]`, `Union[T, None]`,
/// `UnionType[T, NoneType]` — or nothing when the annotation is not a union. `Optional[T]` names
/// only `T`, so the `None` arm it implies is added, which is what `typing.get_args` reports for it.
pub(crate) fn union_of(annotation: &str) -> Option<Vec<&str>> {
    let annotation = annotation.trim();
    match split_generic(annotation) {
        Some((name, inside)) if head_of(name) == "Optional" => Some(
            super::parse::split_top_level(inside, ',')
                .into_iter()
                .chain(["None"])
                .collect(),
        ),
        Some((name, inside)) if head_of(name) == "Union" => {
            Some(super::parse::split_top_level(inside, ','))
        }
        _ => union_parts(annotation),
    }
}

/// Whether a bare string reaches a field of this annotation as itself.
///
/// dspy's `parse_value` takes a union naming both `None` and `str` straight to pydantic, so the
/// text validates as the string it already is and json-repair never sees it — `42` stays `"42"`.
/// The two must both be named: `int | None` is repaired, and so is `str | int`. `Annotated` is not
/// a union head, so it does not take this path even around a union that would.
pub(crate) fn union_takes_text(annotation: &str) -> bool {
    let Some(arms) = union_of(annotation) else {
        return false;
    };
    let arms: Vec<&str> = arms
        .iter()
        .map(|arm| {
            let arm = arm.trim();
            arm.strip_prefix("typing.").unwrap_or(arm)
        })
        .collect();
    arms.iter().any(|arm| is_none_arm(arm)) && arms.contains(&"str")
}

/// The spellings Python prints for the `None` type.
fn is_none_arm(arm: &str) -> bool {
    matches!(arm, "None" | "NoneType" | "type(None)")
}

/// `list[str]` as `("list", "str")`.
fn split_generic(annotation: &str) -> Option<(&str, &str)> {
    let open = annotation.find('[')?;
    let inside = annotation.strip_suffix(']')?.get(open + 1..)?;
    Some((annotation[..open].trim(), inside))
}

/// dspy 3.3.1's `XMLAdapter._uses_nested_xml`, as far as the spelling decides it: a parametrised
/// `list` or `dict` is nested — a `list` of one of dspy's own types is not — and a builtin, a
/// `Literal`, a tuple, a set, a union of two types or one of dspy's own types is not. One `None`
/// arm is looked through first, as upstream looks through it. A bare name that is none of these is
/// a class the spelling cannot classify, and the answer is `None`: the field's schema decides.
pub(crate) fn nested_xml_shape(annotation: &str) -> Option<bool> {
    let annotation = annotation.trim();
    let annotation = single_non_none_arm(annotation).unwrap_or(annotation);
    if union_parts(annotation).is_some() {
        return Some(false);
    }
    match split_generic(annotation) {
        Some(("list" | "List" | "typing.List", item)) => Some(!is_dspy_type(item.trim())),
        Some(("dict" | "Dict" | "typing.Dict", _)) => Some(true),
        Some(_) => Some(false),
        None if scalar(annotation).is_some() || is_dspy_type(annotation) => Some(false),
        None => None,
    }
}

/// The one arm of a union that is not `None`, however the union was spelled — `T | None`,
/// `Optional[T]`, `Union[T, None]` — or nothing when there is not exactly one.
fn single_non_none_arm(annotation: &str) -> Option<&str> {
    let arms: Vec<&str> = union_of(annotation)?;
    let typed: Vec<&str> = arms
        .iter()
        .map(|arm| arm.trim())
        .filter(|arm| !matches!(*arm, "None" | "NoneType"))
        .collect();
    match typed.as_slice() {
        [only] => Some(only),
        _ => None,
    }
}

/// One of dspy's own types, named bare or through the module.
pub(crate) fn is_dspy_type(annotation: &str) -> bool {
    custom_type(annotation).is_some() || CUSTOM.contains(&annotation)
}

/// dspy's `annotation_allows_none`: `None` itself, `Optional[T]`, a union with a `None` arm, or an
/// `Annotated[T, ...]` whose `T` does. Read off the spelling, as the crate reads every annotation.
pub(crate) fn allows_none(annotation: &str) -> bool {
    let annotation = annotation.trim();
    if matches!(annotation, "None" | "NoneType" | "type(None)") {
        return true;
    }
    let arms = super::parse::split_top_level(annotation, '|');
    if arms.len() > 1 {
        return arms.iter().any(|arm| allows_none(arm));
    }
    match split_generic(annotation) {
        Some((name, _)) if head_of(name) == "Optional" => true,
        Some((name, inside)) if head_of(name) == "Union" => {
            super::parse::split_top_level(inside, ',')
                .iter()
                .any(|arm| allows_none(arm))
        }
        Some((name, inside)) if head_of(name) == "Annotated" => {
            super::parse::split_top_level(inside, ',')
                .first()
                .is_some_and(|first| allows_none(first))
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Measured against dspy's own `parse_value`, which takes the text `42` back as the string
    /// `"42"` for the first group and repairs it to the number `42` for the second. `Union[str,
    /// int, NoneType]` against `Union[str, int]` is the pair that shows both halves of the
    /// condition are read: dropping either one flips a case.
    #[test]
    fn a_union_naming_both_none_and_str_takes_the_text_as_itself() {
        let cases = [
            ("UnionType[str, NoneType]", true),
            ("Union[str, NoneType]", true),
            ("Optional[str]", true),
            ("typing.Optional[str]", true),
            ("Union[str, int, NoneType]", true),
            ("str | None", true),
            ("UnionType[int, NoneType]", false),
            ("int | None", false),
            ("UnionType[list[str], NoneType]", false),
            ("Union[dict[str, int], NoneType]", false),
            ("Union[str, int]", false),
            ("str", false),
            // `Annotated` is not a union head to `get_origin`, so upstream repairs the text and
            // then fails to validate it. The shortcut is not taken around a union it wraps.
            ("Annotated[UnionType[str, NoneType], x]", false),
        ];
        for (annotation, expected) in cases {
            assert_eq!(union_takes_text(annotation), expected, "{annotation}");
        }
    }

    /// The PEP 604 spelling reaches the crate as `UnionType[...]`, which is the name of the
    /// `types.UnionType` object `A | B` builds. Measured against `annotation_allows_none`.
    #[test]
    fn a_printed_union_type_admits_none_the_way_its_pipe_spelling_does() {
        let cases = [
            ("UnionType[str, NoneType]", true),
            ("UnionType[list[str], NoneType]", true),
            ("UnionType[dict[str, int], NoneType]", true),
            ("Annotated[UnionType[str, NoneType], x]", true),
            ("Union[str, int, NoneType]", true),
            ("UnionType[str, int]", false),
            ("Union[str, int]", false),
        ];
        for (annotation, expected) in cases {
            assert_eq!(allows_none(annotation), expected, "{annotation}");
        }
    }

    /// Every shape upstream was measured printing, spelled as it prints it.
    #[test]
    fn each_annotation_is_the_schema_dspy_prints() {
        let cases = [
            ("list[str]", r#"{"type":"array","items":{"type":"string"}}"#),
            (
                "list[int]",
                r#"{"type":"array","items":{"type":"integer"}}"#,
            ),
            (
                "dict[str, int]",
                r#"{"type":"object","additionalProperties":{"type":"integer"}}"#,
            ),
            (
                "dict[str, Any]",
                r#"{"type":"object","additionalProperties":true}"#,
            ),
            // Named without parameters, which upstream resolves to the bare builtin.
            ("dict", r#"{"type":"object","additionalProperties":true}"#),
            ("list", r#"{"type":"array","items":{}}"#),
            ("set", r#"{"type":"array","items":{},"uniqueItems":true}"#),
            ("Any", "{}"),
            ("bytes", r#"{"type":"string","format":"binary"}"#),
            (
                "list[list[str]]",
                r#"{"type":"array","items":{"type":"array","items":{"type":"string"}}}"#,
            ),
            (
                "set[str]",
                r#"{"type":"array","items":{"type":"string"},"uniqueItems":true}"#,
            ),
            (
                "tuple[str, int]",
                r#"{"type":"array","maxItems":2,"minItems":2,"prefixItems":[{"type":"string"},{"type":"integer"}]}"#,
            ),
            (
                "Optional[str]",
                r#"{"anyOf":[{"type":"string"},{"type":"null"}]}"#,
            ),
            (
                "str | None",
                r#"{"anyOf":[{"type":"string"},{"type":"null"}]}"#,
            ),
            // `str(typing.Optional[int])`, which is how a reflected annotation spells itself, and
            // upstream evaluates it against `typing` like the bare name.
            (
                "typing.Optional[int]",
                r#"{"anyOf":[{"type":"integer"},{"type":"null"}]}"#,
            ),
            (
                "typing.List[typing.Dict[str, int]]",
                r#"{"type":"array","items":{"type":"object","additionalProperties":{"type":"integer"}}}"#,
            ),
        ];
        for (annotation, expected) in cases {
            let schema = schema_for(annotation).unwrap_or_else(|| panic!("{annotation}"));
            // Compared as text: under `preserve_order` two objects differing only in key order
            // are equal, and the key order is what the prompt carries.
            assert_eq!(
                serde_json::to_string(&schema).expect("serializes"),
                expected,
                "for {annotation}"
            );
        }
    }

    /// A name the string form cannot resolve carries no schema rather than a wrong one.
    #[test]
    fn an_unknown_name_has_no_schema() {
        assert_eq!(schema_for("MyModel"), None);
        assert_eq!(schema_for("list[MyModel]"), None);
    }

    /// dspy's own types are named through the module, and the bare name is not a name at all.
    #[test]
    fn a_custom_type_is_reached_through_dspy() {
        assert_eq!(custom_type("dspy.History"), Some("History"));
        assert_eq!(custom_type("dspy.Image"), Some("Image"));
        assert_eq!(
            custom_type("History"),
            None,
            "the bare name is a ValueError upstream"
        );
        assert_eq!(custom_type("dspy.Nonesuch"), None);
    }

    /// Measured on dspy 3.3.1 over each annotation the string form can spell.
    #[test]
    fn the_spelling_decides_nesting_where_it_can() {
        for nested in [
            "list[str]",
            "dict[str, int]",
            "dict[str, Any]",
            "Optional[list[str]]",
            "list[str] | None",
            "Union[list[str], NoneType]",
            "List[str]",
            "Dict[str, int]",
            "list[list[int]]",
            "list[MyModel]",
        ] {
            assert_eq!(nested_xml_shape(nested), Some(true), "{nested}");
        }
        for text in [
            "str",
            "int",
            "float",
            "bool",
            "list",
            "dict",
            "Any",
            "tuple[int, int]",
            "set[str]",
            "Literal['a', 'b']",
            "dspy.Image",
            "Image",
            "Code",
            "ToolCalls",
            "list[dspy.Image]",
            "list[Image]",
            "int | str",
            "list[int] | dict[str, int]",
            "Optional[int | str]",
        ] {
            assert_eq!(nested_xml_shape(text), Some(false), "{text}");
        }
        assert_eq!(
            nested_xml_shape("MyModel"),
            None,
            "a bare class is the schema's to decide"
        );
        assert_eq!(nested_xml_shape("Optional[MyModel]"), None);
    }
}
