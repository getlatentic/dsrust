//! What a declared Rust type says about itself, for the prompt to state.
//!
//! Two questions, one source. A signature field's type has to reach the model somehow, and dspy
//! reaches for the pydantic annotation at run time — a thing Rust does not have. What Rust does
//! have is the `schemars` schema every non-scalar field already carries, and both answers are read
//! off it:
//!
//! * **the schema itself**, which a structured field's prompt note prints; and
//! * **whether the type is an enumeration**, which decides *which* note it gets at all — dspy
//!   prints an enum's own name and asks for one of its members, and a structure's schema.
//!
//! The second is here rather than in the derive because a macro reads a Rust type as a path and
//! cannot tell an enum from a struct. The schema can.

use serde_json::{Map, Value};

use super::LiteralValue;

/// The schema a structured field's prompt note prints — as dspy prints it.
///
/// dspy builds this from `pydantic.TypeAdapter(t).json_schema()` and puts the result straight into
/// the prompt, so the schema is prompt text rather than an implementation detail. schemars writes
/// valid JSON Schema in a different dialect, which is a different prompt for the same program:
/// four ways they differ, and `signature/pydantic.rs` is the translation between them.
///
/// Subschemas are *not* inlined, because pydantic does not inline them — a named type is hoisted
/// into `$defs` with a `$ref` pointing at it. An earlier version inlined them "so it stays a
/// one-liner in prompts", which read better and matched nothing.
pub fn json_field_schema<T: schemars::JsonSchema>() -> Value {
    let generator = schemars::generate::SchemaSettings::default()
        .with(|settings| {
            settings.meta_schema = None;
        })
        .into_generator();
    super::pydantic::as_dspy_prints_it(generator.into_root_schema_for::<T>().to_value())
}

/// The same type's schema with its structure spelled out in place, and nothing said about how a
/// prompt should print it.
///
/// Two consumers need this rather than [`json_field_schema`], and for the same reason: they *walk*
/// the schema instead of printing it. [`json_field_reflection`](super::json_field_reflection)
/// builds BAML's type tree from it, and a tool's argument map is one entry per argument — a `$ref`
/// into a `$defs` block is a hop neither can follow.
///
/// So the split is not duplication. One answers "what does dspy print", the other "what shape is
/// this type", and they were one function only while those two happened to coincide.
pub(crate) fn inlined_schema<T: schemars::JsonSchema>() -> Value {
    let mut root = inlined_with_title::<T>();
    if let Some(object) = root.as_object_mut() {
        object.remove("title");
    }
    root
}

/// The same, keeping the name schemars gave the type.
///
/// Whether a root title belongs is pydantic's rule, not this function's: a model keeps its name and
/// a container never had one, which [`as_pydantic_prints_it`](super::pydantic::as_pydantic_prints_it)
/// decides. Stripping it here would take the model's name too.
fn inlined_with_title<T: schemars::JsonSchema>() -> Value {
    super::inline::resolved(json_field_schema_raw::<T>())
}

/// The schema schemars writes, references intact.
fn json_field_schema_raw<T: schemars::JsonSchema>() -> Value {
    let generator = schemars::generate::SchemaSettings::default()
        .with(|settings| settings.meta_schema = None)
        .into_generator();
    generator.into_root_schema_for::<T>().to_value()
}

/// One tool argument's schema, as dspy's `Tool.args` carries it.
///
/// Upstream builds this map with `_resolve_json_schema_reference(TypeAdapter(t).json_schema())` and
/// applies none of the ordering it gives a signature's field note — so a `$ref` is expanded where a
/// field note would have kept `$defs`, and the keys stay pydantic's rather than type-first. The map
/// is rendered into the roster as "It takes arguments {args}", which makes both differences prompt
/// text.
pub fn json_argument_schema<T: schemars::JsonSchema>() -> Value {
    super::pydantic::as_pydantic_prints_it(inlined_with_title::<T>())
}

/// A whole argument list read off one Rust type, each field rendered as the parameter it stands
/// for.
///
/// The per-field distinction matters: dspy builds `Tool.args` one parameter at a time, with
/// `TypeAdapter(annotation).json_schema()`, so each entry is the *root* of its own schema. A `str`
/// parameter is `{"type": "string"}` and carries no title, where the same type as a *property* of
/// a model would be titled. Translating the struct in one pass titles every parameter and writes a
/// roster dspy would not.
pub fn arguments_schema<T: schemars::JsonSchema>() -> Option<Map<String, Value>> {
    let schema = inlined_with_title::<T>();
    let properties = schema.get("properties")?.as_object()?;
    Some(
        properties
            .iter()
            .map(|(name, schema)| {
                let rendered = super::pydantic::as_pydantic_prints_it(schema.clone());
                (name.clone(), rendered)
            })
            .collect(),
    )
}

/// The members of a declared type, when that type is an enumeration of strings.
///
/// `#[derive(Signature)]` reads a Rust type and cannot tell an enum from a struct — both are a
/// path. The *schema* can, and dspy renders the two differently: an enum prints its own name as the
/// annotation and asks for one of its members' values, where a structure prints a JSON schema. A
/// Rust enum reaching a prompt as a schema is therefore a different prompt than dspy's for the same
/// program, which is what this exists to prevent.
///
/// `None` for anything that is not a plain string enumeration — a struct, a list, an enum carrying
/// data — all of which dspy does render as a schema.
///
/// ```
/// # use dsrust::signature::{declared_members, LiteralValue};
/// # use schemars::JsonSchema;
/// #[derive(JsonSchema)]
/// #[serde(rename_all = "snake_case")]
/// enum Urgency {
///     Low,
///     High,
/// }
///
/// assert_eq!(
///     declared_members::<Urgency>(),
///     Some(vec![
///         LiteralValue::Str("low".to_owned()),
///         LiteralValue::Str("high".to_owned()),
///     ])
/// );
/// #[derive(JsonSchema)]
/// struct Entity { value: String }
/// assert_eq!(declared_members::<Entity>(), None);
/// ```
pub fn declared_members<T: schemars::JsonSchema>() -> Option<Vec<LiteralValue>> {
    let schema = json_field_schema::<T>();
    // A string enumeration and nothing else: `{"type": "string", "enum": [...]}`. A variant
    // carrying data schematises as `oneOf`, and dspy renders that as a schema too.
    if schema.get("type").and_then(Value::as_str) != Some("string") {
        return None;
    }
    let members: Vec<LiteralValue> = schema
        .get("enum")?
        .as_array()?
        .iter()
        .filter_map(|member| Some(LiteralValue::Str(member.as_str()?.to_owned())))
        .collect();
    match members.is_empty() {
        true => None,
        false => Some(members),
    }
}
