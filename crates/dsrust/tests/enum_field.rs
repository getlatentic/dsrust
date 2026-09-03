//! A field whose type is an enumeration renders as dspy renders one.
//!
//! dspy prints an `Enum` field's own type name and asks for **one of its members' values**; it
//! prints a structure's JSON schema instead. `#[derive(Signature)]` reads a Rust type and cannot
//! tell the two apart — both are a path — so it asked the type's schema, which can.
//!
//! Before it did, a Rust enum reached the prompt as a schema. That is a different prompt than
//! dspy's for the same program, found by porting DSPy's email-extraction tutorial and diffing the
//! two renderings: `class EmailType(str, Enum)` is entirely ordinary DSPy, and the whole system
//! message matched except this one note.

use dsrust::Signature;
use dsrust::adapter::{Adapter, ChatAdapter, Input};
use dsrust::signature::SignatureSpec;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
enum Urgency {
    Low,
    Medium,
    High,
}

/// A structure, to hold the other half of the distinction: dspy renders this as a schema.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
struct Entity {
    value: String,
}

#[derive(Signature)]
/// Classify it.
struct Classify {
    #[input]
    text: String,
    /// How urgent
    #[output]
    urgency: Urgency,
    /// What was found
    #[output]
    entity: Entity,
}

fn system_message() -> String {
    ChatAdapter::default()
        .format(
            &Classify::signature(),
            &[],
            &[Input::new("text", json!("x"))],
        )
        .expect("renders")
        .first()
        .and_then(|message| message.text())
        .unwrap_or_default()
}

/// The note dspy writes for an enum, character for character.
#[test]
fn an_enum_field_names_its_members() {
    let rendered = system_message();
    assert!(
        rendered.contains("must be one of: low; medium; high"),
        "expected dspy's enum note, got:\n{rendered}"
    );
    // And *not* the schema note, which is what it rendered before.
    assert!(
        !rendered.contains(r#"{"type": "string", "enum""#),
        "an enum still rendered as a schema:\n{rendered}"
    );
}

/// The type's own name is the annotation, as dspy prints it — not `str`, and not a schema.
#[test]
fn an_enum_field_is_annotated_with_its_type() {
    assert!(
        system_message().contains("`urgency` (Urgency):"),
        "{}",
        system_message()
    );
}

/// A structure is still a schema. The two renderings are different upstream, so making an enum
/// match must not make a model match it too.
#[test]
fn a_structure_still_renders_as_a_schema() {
    let rendered = system_message();
    assert!(
        rendered.contains("must adhere to the JSON schema"),
        "{rendered}"
    );
    assert!(rendered.contains("`entity` (Entity):"), "{rendered}");
}

/// The members come from the type, so a variant renamed by serde travels under the name the model
/// is actually asked for.
#[test]
fn the_members_are_the_serialized_names() {
    assert_eq!(
        dsrust::signature::declared_members::<Urgency>(),
        Some(vec![
            dsrust::signature::LiteralValue::Str("low".to_owned()),
            dsrust::signature::LiteralValue::Str("medium".to_owned()),
            dsrust::signature::LiteralValue::Str("high".to_owned()),
        ])
    );
    // A structure names no members, which is what keeps it on the schema path.
    assert_eq!(dsrust::signature::declared_members::<Entity>(), None);
}
