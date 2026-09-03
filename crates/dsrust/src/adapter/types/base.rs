//! dspy `adapters/types/base_type.py`: the `Type` a signature field can carry.
//!
//! A custom type states how it *renders* — a list of content blocks for a multimodal value, a bare
//! string for a text-like one — and how it *reads back* from a reply. dspy keeps a live `Type`
//! object and calls its methods; here a field carries a [`serde_json::Value`], so the same reach is
//! a trait a type implements plus its own serde: [`serialized`] is dspy's `serialize_model`, and
//! the value it produces is what `split_custom_types` reads back into a
//! multimodal message.

use serde_json::Value;

use crate::signature::TypeDescription;

/// The two shapes dspy's `Type.format` returns: content blocks that reach a multimodal message, or
/// a bare string that renders as text.
#[derive(Debug, Clone, PartialEq)]
pub enum Formatted {
    Blocks(Vec<Value>),
    Text(String),
}

/// The sentinels dspy wraps a block-list in, so a custom type survives being rendered into a single
/// string field and is split back out. They are reserved — a value carrying one is split at it.
pub(crate) const CUSTOM_TYPE_START: &str = "<<CUSTOM-TYPE-START-IDENTIFIER>>";
pub(crate) const CUSTOM_TYPE_END: &str = "<<CUSTOM-TYPE-END-IDENTIFIER>>";

/// dspy's `Type`: a value a signature field can carry beyond the scalars and plain structures,
/// stating how it renders and how it reads back. The built-ins — [`Image`](super::Image),
/// [`Audio`](super::Audio), [`File`](super::File), [`Code`](super::Code) — implement it, and a
/// caller can too, to put their own type in a field.
pub trait Type {
    /// dspy `Type.format`: the content this value contributes. `Blocks` reach a message as its own
    /// content parts (an image, an audio clip); `Text` renders inline like any other field value.
    fn format(&self) -> Formatted;

    /// dspy `Type.description`: the prose the type states about itself on its field's line, the same
    /// for every field of it. Most types say nothing.
    fn description() -> Option<TypeDescription>
    where
        Self: Sized,
    {
        None
    }

    /// The schema this type prints in an output field's note, when it prints one.
    ///
    /// pydantic renders a custom type's model *and its class docstring*, so upstream's schema for
    /// one is not a shape a Rust struct produces — the crate records each verbatim. Most types
    /// print none: `Code` states its contract in prose instead, and an input never carries a note
    /// at all.
    ///
    /// On the trait rather than beside each type because `#[derive(Signature)]` has to ask it of a
    /// field's declared type without knowing which type that is.
    fn output_schema() -> Option<Value>
    where
        Self: Sized,
    {
        None
    }

    /// dspy `Type.parse_lm_response`: read this type out of a reply that carried it on its own
    /// channel rather than as a rendered field — the way reasoning content comes back beside the
    /// answer. A type that only ever renders returns `None`.
    fn parse_lm_response(_response: &Value) -> Option<Self>
    where
        Self: Sized,
    {
        None
    }

    /// dspy `Type.is_streamable`.
    fn is_streamable() -> bool
    where
        Self: Sized,
    {
        false
    }
}

/// dspy `Type.serialize_model`: a value as the string its field carries.
///
/// A block-list is written as JSON between the sentinels, so it survives a render that turns every
/// field into one string and is split back into content parts afterward; a bare string is carried
/// as itself, which is how a text-like type (dspy's `Code`) bypasses the sentinels. This is what a
/// caller stores in an [`Example`](crate::example::Example) field for the value to render.
pub fn serialized<T: Type + ?Sized>(value: &T) -> String {
    match value.format() {
        Formatted::Blocks(blocks) => {
            let json = serde_json::to_string(&blocks).unwrap_or_else(|_| "[]".to_owned());
            format!("{CUSTOM_TYPE_START}{json}{CUSTOM_TYPE_END}")
        }
        Formatted::Text(text) => text,
    }
}

/// The [`Value`] a caller puts in a field for `value` to render — [`serialized`] as a JSON string.
///
/// ```
/// use dsrust::adapter::types::base::to_field_value;
/// use dsrust::Image;
///
/// // A custom type reaches a field as its *serialized* form — a JSON string, not a nested object.
/// // dspy does the same, and an adapter reads the string back out when it renders.
/// let image = Image::new("https://example.invalid/a.png").expect("a url");
/// assert!(to_field_value(&image).is_string());
/// ```
pub fn to_field_value<T: Type + ?Sized>(value: &T) -> Value {
    Value::String(serialized(value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct Marker;
    impl Type for Marker {
        fn format(&self) -> Formatted {
            Formatted::Blocks(vec![
                json!({ "type": "image_url", "image_url": { "url": "u" } }),
            ])
        }
    }

    struct Bare;
    impl Type for Bare {
        fn format(&self) -> Formatted {
            Formatted::Text("just text".to_owned())
        }
    }

    /// A block-list is wrapped in the sentinels, so the render's string round trip can split it
    /// back out; the JSON between them is exactly the blocks `format` returned.
    #[test]
    fn a_block_list_is_wrapped_in_the_sentinels() {
        assert_eq!(
            serialized(&Marker),
            format!(
                "{CUSTOM_TYPE_START}{}{CUSTOM_TYPE_END}",
                r#"[{"type":"image_url","image_url":{"url":"u"}}]"#
            )
        );
    }

    /// A bare string carries as itself — no sentinels, the way a text-like type renders inline.
    #[test]
    fn a_bare_string_is_carried_as_itself() {
        assert_eq!(serialized(&Bare), "just text");
        assert_eq!(to_field_value(&Bare), json!("just text"));
    }

    /// The defaults match dspy's base class: no description, nothing parsed off a reply, not
    /// streamable.
    #[test]
    fn the_defaults_are_upstreams() {
        assert_eq!(Marker::description(), None);
        assert!(Marker::parse_lm_response(&json!({})).is_none());
        assert!(!Marker::is_streamable());
    }
}
