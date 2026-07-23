//! dspy `adapters/types/reasoning.py`: the `Reasoning` type.

use std::fmt;
use std::ops::Deref;

use serde::de::{self, Deserializer};
use serde::{Deserialize, Serialize, Serializer};

/// dspy's `Reasoning`: the model's reasoning, carried as a type rather than a bare string so a
/// program reads the same whether the reasoning came from a reasoning model's own channel or from
/// a field the prompt asked for.
///
/// It is deliberately **str-like** — upstream forwards every string method to the content, and its
/// adapters treat it as a string: `get_annotation_name` prints `str`, so the field carries no
/// schema note, and the value formats as the content itself. [`Deref`] to `str` is the same
/// promise here, so `reasoning.len()`, `reasoning.trim()` and the rest work directly.
///
/// The one place it is *not* a string is the output-requirement hint, which dspy decides by asking
/// whether the annotation is `str` — it is not — so the field still reads
/// `(must be formatted as a valid Python str)`. [`FieldKind::Reasoning`](crate::signature::FieldKind::Reasoning)
/// is what carries that distinction.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Reasoning {
    pub content: String,
}

impl Reasoning {
    pub fn new(content: impl Into<String>) -> Self {
        Self { content: content.into() }
    }

    /// dspy `Reasoning.format`: the content itself, which is what a prompt renders for this field.
    pub fn format(&self) -> &str {
        &self.content
    }
}

impl fmt::Display for Reasoning {
    /// dspy's `__str__`: the content, unquoted.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.content)
    }
}

impl Deref for Reasoning {
    type Target = str;

    /// dspy forwards unknown attributes to the content string; the same reach, statically.
    fn deref(&self) -> &str {
        &self.content
    }
}

impl AsRef<str> for Reasoning {
    fn as_ref(&self) -> &str {
        &self.content
    }
}

impl From<&str> for Reasoning {
    fn from(content: &str) -> Self {
        Self::new(content)
    }
}

impl From<String> for Reasoning {
    fn from(content: String) -> Self {
        Self { content }
    }
}

impl From<Reasoning> for String {
    fn from(reasoning: Reasoning) -> Self {
        reasoning.content
    }
}

impl PartialEq<str> for Reasoning {
    /// dspy's `__eq__` compares equal to a plain string with the same content.
    fn eq(&self, other: &str) -> bool {
        self.content == other
    }
}

impl PartialEq<&str> for Reasoning {
    fn eq(&self, other: &&str) -> bool {
        self.content == *other
    }
}

impl PartialEq<String> for Reasoning {
    fn eq(&self, other: &String) -> bool {
        &self.content == other
    }
}

impl PartialEq<Reasoning> for str {
    fn eq(&self, other: &Reasoning) -> bool {
        self == other.content
    }
}

impl PartialEq<Reasoning> for String {
    fn eq(&self, other: &Reasoning) -> bool {
        self == &other.content
    }
}

impl Serialize for Reasoning {
    /// The content alone. dspy's adapters format this type through `format`, which yields the
    /// string, so that is what reaches a prompt and what a demo's value carries.
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.content)
    }
}

impl<'de> Deserialize<'de> for Reasoning {
    /// dspy's `validate_input`: a bare string becomes the content, and a mapping must carry a
    /// string `content`. Anything else is refused rather than guessed at.
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        match serde_json::Value::deserialize(deserializer)? {
            serde_json::Value::String(content) => Ok(Self { content }),
            serde_json::Value::Object(mut map) => match map.remove("content") {
                Some(serde_json::Value::String(content)) => Ok(Self { content }),
                Some(other) => Err(de::Error::custom(format!(
                    "`content` field must be a string, but received type: {other}"
                ))),
                None => Err(de::Error::custom("`content` field is required for `Reasoning`")),
            },
            other => Err(de::Error::custom(format!(
                "Received invalid value for `Reasoning`: {other}"
            ))),
        }
    }
}

impl schemars::JsonSchema for Reasoning {
    /// A string, matching how the field schemas and how the value crosses.
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "Reasoning".into()
    }

    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        String::json_schema(generator)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_reads_as_the_string_it_carries() {
        let reasoning = Reasoning::new("  I thought about it.  ");
        // dspy forwards string methods to the content; Deref is the same reach.
        assert_eq!(reasoning.trim(), "I thought about it.");
        assert_eq!(reasoning.len(), "  I thought about it.  ".len());
        assert_eq!(reasoning.format(), "  I thought about it.  ");
        assert_eq!(reasoning.to_string(), "  I thought about it.  ");
    }

    #[test]
    fn it_compares_equal_to_a_plain_string() {
        // dspy's `__eq__` answers True against a str with the same content.
        let reasoning = Reasoning::new("because");
        assert_eq!(reasoning, "because");
        assert_eq!(reasoning, "because".to_owned());
        assert!(reasoning != "other");
    }

    /// dspy's validator takes a bare string or a mapping carrying `content`, and refuses the rest.
    #[test]
    fn it_reads_back_from_a_string_or_a_content_mapping() {
        let from_text: Reasoning = serde_json::from_str(r#""thought""#).expect("a string parses");
        assert_eq!(from_text, "thought");
        let from_map: Reasoning =
            serde_json::from_str(r#"{"content": "thought"}"#).expect("a mapping parses");
        assert_eq!(from_map, "thought");

        let missing = serde_json::from_str::<Reasoning>(r#"{"other": 1}"#).unwrap_err();
        assert!(missing.to_string().contains("`content` field is required"));
        let wrong_type = serde_json::from_str::<Reasoning>(r#"{"content": 3}"#).unwrap_err();
        assert!(wrong_type.to_string().contains("must be a string"));
        assert!(serde_json::from_str::<Reasoning>("3").is_err());
    }

    /// It reaches a prompt as its content, which is what `format` yields upstream.
    #[test]
    fn it_writes_out_as_its_content() {
        let json = serde_json::to_string(&Reasoning::new("short")).expect("serializes");
        assert_eq!(json, r#""short""#);
    }
}
