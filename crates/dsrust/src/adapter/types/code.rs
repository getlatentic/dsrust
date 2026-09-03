//! dspy `adapters/types/code.py`: the `Code` type.

use serde::de::{self, Deserializer};
use serde::{Deserialize, Serialize, Serializer};

use crate::signature::TypeDescription;

use super::base::{Formatted, Type, serialized};

/// dspy's `Code`: source in a string, carried on the `code` field.
///
/// It is text-like — [`format`](Type::format) yields the code itself, and its serialized form
/// bypasses the custom-type sentinels the way upstream's `serialize_model` override does — so a
/// `Code` value reaches a prompt as plain text. On the way back, code the model wrapped in a
/// markdown fence is unwrapped, matching dspy's `_filter_code`.
///
/// `language` names the programming language for the field's description. dspy spells it with the
/// `Code["python"]` subscript, which builds a distinct class per language; here it is a field,
/// defaulting to `python`. Because [`description`](Type::description) is a per-type function with no
/// instance to read, it states the default language; a field in another language sets its own
/// description through the [`Type`] seam.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Code {
    pub code: String,
    pub language: String,
}

/// dspy's `Code.language` class default.
const DEFAULT_LANGUAGE: &str = "python";

impl Code {
    /// Code in the default language, `python`.
    pub fn new(code: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            language: DEFAULT_LANGUAGE.to_owned(),
        }
    }

    /// Code in a named language — dspy's `Code["java"]`.
    pub fn in_language(code: impl Into<String>, language: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            language: language.into(),
        }
    }

    /// dspy `Code.description` for a given language: prose stating the `code` field and, for an
    /// output, the markdown fence the model should answer in.
    pub fn description_for(language: &str) -> TypeDescription {
        TypeDescription {
            name: format!("Code_{language}"),
            text: format!(
                "Code represented in a string, specified in the `code` field. If this is an output \
                 field, the code field should follow the markdown code block format, e.g. \
                 \n```{}\n{{code}}\n```\nProgramming language: {language}",
                language.to_lowercase()
            ),
            // dspy sets `replaces_schema` for `Code` alone: its prose already spells out the block
            // it expects, so the field states its contract once instead of twice.
            replaces_schema: true,
        }
    }
}

impl Type for Code {
    /// dspy `Code.format`: the code itself, so it renders as text rather than a content block.
    fn format(&self) -> Formatted {
        Formatted::Text(self.code.clone())
    }

    /// The bare `dspy.Code`, named `Code`; a subscripted `Code["x"]` is `Code_x`.
    fn description() -> Option<TypeDescription> {
        Some(TypeDescription {
            name: "Code".to_owned(),
            ..Self::description_for(DEFAULT_LANGUAGE)
        })
    }
}

impl Serialize for Code {
    /// dspy `Code.serialize_model`: the code string, without the custom-type sentinels a block-list
    /// carries.
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&serialized(self))
    }
}

impl<'de> Deserialize<'de> for Code {
    /// dspy `Code.validate_input`: a bare string is the code (with a markdown fence stripped), and a
    /// mapping must carry a string `code`. The language is not on the wire — it belongs to the
    /// field's type — so a read-back value keeps the default.
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        match serde_json::Value::deserialize(deserializer)? {
            serde_json::Value::String(text) => Ok(Self::new(filter_code(&text))),
            serde_json::Value::Object(mut map) => match map.remove("code") {
                Some(serde_json::Value::String(code)) => Ok(Self::new(filter_code(&code))),
                Some(other) => Err(de::Error::custom(format!(
                    "`code` field must be a string, but received type: {}",
                    crate::python::type_of(&other)
                ))),
                None => Err(de::Error::custom(
                    "`code` field is required for `dspy.Code`",
                )),
            },
            other => Err(de::Error::custom(format!(
                "Received invalid value for `dspy.Code`: {}",
                crate::python::text(&other)
            ))),
        }
    }
}

impl schemars::JsonSchema for Code {
    /// A string on the wire — the code — so an output field carries a string's schema, which its
    /// own description then replaces.
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "Code".into()
    }

    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        String::json_schema(generator)
    }
}

/// dspy `_filter_code`: the code out of a markdown block, its language identifier and fences
/// stripped, or the string as given when it carries no fence.
///
/// Upstream's two regexes: ```` ```lang\n(code)``` ```` — a language line, then the code to the
/// closing fence — and, failing that, ```` ```(code)``` ```` for a fence with no language line.
fn filter_code(code: &str) -> String {
    let Some(open) = code.find("```") else {
        return code.to_owned();
    };
    let after_open = &code[open + 3..];
    // With a language line: skip to its newline, then take up to the closing fence.
    if let Some(newline) = after_open.find('\n') {
        let body = &after_open[newline + 1..];
        if let Some(close) = body.find("```") {
            return body[..close].trim().to_owned();
        }
    }
    // No language line: take between the fences.
    if let Some(close) = after_open.find("```") {
        return after_open[..close].trim().to_owned();
    }
    code.to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// `format` yields the code, and the serialized value is that code with no sentinels.
    #[test]
    fn it_renders_as_the_code_itself() {
        let code = Code::new("print('hi')");
        assert_eq!(code.format(), Formatted::Text("print('hi')".to_owned()));
        assert_eq!(
            serde_json::to_value(&code).expect("serializes"),
            json!("print('hi')")
        );
    }

    /// dspy `_filter_code`: a fenced block reads back as the code inside it, language line dropped.
    #[test]
    fn a_markdown_fence_is_stripped_on_read_back() {
        let fenced: Code =
            serde_json::from_value(json!("```python\nprint('hi')\n```")).expect("parses");
        assert_eq!(fenced.code, "print('hi')");

        let bare: Code = serde_json::from_value(json!("```x = 1```")).expect("parses");
        assert_eq!(bare.code, "x = 1");

        let plain: Code = serde_json::from_value(json!("x = 1")).expect("parses");
        assert_eq!(plain.code, "x = 1");
    }

    /// A mapping must carry a string `code`; anything else is refused, as upstream's validator does.
    #[test]
    fn it_reads_a_code_mapping_and_refuses_the_rest() {
        let from_map: Code = serde_json::from_value(json!({ "code": "x = 1" })).expect("parses");
        assert_eq!(from_map.code, "x = 1");
        assert!(serde_json::from_value::<Code>(json!({ "other": 1 })).is_err());
        assert!(serde_json::from_value::<Code>(json!({ "code": 3 })).is_err());
        assert!(serde_json::from_value::<Code>(json!(3)).is_err());
    }

    /// dspy sets `replaces_schema` for `Code`, and the prose names the language.
    #[test]
    fn its_description_replaces_the_schema_and_names_the_language() {
        let description = Code::description().expect("a description");
        assert_eq!(description.name, "Code");
        assert!(description.replaces_schema);
        assert!(description.text.contains("Programming language: python"));
        assert!(
            Code::description_for("Java")
                .text
                .contains("Programming language: Java")
        );
        assert!(Code::description_for("Java").text.contains("```java\n"));
    }
}
