//! `LMMessage` and `LMToolSpec`.

use serde_json::Value;

use super::part::{LmPart, Metadata};

/// One role's turn as a list of parts.
///
/// Serializes as itself; *deserializes* from either that or the shape a provider writes — see
/// [`OpenAiShaped`](super::openai_shape::OpenAiShaped), which is upstream's `normalize_parts`.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "super::openai_shape::OpenAiShaped")]
pub struct LmMessage {
    pub role: String,
    pub parts: Vec<LmPart>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Metadata::is_empty")]
    pub metadata: Metadata,
}

impl LmMessage {
    pub fn new(role: impl Into<String>, parts: Vec<LmPart>) -> Self {
        Self {
            role: role.into(),
            parts,
            name: None,
            metadata: Metadata::new(),
        }
    }

    pub fn user(parts: Vec<LmPart>) -> Self {
        Self::new("user", parts)
    }

    pub fn assistant(parts: Vec<LmPart>) -> Self {
        Self::new("assistant", parts)
    }

    pub fn system(parts: Vec<LmPart>) -> Self {
        Self::new("system", parts)
    }

    /// Every text part run together, and `None` when the message holds no prose at all —
    /// upstream returns `None` rather than `""` so a message of pure images is distinguishable
    /// from one whose text is empty.
    pub fn text(&self) -> Option<String> {
        let texts: Vec<&str> = self.parts.iter().filter_map(LmPart::as_text).collect();
        match texts.is_empty() {
            true => None,
            false => Some(texts.concat()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LmToolSpec {
    #[serde(default = "function_type")]
    pub r#type: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: Metadata,
    #[serde(default, skip_serializing_if = "Metadata::is_empty")]
    pub metadata: Metadata,
    #[serde(default, skip_serializing_if = "Metadata::is_empty")]
    pub provider_data: Metadata,
}

impl LmToolSpec {
    pub fn new(name: impl Into<String>, parameters: Metadata) -> Self {
        Self {
            r#type: function_type(),
            name: name.into(),
            description: None,
            parameters,
            metadata: Metadata::new(),
            provider_data: Metadata::new(),
        }
    }

    pub fn described(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// The shape a provider's `tools` array takes.
    pub fn to_openai(&self) -> Value {
        serde_json::json!({
            "type": self.r#type,
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        })
    }
}

fn function_type() -> String {
    "function".to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_messages_text_is_every_text_part_run_together() {
        let message = LmMessage::user(vec![
            LmPart::text("before "),
            LmPart::image_url("https://example.com/a.jpg"),
            LmPart::text("after"),
        ]);
        assert_eq!(message.text(), Some("before after".to_owned()));
    }

    /// A message of pure images has no text, which upstream reports as absent rather than as an
    /// empty string.
    #[test]
    fn a_message_with_no_prose_has_no_text_rather_than_an_empty_one() {
        let message = LmMessage::user(vec![LmPart::image_url("https://example.com/a.jpg")]);
        assert_eq!(message.text(), None);
    }

    #[test]
    fn a_message_forbids_what_it_does_not_declare() {
        assert!(
            serde_json::from_value::<LmMessage>(json!({
                "role": "user",
                "parts": [],
                "content": "the old spelling",
            }))
            .is_err()
        );
    }

    #[test]
    fn a_tool_spec_is_a_function_and_renders_as_one() {
        let mut parameters = Metadata::new();
        parameters.insert("type".to_owned(), json!("object"));
        let spec = LmToolSpec::new("search", parameters).described("look things up");

        assert_eq!(spec.r#type, "function");
        assert_eq!(
            spec.to_openai(),
            json!({
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "look things up",
                    "parameters": { "type": "object" },
                },
            })
        );
    }
}
