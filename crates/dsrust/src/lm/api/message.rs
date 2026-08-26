//! `LMMessage` and `LMToolSpec`.

use serde_json::Value;

use super::part::{LmPart, Metadata};

/// One role's turn as a list of parts.
///
/// Serializes as itself; *deserializes* from either that or the shape a provider writes — see
/// `OpenAiShaped`, which is upstream's `normalize_parts`.
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
    /// Who is speaking, where the role alone does not say — dspy's `name=` keyword on every role
    /// constructor.
    ///
    /// It reaches the wire: OpenAI takes `messages[].name` right after the role, which is how a
    /// multi-agent transcript keeps two `user` turns apart. A builder rather than an argument
    /// because a Rust function has no keyword arguments, and the name is dspy's own.
    ///
    /// ```
    /// # use dsrust::{LmMessage, User};
    /// let turn = User!["hello"].name("alice");
    /// assert_eq!(turn.name.as_deref(), Some("alice"));
    /// ```
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// dspy's `metadata=` keyword: anything a caller wants to carry beside a turn.
    ///
    /// Runtime-only, as upstream's is — it travels with the message and no provider sees it, so it
    /// is a place to put a trace id rather than a way to reach the wire.
    pub fn metadata(mut self, metadata: Metadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Anything that reads as a part — dspy's role constructors are variadic and turn a bare string
    /// into an `LMTextPart` themselves, which is why its call sites read as prose.
    ///
    /// A Rust array holds one type, so a message mixing prose and an image names its parts:
    ///
    /// ```
    /// # use dsrust::{LmMessage, LmPart};
    /// LmMessage::user(["What is the capital of France?"]);
    /// LmMessage::user([
    ///     LmPart::text("Describe this image."),
    ///     LmPart::image_url("https://example.com/a.jpg"),
    /// ]);
    /// ```
    pub fn new(
        role: impl Into<String>,
        parts: impl IntoIterator<Item = impl Into<LmPart>>,
    ) -> Self {
        Self {
            role: role.into(),
            parts: parts.into_iter().map(Into::into).collect(),
            name: None,
            metadata: Metadata::new(),
        }
    }

    /// dspy `User`.
    pub fn user(parts: impl IntoIterator<Item = impl Into<LmPart>>) -> Self {
        Self::new("user", parts)
    }

    /// dspy `Assistant`.
    pub fn assistant(parts: impl IntoIterator<Item = impl Into<LmPart>>) -> Self {
        Self::new("assistant", parts)
    }

    /// dspy `System`.
    pub fn system(parts: impl IntoIterator<Item = impl Into<LmPart>>) -> Self {
        Self::new("system", parts)
    }

    /// dspy `Developer`: the o1-family role that replaces `system`, sent when
    /// `LM(use_developer_role=True)`.
    pub fn developer(parts: impl IntoIterator<Item = impl Into<LmPart>>) -> Self {
        Self::new("developer", parts)
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

/// The system prompt a message list carries, or `""` when it has none.
///
/// Only a lone text part is prose; a multi-part system message collapses to blocks the way
/// [`content_of`](super::content_of) does, and reads as no system prompt.
///
/// One read rather than one per holder. A request, a scripted model's record and a test's recorder
/// all ask this of the same list, and a copy of the answer is what drifts from it.
pub fn system_of(messages: &[LmMessage]) -> &str {
    messages
        .iter()
        .find(|message| message.role == "system")
        .and_then(|message| match message.parts.as_slice() {
            [only] => only.as_text(),
            _ => None,
        })
        .unwrap_or("")
}

/// The conversation without its leading system message — what a reader means by "the second turn"
/// when counting from the first thing anyone said.
///
/// Only the *leading* one: a system message later in the list is part of the conversation, and
/// dropping it would silently renumber everything after it.
pub fn after_system(messages: &[LmMessage]) -> &[LmMessage] {
    match messages.split_first() {
        Some((first, rest)) if first.role == "system" => rest,
        _ => messages,
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
    /// dspy 3.3.0's `strict`: OpenAI's structured-tool flag, carried on the spec and written to a
    /// wire only when the caller set it. Serialised even as `None`, because it is a field of the
    /// typed request rather than of the body — `null` is what upstream's own dump writes.
    #[serde(default)]
    pub strict: Option<bool>,
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
            strict: None,
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

/// dspy's role constructors, spelled as upstream spells them.
///
/// `dspy.User(...)` is a free function carrying `# noqa: N802` — Python suppressing its own naming
/// lint to keep the name capitalised. These do the same with `#[allow(non_snake_case)]`, which is the
/// trade `Predict!` and `ChainOfThought!` already make: a reader moving between the two languages
/// should not have to translate a name.
///
/// The inherent [`LmMessage::user`] and friends are the same thing under Rust's own conventions.
/// Neither is a wrapper for the other's benefit; they are two spellings of one constructor.
///
/// ```
/// # use dsrust::{Assistant, User};
/// User(["What is DSPy?"]);
/// Assistant(["A framework for programming LM pipelines."]);
/// ```
pub mod roles {
    use super::{LmMessage, LmPart};

    /// dspy `System`: model-level instructions — tone, scope, formatting rules.
    #[allow(non_snake_case)]
    pub fn System(parts: impl IntoIterator<Item = impl Into<LmPart>>) -> LmMessage {
        LmMessage::system(parts)
    }

    /// dspy `Developer`: instructions between system guidance and user content, for a provider that
    /// takes the `developer` role. See [`use_developer_role`](crate::lm::LmBuilder::use_developer_role).
    #[allow(non_snake_case)]
    pub fn Developer(parts: impl IntoIterator<Item = impl Into<LmPart>>) -> LmMessage {
        LmMessage::developer(parts)
    }

    /// dspy `User`: the request or the data to answer about.
    #[allow(non_snake_case)]
    pub fn User(parts: impl IntoIterator<Item = impl Into<LmPart>>) -> LmMessage {
        LmMessage::user(parts)
    }

    /// dspy `Assistant`: what the model said, for a turn a caller is replaying rather than one it
    /// just produced — an [`LmResponse`](crate::LmResponse) handed to
    /// [`call`](crate::ChatModel::call) becomes this on its own.
    #[allow(non_snake_case)]
    pub fn Assistant(parts: impl IntoIterator<Item = impl Into<LmPart>>) -> LmMessage {
        LmMessage::assistant(parts)
    }
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

    /// The two reads answer different questions and are not each other's complement, which is the
    /// part worth pinning: `system_of` takes the first `system` message wherever it sits — what a
    /// provider lifting the prompt into its own field wants — while `after_system` strips only a
    /// *leading* one, because a system message later in a conversation is part of it and dropping
    /// it would renumber every turn after it.
    #[test]
    fn the_system_read_and_the_turn_read_disagree_on_a_late_system_message() {
        let messages = vec![
            LmMessage::user(["hello"]),
            LmMessage::system(["be brief"]),
            LmMessage::assistant(["hi"]),
        ];
        assert_eq!(system_of(&messages), "be brief");
        assert_eq!(
            after_system(&messages).len(),
            3,
            "none of it was a preamble"
        );
    }

    #[test]
    fn a_leading_system_message_is_the_prompt_and_not_a_turn() {
        let messages = vec![LmMessage::system(["be brief"]), LmMessage::user(["hello"])];
        assert_eq!(system_of(&messages), "be brief");
        let turns = after_system(&messages);
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].role, "user");
    }

    /// A multi-part system message has no prose to lift, so it reads as no system prompt rather
    /// than as its first part — which would send half a prompt to a provider that carries one.
    #[test]
    fn a_multi_part_system_message_reads_as_no_prompt() {
        let messages = vec![LmMessage::system(["be", "brief"])];
        assert_eq!(system_of(&messages), "");
    }

    #[test]
    fn a_list_with_no_system_message_is_all_turns() {
        let messages = vec![LmMessage::user(["hello"])];
        assert_eq!(system_of(&messages), "");
        assert_eq!(after_system(&messages).len(), 1);
    }
}
