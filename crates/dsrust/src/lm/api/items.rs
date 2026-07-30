//! dspy's `_messages_from_items`: the direct-call input shapes, normalised into a conversation.
//!
//! Upstream's is variadic and dynamically typed — `lm("hello")`, `lm(User(...), Assistant(...))`,
//! `lm(previous_response, User("and now?"))` — and it decides what a call meant by inspecting the
//! items. [`LmItem`] is that decision made in the type system, so a caller writes the same three
//! shapes and the branches below read off the enum rather than off `isinstance`.
//!
//! One of upstream's branches has deliberately no counterpart. `_messages_from_items` unwraps a
//! *single* item that is itself a list of messages, because Python cannot tell `lm(messages)` from
//! `lm(message)`. Here a conversation goes through
//! [`from_messages`](super::LmRequest::from_messages) and a run of items through
//! [`from_items`](super::LmRequest::from_items), so there is nothing to disambiguate — which is also
//! why `from_call`'s "pass messages or direct-call inputs, not both" is a `ValueError` there and
//! unwritable here.

use super::{LmMessage, LmPart, LmResponse};

/// One argument of a direct call: a whole turn, a whole previous reply, or a part of one message.
///
/// The three `From` impls are what let a call site stay prose — `"hello"`, an [`LmMessage`], or an
/// [`LmResponse`] handed straight back in.
#[derive(Clone, Debug, PartialEq)]
pub enum LmItem {
    Message(LmMessage),
    /// A previous reply, continued from. Upstream expands it to one assistant turn per output.
    Response(LmResponse),
    Part(LmPart),
}

impl From<LmMessage> for LmItem {
    fn from(message: LmMessage) -> Self {
        Self::Message(message)
    }
}

impl From<LmResponse> for LmItem {
    fn from(response: LmResponse) -> Self {
        Self::Response(response)
    }
}

impl From<LmPart> for LmItem {
    fn from(part: LmPart) -> Self {
        Self::Part(part)
    }
}

impl From<&str> for LmItem {
    fn from(text: &str) -> Self {
        Self::Part(LmPart::text(text))
    }
}

impl From<String> for LmItem {
    fn from(text: String) -> Self {
        Self::Part(LmPart::text(text))
    }
}

/// The arguments of a direct call, each read as an [`LmItem`] — dspy's variadic `lm(*items)`.
///
/// Rust has no varargs and an array holds one type, so a call mixing a turn, a previous reply and a
/// string needs each element converted. This is that conversion, for the same reason
/// [`call!`](crate::call) and [`input!`](crate::input) exist: the macro supplies what the language
/// does not.
///
/// ```no_run
/// # use dsrust::{ChatModel, LM, User, items};
/// # async fn ask(lm: LM) -> anyhow::Result<()> {
/// let answered = lm.call(items![User(["What is the capital of France?"])]).await?;
/// lm.call(items![answered, User(["And of Belgium?"])]).await?;
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! items {
    ($($item:expr),* $(,)?) => {
        [$($crate::lm::api::LmItem::from($item)),*]
    };
}

/// dspy `_messages_from_items`: what a run of direct-call arguments means.
///
/// Three branches, upstream's own:
///
/// * **Nothing at all** is one empty user message, not an empty conversation — upstream's
///   `items = ("",)`. A provider asked for nothing still gets a turn to answer.
/// * **Every item a turn or a reply** is a conversation, each [`LmResponse`] expanded into one
///   assistant turn per output.
/// * **Anything else** is one user message whose parts are the items, which is what makes
///   `lm("Describe this.", image)` a single multimodal turn rather than two.
pub fn messages_from_items(items: impl IntoIterator<Item = impl Into<LmItem>>) -> Vec<LmMessage> {
    let items: Vec<LmItem> = items.into_iter().map(Into::into).collect();
    if items.is_empty() {
        return vec![LmMessage::user([""])];
    }
    if items.iter().any(|item| matches!(item, LmItem::Part(_))) {
        return vec![LmMessage::user(items.into_iter().map(|item| match item {
            LmItem::Part(part) => part,
            // Unreachable given the guard above; a turn among parts would be a caller mixing the
            // two shapes, and reading it as its own text loses less than dropping it.
            LmItem::Message(message) => LmPart::text(message.text().unwrap_or_default()),
            LmItem::Response(response) => LmPart::text(response.first_text()),
        }))];
    }
    items
        .into_iter()
        .flat_map(|item| match item {
            LmItem::Message(message) => vec![message],
            LmItem::Response(response) => messages_from_response(&response),
            LmItem::Part(part) => vec![LmMessage::user([part])],
        })
        .collect()
}

/// dspy `_messages_from_response`: one assistant turn per output, so a reply handed back into the
/// next call reads as what the model said.
fn messages_from_response(response: &LmResponse) -> Vec<LmMessage> {
    response
        .outputs
        .iter()
        .map(|output| LmMessage::assistant(output.parts.clone()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lm::api::LmOutput;

    #[test]
    fn a_bare_string_is_one_user_turn() {
        let messages = messages_from_items(["What is the capital of France?"]);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");
        assert_eq!(
            messages[0].text().as_deref(),
            Some("What is the capital of France?")
        );
    }

    /// Several parts are one multimodal turn, not one turn each — the branch that makes
    /// `lm("Describe this.", image)` a single question about a single image.
    #[test]
    fn several_parts_are_one_turn() {
        let messages = messages_from_items([
            LmItem::from("Describe this image."),
            LmItem::Part(LmPart::image_url("https://example.com/a.jpg")),
        ]);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].parts.len(), 2);
    }

    /// A run of turns is the conversation it looks like.
    #[test]
    fn a_run_of_turns_is_a_conversation() {
        let messages = messages_from_items([
            LmItem::Message(LmMessage::system(["Be brief."])),
            LmItem::Message(LmMessage::user(["Hello?"])),
        ]);
        assert_eq!(
            messages.iter().map(|m| m.role.as_str()).collect::<Vec<_>>(),
            vec!["system", "user"]
        );
    }

    /// The point of the whole shape: a previous reply goes straight back in, as the assistant turn
    /// it was. Before this a caller had to take the response apart and rebuild it.
    #[test]
    fn a_previous_reply_continues_the_conversation() {
        let earlier = LmResponse {
            outputs: vec![LmOutput::text("Paris.")],
            ..LmResponse::default()
        };
        let messages = messages_from_items([
            LmItem::Message(LmMessage::user(["Capital of France?"])),
            LmItem::Response(earlier),
            LmItem::Message(LmMessage::user(["And of Belgium?"])),
        ]);
        assert_eq!(
            messages.iter().map(|m| m.role.as_str()).collect::<Vec<_>>(),
            vec!["user", "assistant", "user"]
        );
        assert_eq!(messages[1].text().as_deref(), Some("Paris."));
    }

    /// Upstream's `items = ("",)`: nothing asked is still a turn, so a provider has something to
    /// answer rather than an empty conversation to reject.
    #[test]
    fn nothing_at_all_is_one_empty_turn() {
        let messages = messages_from_items(Vec::<LmItem>::new());
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[0].parts.len(), 1);
    }
}
