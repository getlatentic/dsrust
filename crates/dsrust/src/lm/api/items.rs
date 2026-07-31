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

use anyhow::{Result, bail};

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
///   `lm("Describe this.", image)` a single multimodal turn rather than two. A turn or a reply
///   *among* those parts is the one shape upstream refuses, and so does this — see below.
///
/// Fallible for that last reason only. `_coerce_part` handles a part, a string and a tagged dict
/// and raises `TypeError` on anything else, so `lm(dspy.User("a"), "b")` — a message beside a bare
/// part — raises upstream rather than producing a conversation. Measured, not read: the branch
/// condition here (`any Part`) is exactly upstream's `not all(message or reply)`, so this arm is
/// reached by the same inputs. It used to flatten the message to its text and call itself
/// unreachable, which silently turned two turns into one.
pub fn messages_from_items(
    items: impl IntoIterator<Item = impl Into<LmItem>>,
) -> Result<Vec<LmMessage>> {
    let items: Vec<LmItem> = items.into_iter().map(Into::into).collect();
    if items.is_empty() {
        return Ok(vec![LmMessage::user([""])]);
    }
    if items.iter().any(|item| matches!(item, LmItem::Part(_))) {
        let mut parts = Vec::with_capacity(items.len());
        for item in items {
            parts.push(match item {
                LmItem::Part(part) => part,
                // dspy's `_coerce_part` raises for both, and its message names the type it could
                // not convert.
                LmItem::Message(_) => bail!("cannot convert a message to a part"),
                LmItem::Response(_) => bail!("cannot convert a reply to a part"),
            });
        }
        return Ok(vec![LmMessage::user(parts)]);
    }
    Ok(items
        .into_iter()
        .flat_map(|item| match item {
            LmItem::Message(message) => vec![message],
            LmItem::Response(response) => messages_from_response(&response),
            LmItem::Part(part) => vec![LmMessage::user([part])],
        })
        .collect())
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
        let messages =
            messages_from_items(["What is the capital of France?"]).expect("these items normalise");
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
        ])
        .expect("these items normalise");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].parts.len(), 2);
    }

    /// A run of turns is the conversation it looks like.
    #[test]
    fn a_run_of_turns_is_a_conversation() {
        let messages = messages_from_items([
            LmItem::Message(LmMessage::system(["Be brief."])),
            LmItem::Message(LmMessage::user(["Hello?"])),
        ])
        .expect("these items normalise");
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
        ])
        .expect("these items normalise");
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
        let messages = messages_from_items(Vec::<LmItem>::new()).expect("these items normalise");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[0].parts.len(), 1);
    }

    /// A turn beside a bare part is what upstream refuses, and the arm that refuses it was written
    /// as "unreachable" while being reached by exactly these inputs.
    ///
    /// Measured against dspy 3.3.0b1: `LMRequest.from_call(items=(dspy.User("a"), "b"))` raises
    /// `TypeError: Cannot convert <class 'LMMessage'> to an LMPart.` Flattening it here turned two
    /// turns into one user turn holding both, silently.
    #[test]
    fn a_turn_among_parts_is_refused_as_dspy_refuses_it() {
        let mixed = messages_from_items([
            LmItem::Message(LmMessage::user(["a"])),
            LmItem::Part(LmPart::text("b")),
        ]);
        assert!(
            mixed.is_err(),
            "a message beside a part should not normalise"
        );

        let replied = messages_from_items([
            LmItem::Response(LmResponse::text("a")),
            LmItem::Part(LmPart::text("b")),
        ]);
        assert!(
            replied.is_err(),
            "a reply beside a part should not normalise"
        );
    }

    /// The two shapes that *do* normalise still do, so the refusal above is not over-broad: every
    /// item a turn is a conversation, and every item a part is one multimodal turn.
    #[test]
    fn the_unmixed_shapes_still_normalise() {
        let conversation = messages_from_items([
            LmItem::Message(LmMessage::user(["a"])),
            LmItem::Message(LmMessage::assistant(["b"])),
        ])
        .expect("all turns");
        assert_eq!(conversation.len(), 2);

        let multimodal = messages_from_items([
            LmItem::Part(LmPart::text("look:")),
            LmItem::Part(LmPart::text("b")),
        ])
        .expect("all parts");
        assert_eq!(multimodal.len(), 1);
        assert_eq!(multimodal[0].parts.len(), 2);
    }
}
