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

/// dspy `dspy.User(*parts)`: one user turn from parts written positionally.
///
/// The function [`User`](fn@crate::User) takes an iterable, because that is what a Rust function can
/// take. Upstream's takes `*parts`, and the difference shows the moment the parts differ in type:
/// `["Describe this:", image]` is an array, and an array holds one type. The macro converts each
/// expression on its own, so a string and an [`Image`](crate::Image) sit side by side the way they
/// do in `dspy.User("Describe this:", image)`.
///
/// Thin on purpose — it expands to the same [`User`](fn@crate::User) the typed API offers, so nothing
/// this crate decides lives inside a macro.
///
/// **An image part here is [`LmPart::image_url`](crate::lm::api::LmPart::image_url), not
/// [`Image`](crate::Image)** — and the pinned dspy documents the opposite. `dspy.User`'s own
/// docstring at 3.3.0b1 shows `dspy.User("Describe this.", dspy.Image(url))`, which raises:
/// `TypeError: Cannot convert <class 'dspy.adapters.types.image.Image'> to an LMPart.` Measured, not
/// read. Upstream's main has since rewritten every one of those examples to `LMImagePart(url=…)`
/// while leaving `_coerce_part` byte-identical, so the docstring was the error and this spelling is
/// the fixed one. `Image` is a *signature field* type; a message part is a different layer.
///
/// ```no_run
/// # use dsrust::{Assistant, ChatModel, LM, System, User, items};
/// # async fn ask(lm: LM) -> anyhow::Result<()> {
/// # let image = dsrust::lm::api::LmPart::image_url("https://example.com/cat.png");
/// lm.call(items![
///     System!["You are terse."],
///     User!["Describe this:", image],
///     Assistant!["A cat."],
/// ])
/// .await?;
/// # Ok(())
/// # }
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! User {
    ($($part:expr),* $(,)?) => {
        $crate::User([$($crate::lm::api::LmPart::from($part)),*])
    };
}

/// dspy `dspy.Assistant(*parts)`: a turn the model took, replayed. See [`User!`](crate::User!).
#[macro_export]
#[allow(non_snake_case)]
macro_rules! Assistant {
    ($($part:expr),* $(,)?) => {
        $crate::Assistant([$($crate::lm::api::LmPart::from($part)),*])
    };
}

/// dspy `dspy.System(*parts)`: the instruction before the conversation. See [`User!`](crate::User!).
#[macro_export]
#[allow(non_snake_case)]
macro_rules! System {
    ($($part:expr),* $(,)?) => {
        $crate::System([$($crate::lm::api::LmPart::from($part)),*])
    };
}

/// dspy `dspy.Developer(*parts)`: the o1 family's system role. See [`User!`](crate::User!).
#[macro_export]
#[allow(non_snake_case)]
macro_rules! Developer {
    ($($part:expr),* $(,)?) => {
        $crate::Developer([$($crate::lm::api::LmPart::from($part)),*])
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
/// ```
/// use dsrust::lm::api::{LmItem, LmMessage, LmPart, messages_from_items};
///
/// // Bare parts become one turn — `lm("Describe this.", image)` is a single multimodal message.
/// let one = messages_from_items([LmPart::text("Describe this."), LmPart::text("(an image)")])
///     .expect("bare parts are one turn");
/// assert_eq!(one.len(), 1);
///
/// // Turns stay separate.
/// let two = messages_from_items([
///     LmMessage::user(vec![LmPart::text("a")]),
///     LmMessage::user(vec![LmPart::text("b")]),
/// ])
/// .expect("two turns");
/// assert_eq!(two.len(), 2);
///
/// // A turn *beside* a bare part is the one shape upstream refuses, and so does this — flattening
/// // it would silently turn two turns into one.
/// let mixed: Vec<LmItem> = vec![
///     LmMessage::user(vec![LmPart::text("a")]).into(),
///     LmPart::text("b").into(),
/// ];
/// assert!(messages_from_items(mixed).is_err());
/// ```
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

#[cfg(test)]
mod macros {
    //! The variadic spelling, which exists because an array holds one type and a turn's parts
    //! need not. Upstream writes `dspy.User("Describe this:", image)`; the function form here
    //! cannot take that, and the macro can.

    use crate::lm::api::LmPart;
    use crate::{Assistant, System, User};

    /// Parts of different types in one turn — the case the function form cannot express, because
    /// `["Describe this:", LmPart::image_url(…)]` would need `&str` and `LmPart` to be one type.
    /// The macro converts each expression on its own, so they need not be.
    #[test]
    fn parts_of_different_types_sit_in_one_turn() {
        let turn = User![
            "Describe this:",
            LmPart::image_url("https://example.com/cat.png")
        ];
        assert_eq!(turn.role, "user");
        assert_eq!(turn.parts.len(), 2);
        assert_eq!(turn.parts[0], LmPart::text("Describe this:"));
    }

    /// The macro is the function: same role, same parts, so nothing this crate decides lives
    /// inside a macro.
    #[test]
    fn the_macro_is_the_function() {
        assert_eq!(User!["a"], User([LmPart::text("a")]));
        assert_eq!(Assistant!["b"], Assistant([LmPart::text("b")]));
        assert_eq!(System!["c"], System([LmPart::text("c")]));
    }

    /// A whole conversation reads the way upstream's does, nested rather than flattened.
    #[test]
    fn a_conversation_nests_the_way_dspy_writes_one() {
        let conversation = items![System!["terse"], User!["hello"], Assistant!["hi"]];
        let messages = super::messages_from_items(conversation).expect("all turns");
        let roles: Vec<&str> = messages.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, ["system", "user", "assistant"]);
    }

    /// dspy's `name=` keyword reaches the OpenAI wire as `messages[].name`, right after the role —
    /// measured against `to_openai_chat_request`, which emits
    /// `{"role": "user", "name": "alice", "content": "hello"}`.
    ///
    /// It is what keeps two `user` turns apart in a multi-agent transcript, and nothing could set
    /// it here until the builder existed: the wire renderer had emitted it all along.
    #[test]
    fn a_speakers_name_reaches_the_wire() {
        let request = crate::lm::api::LmRequest::from_items(
            "openai/gpt-4o-mini",
            [User!["hello"].name("alice"), Assistant!["hi"].name("bot")],
        )
        .expect("all turns");

        let wire = request.wire_messages();
        assert_eq!(wire[0]["role"], "user");
        assert_eq!(wire[0]["name"], "alice");
        assert_eq!(wire[1]["name"], "bot");
    }

    /// A turn with no name carries no `name` key at all, rather than a null — upstream omits it,
    /// and a provider that rejects unknown nulls would refuse the call.
    #[test]
    fn an_unnamed_turn_carries_no_name_key() {
        let request = crate::lm::api::LmRequest::from_items("m", [User!["hello"]]).expect("a turn");
        assert!(
            request.wire_messages()[0].get("name").is_none(),
            "an unnamed turn should not carry the key"
        );
    }
}
