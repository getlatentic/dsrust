//! Asking a model directly, with no signature: dspy's `BaseLM.__call__`.
//!
//! What each case asserts is the *conversation that reached the wire*, not the return value. The
//! whole of `call` is the normalisation between what a caller wrote and what a provider is sent, so
//! a case reading only the answer would pass for a `call` that threw the input away.

use std::sync::Mutex;

use dsrust::lm::{ChatModel, api};
use dsrust::{Assistant, Developer, LmMessage, LmPart, LmRequest, LmResponse, System, User, items};

/// A model that answers a fixed reply and keeps what it was asked.
#[derive(Default)]
struct Recording {
    asked: Mutex<Vec<api::LmRequest>>,
}

impl ChatModel for Recording {
    async fn forward(&self, request: &api::LmRequest) -> anyhow::Result<api::LmResponse> {
        self.asked
            .lock()
            .expect("not poisoned")
            .push(request.clone());
        Ok(api::LmResponse::text("Paris."))
    }
}

impl Recording {
    /// The roles of the last conversation sent, which is what the normalisation decides.
    fn last_roles(&self) -> Vec<String> {
        self.asked
            .lock()
            .expect("not poisoned")
            .last()
            .expect("something was asked")
            .messages
            .iter()
            .map(|message| message.role.clone())
            .collect()
    }

    fn last_messages(&self) -> Vec<LmMessage> {
        self.asked
            .lock()
            .expect("not poisoned")
            .last()
            .expect("something was asked")
            .messages
            .clone()
    }
}

/// The shortest thing a caller can write reaches the model as one user turn.
#[tokio::test]
async fn a_bare_string_is_asked_as_one_user_turn() {
    let model = Recording::default();
    let answered = model
        .call(["What is the capital of France?"])
        .await
        .expect("it answers");

    assert_eq!(answered.first_text(), "Paris.");
    assert_eq!(model.last_roles(), vec!["user"]);
    assert_eq!(
        model.last_messages()[0].text().as_deref(),
        Some("What is the capital of France?")
    );
}

/// Prose and an image are one multimodal turn. Two turns would ask the model to describe nothing and
/// then hand it an image with no question.
#[tokio::test]
async fn prose_and_an_image_are_one_turn() {
    let model = Recording::default();
    model
        .call(items![
            "Describe this image.",
            LmPart::image_url("https://example.com/a.jpg")
        ])
        .await
        .expect("it answers");

    assert_eq!(model.last_roles(), vec!["user"]);
    assert_eq!(model.last_messages()[0].parts.len(), 2);
}

/// The thing this story exists for: a reply goes straight back into the next call. Before `call` a
/// caller had to take the response apart and rebuild the assistant turn by hand.
#[tokio::test]
async fn a_previous_reply_is_handed_back_in() {
    let model = Recording::default();
    let first = model
        .call(["Capital of France?"])
        .await
        .expect("it answers");

    model
        .call(items![
            User(["Capital of France?"]),
            first,
            User(["And of Belgium?"]),
        ])
        .await
        .expect("it answers again");

    assert_eq!(model.last_roles(), vec!["user", "assistant", "user"]);
    assert_eq!(
        model.last_messages()[1].text().as_deref(),
        Some("Paris."),
        "the model's own words, as the assistant turn they were"
    );
}

/// A multi-turn conversation written by hand is sent as written, in order.
#[tokio::test]
async fn a_written_conversation_is_sent_in_order() {
    let model = Recording::default();
    model
        .call(items![System(["Be brief."]), User(["Hello?"])])
        .await
        .expect("it answers");

    assert_eq!(model.last_roles(), vec!["system", "user"]);
}

/// `call` is defaulted on the trait, so a model that implements only `forward` has it. That is the
/// same reason upstream decorates `__call__` on `BaseLM` rather than in each subclass.
#[tokio::test]
async fn a_model_implementing_only_forward_still_has_call() {
    struct BareMinimum;
    impl ChatModel for BareMinimum {
        async fn forward(&self, request: &api::LmRequest) -> anyhow::Result<api::LmResponse> {
            Ok(api::LmResponse::text(format!(
                "{} turn(s)",
                request.messages.len()
            )))
        }
    }

    let answered = BareMinimum.call(["anything"]).await.expect("it answers");
    assert_eq!(answered.first_text(), "1 turn(s)");
}

/// The two doors converge on one type, which is the reason `Predict` needed no change: an adapter
/// enters through `from_messages` and a direct call through `from_items`, and below that they are the
/// same request.
#[test]
fn both_doors_build_the_same_request() {
    let written =
        LmRequest::from_messages("openai/gpt-4o-mini", vec![User(["Capital of France?"])]);
    let direct = LmRequest::from_items("openai/gpt-4o-mini", ["Capital of France?"])
        .expect("one string normalises");
    assert_eq!(written, direct);
}

/// A caller reaches all of this from the crate root. `LmMessage` was previously only at
/// `dsrust::lm::api::LmMessage`, which is a path nobody would guess.
#[test]
fn the_vocabulary_is_at_the_crate_root() {
    let _: LmPart = LmPart::text("hello");
    let _: LmRequest =
        LmRequest::from_items("openai/gpt-4o-mini", ["hello"]).expect("one string normalises");
    let _: LmResponse = LmResponse::text("hi");
}

/// dspy's own spelling and Rust's are two spellings of one constructor, not a wrapper and a wrapped.
/// `dspy.User("…")` is a free function carrying `# noqa: N802`, so these carry
/// `#[allow(non_snake_case)]` — the same trade `Predict!` already makes.
#[test]
fn dspys_role_names_are_the_inherent_constructors() {
    assert_eq!(User(["Hello?"]), LmMessage::user(["Hello?"]));
    assert_eq!(Assistant(["Paris."]), LmMessage::assistant(["Paris."]));
    assert_eq!(System(["Be brief."]), LmMessage::system(["Be brief."]));
    assert_eq!(
        Developer(["Be brief."]),
        LmMessage::developer(["Be brief."])
    );
    assert_eq!(Developer(["x"]).role, "developer");
}
