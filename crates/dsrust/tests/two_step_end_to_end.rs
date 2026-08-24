//! A two-step program end to end: the task model answers in prose, a second model reads the
//! fields out of it, and the caller sees only the fields.

use std::sync::Arc;

use dsrust::adapter::parse::FieldMismatch;
use dsrust::signature::{InField, OutField, Signature};
use dsrust::{DummyLM, Predict, TwoStepAdapter, example};
use serde_json::Value;

/// What dspy's own error types render, captured by running the pinned dspy.
fn golden() -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/conformance/lm/exceptions.json");
    serde_json::from_str(&std::fs::read_to_string(&path).expect("the golden is committed"))
        .expect("the golden parses")
}

fn signature() -> Signature {
    let mut signature = Signature::single_input(
        "Answer the question.",
        vec![OutField {
            name: "answer".into(),
            desc: "the reply".into(),
            ..Default::default()
        }],
    );
    signature.inputs = vec![InField {
        name: "question".into(),
        desc: "the ask".into(),
        ..Default::default()
    }];
    signature
}

#[tokio::test]
async fn the_task_model_answers_in_prose_and_a_second_model_names_the_fields() {
    // The task model is told nothing about markers, and answers as it likes.
    let task = DummyLM::new([example! { reply: "The capital of France is Paris." }])
        .fallback(example! { reply: "The capital of France is Paris." });
    let extractor = Arc::new(DummyLM::new([example! { answer: "Paris" }]));

    let predict =
        Predict::from_signature(signature()).adapter(TwoStepAdapter::new(extractor.clone()));
    let value = predict
        .call_with(&task, "What is the capital of France?")
        .await
        .expect("the two-step run succeeds");

    assert_eq!(value["answer"], "Paris");
    assert_eq!(task.call_count(), 1, "the task model is asked once");
    assert_eq!(
        extractor.call_count(),
        1,
        "the extraction model is asked once"
    );
}

#[tokio::test]
async fn the_task_model_is_never_shown_a_wire_format() {
    // The whole point of the adapter: the model that solves the task is not also asked to
    // format. A marker or a brace in its prompt would put that burden back.
    let task = DummyLM::new([example! { reply: "Paris." }]).fallback(example! { reply: "Paris." });
    let extractor = Arc::new(DummyLM::new([example! { answer: "Paris" }]));

    Predict::from_signature(signature())
        .adapter(TwoStepAdapter::new(extractor))
        .call_with(&task, "Where?")
        .await
        .expect("succeeds");

    let asked = task.asked();
    assert!(
        !asked[0].system().contains("[[ ##"),
        "got: {}",
        asked[0].system()
    );
    assert!(asked[0].system().starts_with("You are a helpful assistant"));
    assert_eq!(asked[0].last_message(), "question: Where?");
}

#[tokio::test]
async fn the_extraction_model_is_shown_the_first_reply_as_its_text_field() {
    let prose = "After consideration, the answer is Paris.";
    let task = DummyLM::new([example! { reply: prose }]).fallback(example! { reply: prose });
    let extractor = Arc::new(DummyLM::new([example! { answer: "Paris" }]));

    Predict::from_signature(signature())
        .adapter(TwoStepAdapter::new(extractor.clone()))
        .call_with(&task, "Where?")
        .await
        .expect("succeeds");

    // The extraction speaks the chat adapter over `text -> answer`, so the first reply arrives
    // as an ordinary marker section rather than as anything two-step-specific.
    let asked = extractor.asked();
    assert!(
        asked[0].last_message().starts_with("[[ ## text ## ]]\n"),
        "got: {}",
        asked[0].last_message()
    );
    assert!(asked[0].last_message().contains(prose));
    assert!(
        asked[0]
            .system()
            .contains("extract the fields from the text verbatim")
    );
}

/// A failed extraction reports what went wrong, against the error dspy raises for the same run.
///
/// The golden is captured by *running* `TwoStepAdapter` (`scripts/generate_exceptions_fixture.py`),
/// not by constructing an `AdapterParseError`, because the two things it pins are the adapter's
/// decisions and not the exception's: the message carries `f"…: {e}"` — the failure, not the reply
/// — and `lm_response` carries the **first** completion rather than the extraction's. This port had
/// the reply written where upstream writes the error, so the sentence a caller reads first named
/// the text and never said what was wrong with it.
#[tokio::test]
async fn a_failed_extraction_reports_the_failure_as_dspy_does() {
    let recorded = golden();
    let dspy = &recorded["two_step_extraction_failure"];
    let prefix = "Failed to parse response from the original completion: ";
    assert!(
        dspy["message"]
            .as_str()
            .expect("a message")
            .starts_with(prefix),
        "the golden's own prefix moved; dspy said: {}",
        dspy["message"]
    );

    let prose = "a reply that never answers the question";
    let task = DummyLM::new([example! { reply: prose }]).fallback(example! { reply: prose });
    let extractor = Arc::new(DummyLM::new([]).fallback(example! { unrelated: "nothing" }));

    let error = Predict::from_signature(signature())
        .adapter(TwoStepAdapter::new(extractor))
        .call_with(&task, "Where?")
        .await
        .expect_err("the extraction produced no `answer`");
    let reported = error
        .downcast_ref::<FieldMismatch>()
        .expect("the two-step failure is dspy's AdapterParseError");

    assert_eq!(
        reported.adapter_name,
        dspy["adapter_name"].as_str().expect("an adapter name"),
        "upstream names the two-step adapter, not the extraction's"
    );
    let message = reported.message.as_deref().expect("a message");
    assert!(message.starts_with(prefix), "got: {message}");

    // The half that was wrong: what follows the prefix is the *failure*. dspy's names the adapter
    // that failed last — the JSON one its `ChatAdapter.__call__` fell back to — and so must this,
    // which it cannot do unless the extraction falls back at all.
    assert!(
        dspy["message"]
            .as_str()
            .expect("a message")
            .contains("Adapter JSONAdapter"),
        "the golden no longer shows a fallback; dspy said: {}",
        dspy["message"]
    );
    assert!(
        message.contains("Adapter JSONAdapter"),
        "the extraction did not fall back to JSON as upstream's does: {message}"
    );

    // And `lm_response` is the first reply, which is the one a caller can act on.
    assert!(
        dspy["lm_response_is_the_first_reply"].as_bool() == Some(true),
        "the golden stopped carrying the first reply"
    );
    assert!(
        reported.lm_response.contains(prose),
        "got: {}",
        reported.lm_response
    );
}
