//! A two-step program end to end: the task model answers in prose, a second model reads the
//! fields out of it, and the caller sees only the fields.

use std::sync::Arc;

use dsrs::signature::{FieldKind, InField, OutField, Signature};
use dsrs::{DummyLM, Predict, TwoStepAdapter, example};

fn signature() -> Signature {
    let mut signature = Signature::single_input(
        "Answer the question.",
        vec![OutField {
            name: "answer".into(),
            desc: "the reply".into(),
            kind: FieldKind::Str,
            values: None,
            schema: None,
        }],
    );
    signature.inputs = vec![InField {
        name: "question".into(),
        desc: "the ask".into(),
        kind: FieldKind::Str,
        values: None,
    }];
    signature
}

#[tokio::test]
async fn the_task_model_answers_in_prose_and_a_second_model_names_the_fields() {
    // The task model is told nothing about markers, and answers as it likes.
    let task = DummyLM::new([example! { reply: "The capital of France is Paris." }])
        .with_fallback(example! { reply: "The capital of France is Paris." });
    let extractor = Arc::new(DummyLM::new([example! { answer: "Paris" }]));

    let predict = Predict::new(signature()).with_adapter(TwoStepAdapter::new(extractor.clone()));
    let value = predict
        .call_with(
            &reqwest::Client::new(),
            &task,
            "What is the capital of France?",
        )
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
    let task =
        DummyLM::new([example! { reply: "Paris." }]).with_fallback(example! { reply: "Paris." });
    let extractor = Arc::new(DummyLM::new([example! { answer: "Paris" }]));

    Predict::new(signature())
        .with_adapter(TwoStepAdapter::new(extractor))
        .call_with(&reqwest::Client::new(), &task, "Where?")
        .await
        .expect("succeeds");

    let asked = task.asked();
    assert!(
        !asked[0].system.contains("[[ ##"),
        "got: {}",
        asked[0].system
    );
    assert!(asked[0].system.starts_with("You are a helpful assistant"));
    assert_eq!(asked[0].last_message(), "question: Where?");
}

#[tokio::test]
async fn the_extraction_model_is_shown_the_first_reply_as_its_text_field() {
    let prose = "After consideration, the answer is Paris.";
    let task = DummyLM::new([example! { reply: prose }]).with_fallback(example! { reply: prose });
    let extractor = Arc::new(DummyLM::new([example! { answer: "Paris" }]));

    Predict::new(signature())
        .with_adapter(TwoStepAdapter::new(extractor.clone()))
        .call_with(&reqwest::Client::new(), &task, "Where?")
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
            .system
            .contains("extract the fields from the text verbatim")
    );
}

#[tokio::test]
async fn a_failed_extraction_names_the_prose_it_was_reading() {
    // dspy points the caller at the first model's reply rather than the extraction's, because
    // an extraction that found nothing usually means the prose never said it.
    let prose = "a reply that never answers the question";
    let task = DummyLM::new([example! { reply: prose }]).with_fallback(example! { reply: prose });
    let extractor = Arc::new(DummyLM::new([]).with_fallback(example! { unrelated: "nothing" }));

    let error = Predict::new(signature())
        .with_adapter(TwoStepAdapter::new(extractor))
        .call_with(&reqwest::Client::new(), &task, "Where?")
        .await
        .expect_err("the extraction produced no `answer`");
    let shown = format!("{error:#}");
    assert!(
        shown.contains("Failed to parse response from the original completion"),
        "got: {shown}"
    );
    assert!(shown.contains(prose), "got: {shown}");
}
