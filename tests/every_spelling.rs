//! Every way a task can be declared and asked, so the shapes in the README are ones that compile.
//!
//! Two ways to declare a signature (field names in a string, or a struct), two modules that ask
//! it (`Predict`, `ChainOfThought`), and both the long spelling and the macro. A change that
//! makes one of them read differently fails here rather than in someone's editor.

use std::sync::Arc;

use dsrs::{
    Ask, ChainOfThought, DummyLM, Example, Module, Predict, Signature, call, chain_of_thought,
    example, input, predict,
};

/// One in, one out.
#[derive(Signature)]
/// Answer the question.
struct QA {
    #[input]
    question: String,
    #[output]
    answer: String,
}

/// Two in, two out.
#[derive(Signature)]
/// Draft a haiku and say what it is about.
struct Haiku {
    #[input]
    subject: String,
    #[input]
    tone: String,
    #[output]
    haiku: String,
    #[output]
    mood: String,
}

/// One model for the whole file, answering by what the request mentions.
///
/// The configured model is process-wide and these tests run at once, so a model that answered in
/// order would hand a test another test's reply. Keying by content makes the answers independent
/// of who asks first.
fn install() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        dsrs::lm::global::configure_model(
            reqwest::Client::new(),
            Arc::new(DummyLM::keyed([
                ("capital of France?", example! { answer: "Paris" }),
                ("capital of Germany?", example! { answer: "Berlin" }),
                (
                    "computer science",
                    example! { haiku: "silicon dreaming", mood: "wry" },
                ),
                (
                    "quantum computing",
                    example! { haiku: "qubits entangle", mood: "curious" },
                ),
                (
                    "a calm colour?",
                    example! { reasoning: "cold colours read calm", answer: "blue" },
                ),
                (
                    "machine learning",
                    example! { reasoning: "gradients descend", haiku: "weights settle down", mood: "patient" },
                ),
            ])),
        );
    });
}

// ---------------------------------------------------------------------------
// A signature written as field names
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_string_signature_one_in_one_out() {
    install();

    // Long: build the signature, then the module, then name the inputs.
    let signature: Signature = "question -> answer".parse().expect("parses");
    let qa = Predict::from_signature(signature);
    let out = qa
        .forward(input! { question: "capital of France?" })
        .await
        .expect("asks");
    assert_eq!(out.get("answer").unwrap(), "Paris");

    // Short: the spelling is checked as this test compiles.
    let qa = predict!("question -> answer");
    let out = call!(qa, question = "capital of France?")
        .await
        .expect("asks");
    assert_eq!(out.get("answer").unwrap(), "Paris");
}

#[tokio::test]
async fn a_string_signature_two_in_two_out() {
    install();

    let qa = predict!("subject, tone -> haiku, mood");
    let out = call!(qa, subject = "computer science", tone = "wry")
        .await
        .expect("asks");
    assert_eq!(out.get("haiku").unwrap(), "silicon dreaming");
    assert_eq!(out.get("mood").unwrap(), "wry");
}

#[tokio::test]
async fn a_string_signature_through_chain_of_thought() {
    // Chain of thought asks for a leading `reasoning` field and keeps it out of the answer.
    install();

    let picked = chain_of_thought!("question -> answer");
    let out = call!(picked, question = "a calm colour?")
        .await
        .expect("asks");
    assert_eq!(out.get("answer").unwrap(), "blue");
}

// ---------------------------------------------------------------------------
// A signature written as a struct
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_derived_signature_one_in_one_out() {
    install();

    // Long: the module for the task, asked with the task's own inputs struct, answering with
    // its own outputs struct.
    let qa = Predict::<QA>::new();
    let out = qa
        .call_inputs(&QAInputs {
            question: "capital of Germany?".into(),
        })
        .await
        .expect("asks");
    assert_eq!(out.answer, "Berlin");

    // Short: the same module through the macro, asked the way a string signature is asked.
    let qa = predict!(QA);
    let out = call!(qa, question = "capital of France?")
        .await
        .expect("asks");
    assert_eq!(out.answer, "Paris");
}

#[tokio::test]
async fn a_derived_signature_two_in_two_out() {
    install();

    let poet = predict!(Haiku);
    let out = call!(poet, subject = "quantum computing", tone = "wry")
        .await
        .expect("asks");
    assert_eq!(out.haiku, "qubits entangle");
    assert_eq!(out.mood, "curious");

    // One invocation naming the task and filling it, which evaluates to the call itself.
    let out = predict!(Haiku {
        subject: "quantum computing",
        tone: "wry"
    })
    .await
    .expect("asks");
    assert_eq!(out.haiku, "qubits entangle");
}

#[tokio::test]
async fn a_derived_signature_through_chain_of_thought() {
    install();

    let out = chain_of_thought!(Haiku {
        subject: "machine learning",
        tone: "patient"
    })
    .await
    .expect("asks");
    assert_eq!(out.haiku, "weights settle down");
    assert_eq!(out.mood, "patient");
}

/// Whichever way it was declared, it is the same type and the same trait.
#[tokio::test]
async fn both_forms_are_one_type_and_one_module() {
    fn is_a_module<M: Module>(_: &M) {}
    fn is_askable<A: Ask>(_: &A) {}

    let declared = predict!("question -> answer");
    let derived = predict!(QA);
    let thinking = chain_of_thought!(QA);

    is_a_module(&declared);
    is_a_module(&derived);
    is_a_module(&thinking);
    is_askable(&declared);
    is_askable(&derived);
}
