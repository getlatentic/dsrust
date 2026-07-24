//! Every way a task can be declared and asked, so the shapes in the README are ones that compile.
//!
//! Two ways to declare a signature (field names in a string, or a struct), two modules that ask
//! it (`Predict`, `ChainOfThought`), and both the long spelling and the macro. A change that
//! makes one of them read differently fails here rather than in someone's editor.

use std::sync::Arc;

use dsrust::{
    Ask, DummyLM, Forward, Module, Predict, Signature, call, chain_of_thought, example, input,
    predict,
};

/// One in, one out.
///
/// The fields are read through the generated companions rather than the struct itself, which
/// is what the derive exists to do.
#[allow(dead_code)]
#[derive(Signature)]
/// Answer the question.
struct QA {
    #[input]
    question: String,
    #[output]
    answer: String,
}

/// Two in, two out.
#[allow(dead_code)]
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
        dsrust::lm::global::configure_model(
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
                // The two steps of the caller-defined module below, each keyed by what it is
                // handed rather than by what the program was originally asked.
                ("winter mornings", example! { angle: "stillness before the day" }),
                (
                    "stillness before the day",
                    example! { haiku: "frost holds the window" },
                ),
                ("one word please", example! { answer: "Paris" }),
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

// ---------------------------------------------------------------------------
// A module of your own
// ---------------------------------------------------------------------------

/// Two steps composed into one program, the way a caller writes theirs.
///
/// The derive supplies what Python inherits: the walk an optimizer works through, and being
/// callable through `call!`. What is left is the part only the author knows.
#[derive(dsrust::Module)]
struct Outline {
    plan: Predict,
    write: Predict,
}

impl Outline {
    fn new() -> Self {
        Self {
            plan: predict!("subject -> angle"),
            write: predict!("angle -> haiku"),
        }
    }
}

impl Forward for Outline {
    async fn forward(&self, inputs: dsrust::Example) -> anyhow::Result<dsrust::Prediction> {
        let angle = self.plan.forward(inputs).await?;
        let handed = input! { angle: angle.get("angle").cloned().unwrap_or_default() };
        self.write.forward(handed).await
    }
}

#[tokio::test]
async fn a_module_of_your_own_composes_and_is_optimizable() {
    install();

    let mut mine = Outline::new();
    let out = call!(mine, subject = "winter mornings")
        .await
        .expect("asks");
    assert_eq!(out.get("haiku").unwrap(), "frost holds the window");

    // The seam that matters: an optimizer can see both steps and write to each.
    let named: Vec<String> = mine
        .named_predictors()
        .into_iter()
        .map(|predictor| predictor.name)
        .collect();
    assert_eq!(named, ["plan", "write"]);
}

// ---------------------------------------------------------------------------
// Wrapping a module in another module
// ---------------------------------------------------------------------------

/// A reward is a named function in dspy's own example, and reads better than a closure written
/// inline between the three arguments around it.
fn one_word(_inputs: &dsrust::Example, out: &dsrust::Prediction) -> f64 {
    match out.get("answer").and_then(|answer| answer.as_str()) {
        Some(answer) if answer.split_whitespace().count() == 1 => 1.0,
        _ => 0.0,
    }
}

/// `BestOfN` takes a *module*, not a signature — upstream's is
/// `BestOfN(module=qa, N=…, reward_fn=…, threshold=…)`. There is no signature to hand it; the
/// signature lives in whatever it wraps.
#[tokio::test]
async fn best_of_n_wraps_a_module_and_is_called_like_one() {
    install();

    let best = dsrust::BestOfN::new(
        predict!("question -> answer"),
        3,
        |_inputs: &dsrust::Example, prediction: &dsrust::Prediction| match prediction
            .get("answer")
            .and_then(|answer| answer.as_str())
        {
            Some(answer) if answer.split_whitespace().count() == 1 => 1.0,
            _ => 0.0,
        },
        1.0,
    );

    let out = call!(best, question = "one word please")
        .await
        .expect("asks");
    assert_eq!(out.get("answer").unwrap(), "Paris");
}

/// And being a `Module` is the point: it nests, and an optimizer's walk reaches the predictor
/// inside it rather than stopping at the wrapper.
#[tokio::test]
async fn best_of_n_is_a_module_an_optimizer_can_walk() {
    use dsrust::Module;

    let mut best = dsrust::best_of_n!(
        predict!("question -> answer"),
        n = 2,
        reward = one_word,
        threshold = 1.0
    );
    assert_eq!(best.named_predictors().len(), 1);
}
