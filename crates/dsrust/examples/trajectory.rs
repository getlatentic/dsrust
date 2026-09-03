//! The Rust half of `scripts/compare_trajectories.py`: one signature, two modules, one engine.
//!
//! Its opposite number is a dspy program declaring the same signature, and a proxy between both and
//! the engine records what each actually PUT. Run it through that script rather than directly —
//! alone it just asks a model a question.
//!
//! The declaration has to match the Python one field for field, including the instruction text,
//! because a difference here would read as a difference between the libraries.

use dsrust::{ChainOfThought, LM, Predict, Signature, configure};

#[derive(Signature)]
/// Write independent practice for this lesson step. The practice must set a
/// different task from the worked example a learner has just been shown the
/// answer to: keep the skill, change the values. Give the expected answer.
struct PracticeForStep {
    #[input]
    learning_goal: String,
    #[input]
    worked_example_problem: String,
    #[input]
    worked_example_answer: String,
    #[output]
    practice_question: String,
    #[output]
    expected_answer: String,
}

const GOAL: &str = "Order common fractions.";
const PROBLEM: &str = "Order 1/6, 1/3 and 1/2 from least to greatest.";
const ANSWER: &str = "1/6 < 1/3 < 1/2";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let base_url = std::env::var("TRAJECTORY_BASE_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:8098/v1".to_owned());
    let model = std::env::var("TRAJECTORY_MODEL").unwrap_or_else(|_| "gemma-4-e2b".to_owned());

    configure(
        // The key is how the proxy tells the two clients apart, and the cache is off so a second
        // run asks again rather than replaying an answer the proxy never saw.
        LM::new(format!("openai/{model}"))?
            .openai_base_url(&base_url)
            .openai_api_key("rust-dsrust")
            .cache(false),
    );

    let inputs = || PracticeForStepInputs {
        learning_goal: GOAL.to_owned(),
        worked_example_problem: PROBLEM.to_owned(),
        worked_example_answer: ANSWER.to_owned(),
    };

    // A reply this crate cannot parse is not a failure of the comparison: the ask was recorded
    // before the engine answered, which is the whole thing under test.
    for (module, answered) in [
        (
            "Predict",
            Predict!(PracticeForStep).call_inputs(&inputs()).await,
        ),
        (
            "ChainOfThought",
            ChainOfThought!(PracticeForStep)
                .call_inputs(&inputs())
                .await,
        ),
    ] {
        match answered {
            Ok(out) => println!("{module}: {}", out.practice_question),
            Err(error) => println!("{module}: {error}"),
        }
    }
    Ok(())
}
