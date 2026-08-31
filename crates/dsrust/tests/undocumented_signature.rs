//! A signature nobody wrote a docstring for gets dspy's own default objective.
//!
//! `class QA(dspy.Signature)` with no docstring is ordinary DSPy — the `conversation_history`
//! tutorial writes exactly that — and upstream fills the objective in from the field names with
//! `_default_instructions`. The derive used to refuse, so the tutorial could not be ported.
//!
//! The string form always did this. Only the derive disagreed, which made the two spellings of
//! the same signature render different prompts.

use dsrust::Signature;
use dsrust::signature::SignatureSpec;

#[derive(Signature)]
struct QA {
    #[input]
    question: String,
    #[input]
    context: String,
    #[output]
    answer: String,
}

#[derive(Signature)]
/// Answer the question.
struct Documented {
    #[input]
    question: String,
    #[output]
    answer: String,
}

/// The sentence upstream writes, field for field.
#[test]
fn an_undocumented_signature_takes_dspys_default_objective() {
    assert_eq!(
        QA::signature().instructions,
        "Given the fields `question`, `context`, produce the fields `answer`."
    );
}

/// The two spellings of one signature agree, which is the reason this is not the derive's own
/// sentence.
#[test]
fn the_derive_and_the_string_form_write_the_same_objective() {
    let parsed = dsrust::signature::parse("question, context -> answer").expect("parses");
    assert_eq!(QA::signature().instructions, parsed.instructions);
}

/// A docstring still wins; the default is a fallback, not a replacement.
#[test]
fn a_docstring_is_still_the_objective() {
    assert_eq!(Documented::signature().instructions, "Answer the question.");
}

/// A field name that is not snake case still compiles without a warning a caller cannot silence.
///
/// A field's name is prompt text. `gepa_papillon` declares `response_A` and `response_B`, and
/// renaming them to suit Rust's style would write a different prompt for the same program. The
/// derive's companion structs re-declare every field, so `non_snake_case` fired inside code the
/// caller never wrote — an `#[allow]` on their own struct does not reach there.
///
/// This file is compiled with `#![deny(warnings)]` for the length of this test's struct, which is
/// what makes the assertion the compiler's rather than a reader's.
#[deny(non_snake_case)]
mod prompt_text_names {
    use dsrust::Signature;

    #[derive(Signature)]
    /// You are comparing the quality of two responses, given a user query.
    #[allow(dead_code, non_snake_case)]
    pub struct JudgeQuality {
        #[input]
        pub user_query: String,
        #[input]
        pub response_A: String,
        #[input]
        pub response_B: String,
        #[output]
        pub judgment: bool,
    }
}

/// And the name reaches the prompt as written.
#[test]
fn a_field_name_is_prompt_text_and_keeps_its_spelling() {
    use dsrust::signature::SignatureSpec;
    let signature = prompt_text_names::JudgeQuality::signature();
    let names: Vec<&str> = signature
        .inputs
        .iter()
        .map(|field| field.name.as_str())
        .collect();
    assert_eq!(names, ["user_query", "response_A", "response_B"]);
}
