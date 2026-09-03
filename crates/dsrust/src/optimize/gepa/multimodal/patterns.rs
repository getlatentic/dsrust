//! The feedback summary dspy prepends to a multimodal reflective dataset.
//!
//! A keyword scan, not a classification: each example's `Feedback` is lowercased and tested for a
//! substring from three lists. The lists are scanned independently, so one sentence can count as an
//! error, a success and a knowledge gap at once — and a substring is enough, so "wellington"
//! counts as a success because it contains "well".

use gepa::Reflective;

use crate::optimize::gepa::proposer::ReflectiveDataset;

const ERRORS: [&str; 5] = ["incorrect", "wrong", "error", "failed", "missing"];
const SUCCESSES: [&str; 5] = ["correct", "good", "accurate", "well", "successfully"];
/// dspy's fourth bucket, `task_specific_guidance`, is declared and never filled — no keyword list
/// writes to it — so it can never make the summary appear.
const KNOWLEDGE: [&str; 5] = ["should know", "domain", "specific", "context", "background"];

/// The `## Feedback Pattern Analysis` block, or `None` when no example matched anything.
///
/// Absent rather than empty: upstream only prepends when `analysis["summary"]` is truthy, and the
/// summary is only written when one of the three buckets is non-empty.
pub(super) fn summary(dataset: &ReflectiveDataset) -> Option<String> {
    let (mut errors, mut successes, mut knowledge) = (0usize, 0usize, 0usize);
    for sample in dataset {
        // `example.get("Feedback", "")` — an example without the section contributes nothing, and
        // one whose section is not a plain string cannot be lowercased in Python either.
        let feedback = match section(sample, "Feedback") {
            Some(text) => text.to_lowercase(),
            None => continue,
        };
        errors += usize::from(ERRORS.iter().any(|word| feedback.contains(word)));
        successes += usize::from(SUCCESSES.iter().any(|word| feedback.contains(word)));
        knowledge += usize::from(KNOWLEDGE.iter().any(|word| feedback.contains(word)));
    }
    if errors == 0 && successes == 0 && knowledge == 0 {
        return None;
    }
    let mut parts = vec!["## Feedback Pattern Analysis\n".to_owned()];
    if errors > 0 {
        parts.push(format!("**Common Issues Found ({errors} examples):**"));
        parts.push(
            "Focus on preventing these types of mistakes in the new instruction.\n".to_owned(),
        );
    }
    if successes > 0 {
        parts.push(format!(
            "**Successful Approaches Found ({successes} examples):**"
        ));
        parts.push("Build on these successful strategies in the new instruction.\n".to_owned());
    }
    if knowledge > 0 {
        parts.push(format!(
            "**Domain Knowledge Needs Identified ({knowledge} examples):**"
        ));
        parts.push("Include this specialized knowledge in the new instruction.\n".to_owned());
    }
    Some(parts.join("\n"))
}

fn section<'a>(sample: &'a [(String, Reflective)], name: &str) -> Option<&'a str> {
    match sample.iter().find(|(key, _)| key == name).map(|(_, v)| v) {
        Some(Reflective::Text(text)) => Some(text),
        _ => None,
    }
}
