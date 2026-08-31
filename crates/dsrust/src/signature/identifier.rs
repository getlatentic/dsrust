//! Whether a name in a signature string is one Python would have accepted.
//!
//! dspy parses a signature string with `ast.parse`, so a field name is a Python identifier or the
//! string does not parse at all: `1q -> a`, `q-x -> a`, `q x -> a` and `class -> a` are each a
//! `SyntaxError` upstream. This crate split on commas and took whatever was between them, so all
//! four became fields — and a field named `q-x` reaches the model as a marker it can never emit.
//!
//! The message cannot match upstream's, which is CPython's own parser talking. What must match is
//! the **boundary**: the set of strings accepted. Found by running thirty-five edge cases through
//! both, where fifteen differed and dsrust accepted every one dspy refused.

use anyhow::{Result, anyhow};

/// Python's reserved words. The soft keywords — `match`, `case`, `type`, `_` — are missing on
/// purpose: Python allows them as identifiers and so does this.
const KEYWORDS: [&str; 35] = [
    "False", "None", "True", "and", "as", "assert", "async", "await", "break", "class", "continue",
    "def", "del", "elif", "else", "except", "finally", "for", "from", "global", "if", "import",
    "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try", "while",
    "with", "yield",
];

/// A field name Python would have parsed, or the reason it would not have.
pub(crate) fn refuse_unless_identifier(name: &str, side: &str) -> Result<()> {
    if name.is_empty() {
        return Err(anyhow!(
            "Invalid signature format: an {side} field has no name."
        ));
    }
    if KEYWORDS.contains(&name) {
        return Err(anyhow!(
            "Invalid signature format: '{name}' is a reserved word and cannot name an {side} field."
        ));
    }
    if !is_identifier(name) {
        return Err(anyhow!(
            "Invalid signature format: '{name}' is not a valid field name."
        ));
    }
    refuse_leading_underscore(name)
}

/// dspy's own rule, and the one message here that can be reproduced word for word: the others are
/// CPython's parser talking, and this one is upstream's.
///
/// A leading underscore is a valid Python identifier, so nothing about parsing rejects it — which
/// is why assuming `_q` was fine wrote a test that agreed with the wrong answer. Upstream refuses
/// it explicitly, suggesting the name without its underscores.
fn refuse_leading_underscore(name: &str) -> Result<()> {
    if !name.starts_with('_') {
        return Ok(());
    }
    let suggestion = match name.trim_start_matches('_') {
        "" => "my_field",
        stripped => stripped,
    };
    Err(anyhow!(
        "Fields must not use names with leading underscores; \
         e.g., use '{suggestion}' instead of '{name}'."
    ))
}

/// Python's identifier rule, taken at its practical width: a letter or underscore, then letters,
/// digits and underscores. Python also admits most unicode letters, and `char::is_alphabetic`
/// follows it there rather than narrowing to ASCII.
fn is_identifier(name: &str) -> bool {
    let mut characters = name.chars();
    let leads = characters
        .next()
        .is_some_and(|first| first == '_' || first.is_alphabetic());
    leads && characters.all(|c| c == '_' || c.is_alphanumeric())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_python_identifier_is_a_field_name() {
        // `match`, `case` and `type` are soft keywords: Python allows them as identifiers and so
        // does upstream — measured, not assumed. A *trailing* underscore is fine; a leading one is
        // refused, and this list asserted the opposite until it was checked.
        for good in ["q", "q1", "q_", "question_text", "match", "case", "type"] {
            assert!(
                refuse_unless_identifier(good, "input").is_ok(),
                "rejected {good}"
            );
        }
    }

    /// A leading underscore parses as an identifier and is still refused — upstream's own rule,
    /// with upstream's own message. This test asserted `_q` was *fine* until it was measured.
    #[test]
    fn a_leading_underscore_is_refused_in_dspys_words() {
        let refusal = |name: &str| {
            refuse_unless_identifier(name, "input")
                .expect_err("refused")
                .to_string()
        };
        assert_eq!(
            refusal("_q"),
            "Fields must not use names with leading underscores; e.g., use 'q' instead of '_q'."
        );
        assert_eq!(
            refusal("__q"),
            "Fields must not use names with leading underscores; e.g., use 'q' instead of '__q'."
        );
        assert_eq!(
            refusal("_"),
            "Fields must not use names with leading underscores; \
             e.g., use 'my_field' instead of '_'."
        );
    }

    /// Each of these is a `SyntaxError` upstream, and each was a field here.
    #[test]
    fn what_python_cannot_parse_is_refused() {
        for bad in [
            "", "1q", "q-x", "q x", "class", "lambda", "\"q\"", "q.x", "q()",
        ] {
            assert!(
                refuse_unless_identifier(bad, "output").is_err(),
                "accepted {bad:?}"
            );
        }
    }
}
