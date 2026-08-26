//! dspy `dsp/utils/dpr.py`: the tokenizer a passage and an answer are compared through.
//!
//! One thing upstream reaches for it: [`answer_passage_match`], which asks whether a passage holds
//! the answer. That question cannot be put to the raw text, because `Paris.` and `paris` are the
//! same answer while `北京` is not in `北京市` at all — it is about *tokens*, and this is the
//! tokenizer that decides what one is.
//!
//! [`answer_passage_match`]: super::metrics::answer_passage_match
//!
//! Upstream is a single ordered regex, `([\p{L}\p{N}\p{M}]+)|([^\p{Z}\p{C}])`, over the NFD form.
//! Reproduced as a left-to-right walk, because that is what the alternation already is.

use unicode_general_category::get_general_category;
use unicode_normalization::UnicodeNormalization;

/// Whether a character falls in any of the Unicode groups named by their one-letter codes.
///
/// dspy writes its two character classes as groups — `\p{L}\p{N}\p{M}` and `\p{Z}\p{C}` — and a
/// general category's abbreviation opens with the letter of its group, so asking for that letter
/// asks upstream's question directly. Enumerating the thirty categories instead would answer it
/// only until Unicode adds the thirty-first.
fn is_in(character: char, groups: &str) -> bool {
    let abbreviation = get_general_category(character).abbreviation();
    groups.chars().any(|group| abbreviation.starts_with(group))
}

/// dspy `DPR_normalize`: the text as lowercased tokens.
///
/// The alternation is ordered, so a run of letters, numbers and marks is one token and anything
/// else that is neither a separator nor an "other" is a token on its own — `a-b` is three. What
/// `\p{Z}` and `\p{C}` cover falls out entirely: spaces, controls, formats like a zero-width
/// joiner, surrogates, private use, and codepoints Unicode has not assigned.
///
/// Each token is lowercased as the whole string upstream lowercases, not character by character,
/// because Greek keeps a different sigma at the end of a word: `ΟΔΥΣΣΕΥΣ` ends in `ς` and a lone
/// `Σ` does not.
pub fn normalize(text: &str) -> Vec<String> {
    let decomposed: Vec<char> = text.nfd().collect();
    let mut tokens = Vec::new();
    let mut rest: &[char] = &decomposed;
    while let Some(&first) = rest.first() {
        let length = match is_in(first, "LNM") {
            true => rest.iter().take_while(|&&c| is_in(c, "LNM")).count(),
            false => 1,
        };
        let (token, remainder) = rest.split_at(length);
        if !is_in(first, "ZC") {
            tokens.push(token.iter().collect::<String>().to_lowercase());
        }
        rest = remainder;
    }
    tokens
}

/// dspy `has_answer`: whether any of the tokenized answers appears in the text as a run of tokens.
///
/// An empty answer matches everything, which is upstream's `[] == text[i:i]` on the first index it
/// tries and the reason this cannot be a bare `windows` — Rust's panics on a zero width.
pub fn has_answer(tokenized_answers: &[Vec<String>], text: &str) -> bool {
    let tokens = normalize(text);
    tokenized_answers.iter().any(|answer| {
        answer.is_empty() || tokens.windows(answer.len()).any(|window| window == answer)
    })
}
