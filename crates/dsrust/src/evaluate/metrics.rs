//! dspy `evaluate/metrics.py`: the answer-comparison metrics.
//!
//! Every one of them compares *normalised* text, so [`normalize_text`] is what they all agree on:
//! the same NFD, lowercasing, punctuation and article stripping upstream applies before a single
//! token is counted.

use std::collections::HashMap;

use serde_json::Value;
use unicode_normalization::UnicodeNormalization;

use crate::evaluate::dpr;
use crate::example::{Example, Prediction};

/// The answers dspy treats as labels rather than prose: a HotPotQA answer of `yes` never partly
/// matches `no`, however many tokens they share.
const LABELS: [&str; 3] = ["yes", "no", "noanswer"];

/// dspy `normalize_text`: Unicode NFD, lowercase, drop ASCII punctuation, drop English articles,
/// then collapse whitespace — in that order, which is the order upstream nests the steps in.
pub fn normalize_text(text: &str) -> String {
    let folded: String = text.nfd().collect::<String>().to_lowercase();
    let unpunctuated: String = folded
        .chars()
        .filter(|c| !is_ascii_punctuation(*c))
        .collect();
    collapse_whitespace(&remove_articles(&unpunctuated))
}

/// Python's `string.punctuation`, which is ASCII only — a Unicode dash or quote survives it, and
/// upstream's normalisation leaves those in place too.
fn is_ascii_punctuation(c: char) -> bool {
    c.is_ascii_punctuation()
}

/// Python's `\w`: a letter, a digit, or an underscore. What decides where a word ends, and so
/// which `a` is an article and which is part of a longer word.
fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// dspy's `re.sub(r"\b(a|an|the)\b", " ", text)`: each standalone article becomes a space.
///
/// The alternation is tried in upstream's order, so at `an` the pattern first tries `a`, finds a
/// word character after it, and falls through to `an` — which is why matching longest-first here
/// would be the wrong reading of the same regex.
fn remove_articles(text: &str) -> String {
    const ARTICLES: [&str; 3] = ["a", "an", "the"];
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    // The walk shrinks a slice rather than advancing a cursor: every arm continues from a strict
    // suffix of `rest`, so there is no index arithmetic for a mutant to stall — the shape the
    // cursor-arithmetic lint enforces, after four spins of the `index += n` form elsewhere.
    let mut rest: &[char] = &chars;
    let mut opens = true;
    while let Some((&first, tail)) = rest.split_first() {
        let article = opens
            .then(|| {
                ARTICLES.into_iter().find(|article| {
                    let length = article.chars().count();
                    length <= rest.len()
                        && rest[..length].iter().copied().eq(article.chars())
                        && rest.get(length).copied().is_none_or(|next| !is_word(next))
                })
            })
            .flatten();
        match article {
            Some(article) => {
                out.push(' ');
                rest = &rest[article.chars().count()..];
                opens = true;
            }
            None => {
                out.push(first);
                opens = !is_word(first);
                rest = tail;
            }
        }
    }
    out
}

/// Python's `" ".join(text.split())`: split on any run of whitespace, joined by single spaces.
fn collapse_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// How many tokens the two share, counting repeats — Python's `Counter(a) & Counter(b)` summed.
fn overlap(prediction: &[&str], truth: &[&str]) -> usize {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for token in prediction {
        *counts.entry(*token).or_default() += 1;
    }
    truth
        .iter()
        .filter(|token| {
            counts.get_mut(*token).is_some_and(|left| {
                *left -= 1;
                true
            })
        })
        .count()
}

/// The token F1 of two already-normalised strings, and 0 where they share nothing.
///
/// dspy prints a diagnostic when both sides are empty and scores it 0 regardless; the score is
/// what a metric is read for, so only the score is reproduced.
fn token_f1(prediction: &str, truth: &str) -> f64 {
    let prediction: Vec<&str> = prediction.split_whitespace().collect();
    let truth: Vec<&str> = truth.split_whitespace().collect();
    let same = overlap(&prediction, &truth);
    if same == 0 {
        return 0.0;
    }
    let precision = same as f64 / prediction.len() as f64;
    let recall = same as f64 / truth.len() as f64;
    2.0 * precision * recall / (precision + recall)
}

/// dspy `em_score`: the two normalise to the same text.
pub fn em_score(prediction: &str, truth: &str) -> bool {
    normalize_text(prediction) == normalize_text(truth)
}

/// dspy `f1_score`: token F1 after normalisation.
pub fn f1_score(prediction: &str, truth: &str) -> f64 {
    token_f1(&normalize_text(prediction), &normalize_text(truth))
}

/// dspy `hotpot_f1_score`: token F1, except that a `yes`/`no`/`noanswer` on either side scores
/// zero against anything but itself — a wrong label is wrong, not partly right.
pub fn hotpot_f1_score(prediction: &str, truth: &str) -> f64 {
    let prediction = normalize_text(prediction);
    let truth = normalize_text(truth);
    let labelled = LABELS.contains(&prediction.as_str()) || LABELS.contains(&truth.as_str());
    if labelled && prediction != truth {
        return 0.0;
    }
    token_f1(&prediction, &truth)
}

/// dspy `precision_score`: the share of the prediction's tokens that the reference also has.
pub fn precision_score(prediction: &str, truth: &str) -> f64 {
    let prediction: Vec<String> = normalize_text(prediction)
        .split_whitespace()
        .map(str::to_owned)
        .collect();
    let truth: Vec<String> = normalize_text(truth)
        .split_whitespace()
        .map(str::to_owned)
        .collect();
    let borrowed: Vec<&str> = prediction.iter().map(String::as_str).collect();
    let truth: Vec<&str> = truth.iter().map(String::as_str).collect();
    let same = overlap(&borrowed, &truth);
    if same == 0 {
        return 0.0;
    }
    same as f64 / borrowed.len() as f64
}

/// dspy `EM`: whether any reference answer matches exactly, after normalisation.
pub fn em(prediction: &str, answers: &[impl AsRef<str>]) -> bool {
    answers
        .iter()
        .any(|answer| em_score(prediction, answer.as_ref()))
}

/// dspy `F1`: the best token F1 across the reference answers.
pub fn f1(prediction: &str, answers: &[impl AsRef<str>]) -> f64 {
    best(
        answers
            .iter()
            .map(|answer| f1_score(prediction, answer.as_ref())),
    )
}

/// dspy `HotPotF1`: the best HotPotQA-style F1 across the reference answers.
pub fn hotpot_f1(prediction: &str, answers: &[impl AsRef<str>]) -> f64 {
    best(
        answers
            .iter()
            .map(|answer| hotpot_f1_score(prediction, answer.as_ref())),
    )
}

/// Python's `max` over the scores, and 0 for no answers at all — where upstream would raise on an
/// empty sequence, there is no score to report and nothing matched.
fn best(scores: impl Iterator<Item = f64>) -> f64 {
    scores.fold(0.0, f64::max)
}

/// The `answer` field as the list of references it stands for: one string, or several.
fn answers_of(example: &Example) -> Vec<String> {
    match example.get("answer") {
        Some(Value::String(answer)) => vec![answer.clone()],
        Some(Value::Array(answers)) => answers
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect(),
        _ => Vec::new(),
    }
}

fn answered(prediction: &Prediction) -> String {
    prediction
        .get("answer")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned()
}

/// dspy `answer_exact_match`: whether the prediction's answer matches any reference exactly.
///
/// Scored rather than boolean, because that is what a metric is: upstream returns Python's `True`,
/// which the same arithmetic reads as `1.0`.
pub fn answer_exact_match(example: &Example, prediction: &Prediction) -> f64 {
    answer_match(example, prediction, 1.0)
}

/// The `context` field as the passages it stands for.
///
/// Upstream iterates whatever it finds there, so a `context` that is one string is a passage per
/// *character* rather than a single passage — `answer="y"` against `context="xyz"` scores 1. That
/// is a caller's mistake either way, and reproducing it costs a line.
fn passages_of(prediction: &Prediction) -> Vec<String> {
    match prediction.get("context") {
        Some(Value::Array(passages)) => passages
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect(),
        Some(Value::String(passage)) => passage.chars().map(String::from).collect(),
        _ => Vec::new(),
    }
}

/// dspy `answer_passage_match`: whether any passage in the prediction's `context` holds an answer.
///
/// Both sides go through [`normalize_text`] first, as every metric here does, and the answers go
/// on through [`dpr::normalize`] — because containment is asked of tokens rather than characters,
/// so `北京市` is not in `北京市中心` at all.
///
/// An `answer` that is neither a string nor a list scores zero where upstream raises, which is the
/// reading [`answer_match`] already takes of the same field.
pub fn answer_passage_match(example: &Example, prediction: &Prediction) -> f64 {
    let answers: Vec<Vec<String>> = answers_of(example)
        .iter()
        .map(|answer| dpr::normalize(&normalize_text(answer)))
        .collect();
    let matched = passages_of(prediction)
        .iter()
        .any(|passage| dpr::has_answer(&answers, &normalize_text(passage)));
    f64::from(u8::from(matched))
}

/// The same with dspy's `frac`: below 1.0 it asks for a token F1 of at least `frac` instead of an
/// exact match, which is how upstream grades an answer that need only be close.
pub fn answer_match(example: &Example, prediction: &Prediction, frac: f64) -> f64 {
    let answers = answers_of(example);
    let answered = answered(prediction);
    let matched = match frac >= 1.0 {
        true => em(&answered, &answers),
        false => f1(&answered, &answers) >= frac,
    };
    f64::from(u8::from(matched))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::example;

    /// Every step of upstream's normalisation, and the order they compose in.
    #[test]
    fn it_normalises_the_way_dspy_does() {
        assert_eq!(normalize_text("The,  Eiffel  Tower!"), "eiffel tower");
        assert_eq!(normalize_text("A cat and an apple"), "cat and apple");
        // `and` keeps its `a`: the article is a whole word or it is not an article.
        assert_eq!(normalize_text("Paris"), "paris");
        assert_eq!(normalize_text("  "), "");
    }

    /// Punctuation is *removed*, not replaced, and only then are articles read — so `the,cat`
    /// closes up into one word and its `the` is no longer standalone. Upstream's own output;
    /// replacing punctuation with a space would score this pair differently.
    #[test]
    fn punctuation_is_removed_before_articles_are_read() {
        assert_eq!(normalize_text("the,cat"), "thecat");
        assert_eq!(normalize_text("theatre"), "theatre");
        // `_` is in Python's `string.punctuation`, so it goes with the rest.
        assert_eq!(normalize_text("a_b the_thing"), "ab thething");
    }

    #[test]
    fn exact_match_is_read_after_normalisation() {
        assert!(em_score("Paris", "paris"));
        assert!(em_score("The Eiffel Tower", "Eiffel Tower"));
        assert!(!em_score("paris", "Paris, France"));
        assert!(em("The Eiffel Tower", &["Eiffel Tower", "Louvre"]));
        assert!(!em("Berlin", &["Eiffel Tower", "Louvre"]));
    }

    /// dspy's own documented examples: `F1("Eiffel Tower is in Paris", ["Paris"])` is 1/3.
    #[test]
    fn f1_scores_the_tokens_dspy_scores() {
        assert!((f1("Eiffel Tower is in Paris", &["Paris"]) - 1.0 / 3.0).abs() < 1e-12);
        assert_eq!(f1_score("the Eiffel Tower", "Eiffel Tower"), 1.0);
        assert_eq!(f1_score("Paris", "Berlin"), 0.0);
        // Both sides empty: no overlap, so no score — upstream's rare edge case.
        assert_eq!(f1_score("", ""), 0.0);
    }

    /// A label answer scores zero against a different one, however the tokens fall.
    #[test]
    fn hotpot_refuses_a_label_that_does_not_match() {
        assert_eq!(hotpot_f1("yes", &["no"]), 0.0);
        assert_eq!(hotpot_f1_score("yes", "yes"), 1.0);
        assert_eq!(hotpot_f1_score("noanswer", "the answer is unknown"), 0.0);
        // Neither side is a label, so it is ordinary token F1.
        assert_eq!(hotpot_f1_score("the Eiffel Tower", "Eiffel Tower"), 1.0);
    }

    /// Two of the prediction's *four* tokens are in the reference. dspy's docstring says 0.67 for
    /// this pair, but its code counts `in` as a token like any other and returns 0.5 — which the
    /// golden records, since what upstream does is the contract and what it documents is not.
    #[test]
    fn precision_is_the_share_of_predicted_tokens_that_land() {
        assert_eq!(
            precision_score("eiffel tower in paris", "eiffel tower"),
            0.5
        );
        assert_eq!(precision_score("berlin", "paris"), 0.0);
    }

    /// Repeats are counted as a multiset, so one `tower` in the reference matches one of two.
    #[test]
    fn repeated_tokens_count_once_each() {
        assert!((precision_score("tower tower", "tower") - 0.5).abs() < 1e-12);
    }

    /// The metric shape: a string answer, a list of them, and the `frac` threshold.
    #[test]
    fn answer_exact_match_reads_either_answer_shape() {
        let prediction = Prediction::new(example! { answer: "The Eiffel Tower" }, "");
        assert_eq!(
            answer_exact_match(&example! { answer: "Eiffel Tower" }, &prediction),
            1.0
        );
        let listed = Example::new([("answer", serde_json::json!(["Eiffel Tower", "Louvre"]))]);
        assert_eq!(answer_exact_match(&listed, &prediction), 1.0);
        assert_eq!(
            answer_exact_match(&example! { answer: "Louvre" }, &prediction),
            0.0
        );
        // Below 1.0 the threshold is on token F1 instead, so a partial answer passes.
        let partial = Prediction::new(example! { answer: "Eiffel Tower is in Paris" }, "");
        assert_eq!(
            answer_match(&example! { answer: "Paris" }, &partial, 0.3),
            1.0
        );
        assert_eq!(
            answer_match(&example! { answer: "Paris" }, &partial, 0.5),
            0.0
        );
    }
}
