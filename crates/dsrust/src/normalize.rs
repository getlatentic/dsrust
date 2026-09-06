//! Text normalization shared by inference-time aggregation and native evaluation metrics.

use unicode_normalization::UnicodeNormalization;

/// dspy `normalize_text`: Unicode NFD, lowercase, drop ASCII punctuation, drop English articles,
/// then collapse whitespace.
pub fn normalize_text(text: &str) -> String {
    let folded: String = text.nfd().collect::<String>().to_lowercase();
    let unpunctuated: String = folded
        .chars()
        .filter(|character| !character.is_ascii_punctuation())
        .collect();
    collapse_whitespace(&remove_articles(&unpunctuated))
}

fn is_word(character: char) -> bool {
    character.is_alphanumeric() || character == '_'
}

fn remove_articles(text: &str) -> String {
    const ARTICLES: [&str; 3] = ["a", "an", "the"];
    let characters: Vec<char> = text.chars().collect();
    let mut normalized = String::with_capacity(text.len());
    let mut remaining: &[char] = &characters;
    let mut opens_word = true;
    while let Some((&first, tail)) = remaining.split_first() {
        let article = opens_word
            .then(|| {
                ARTICLES.into_iter().find(|article| {
                    let length = article.chars().count();
                    length <= remaining.len()
                        && remaining[..length].iter().copied().eq(article.chars())
                        && remaining
                            .get(length)
                            .copied()
                            .is_none_or(|next| !is_word(next))
                })
            })
            .flatten();
        match article {
            Some(article) => {
                normalized.push(' ');
                remaining = &remaining[article.chars().count()..];
                opens_word = true;
            }
            None => {
                normalized.push(first);
                opens_word = !is_word(first);
                remaining = tail;
            }
        }
    }
    normalized
}

fn collapse_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}
