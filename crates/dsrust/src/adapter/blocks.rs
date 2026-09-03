//! dspy `split_message_content_for_custom_types`: turning a rendered message into the content
//! blocks a multimodal field needs.
//!
//! A custom type like `dspy.Image` cannot reach a provider inside a string, but the adapters
//! render every field into one. Upstream resolves that by rendering the type's blocks as JSON
//! between two sentinels, assembling the message as usual, then splitting it apart again — so
//! the prose either side of the image keeps whatever spacing the ordinary rendering gave it.
//! Splitting afterwards is the reason a text block reads `"[[ ## image ## ]]\n"`, trailing
//! newline and all, rather than something a separate assembly path would have to reproduce.

use serde_json::Value;

use crate::lm::api::{LmPart, blocks_content, part_of_block};
use crate::lm::{ChatTurn, Role};

/// The sentinels dspy wraps a custom type's blocks in. They are reserved: a value containing
/// one would be split at, which upstream accepts for the same reason.
const START: &str = "<<CUSTOM-TYPE-START-IDENTIFIER>>";
const END: &str = "<<CUSTOM-TYPE-END-IDENTIFIER>>";

/// Split every user turn around the custom-type blocks embedded in it.
///
/// Only user turns: upstream splits those alone, because a custom type reaches a prompt as an
/// input and an assistant turn carries the model's own words back. A turn with no sentinel in
/// it is left as prose rather than wrapped in a single block, which keeps every text-only
/// message on the shape it already had.
pub(super) fn split_custom_types(turns: Vec<ChatTurn>) -> Vec<ChatTurn> {
    turns
        .into_iter()
        .map(|turn| match (turn.role, turn.content.text()) {
            (Role::User, Some(text)) => match split(text) {
                Some(parts) => ChatTurn {
                    role: turn.role,
                    content: blocks_content(&parts).unwrap_or_else(|_| turn.content.clone()),
                },
                None => turn,
            },
            _ => turn,
        })
        .collect()
}

/// The parts a message splits into, or `None` when it carries no custom type at all.
fn split(content: &str) -> Option<Vec<LmPart>> {
    let mut parts = Vec::new();
    let mut rest = content;
    let mut found = false;

    while let Some((before, embedded, after)) = next_embedded(rest) {
        found = true;
        if !before.is_empty() {
            parts.push(LmPart::text(before));
        }
        parts.extend(embedded_parts(embedded));
        rest = after;
    }

    if !found {
        return None;
    }
    if !rest.is_empty() {
        parts.push(LmPart::text(rest));
    }
    Some(parts)
}

/// The text before the next sentinel pair, what it wraps, and what follows it.
fn next_embedded(content: &str) -> Option<(&str, &str, &str)> {
    let start = content.find(START)?;
    let after_start = start + START.len();
    let end = content[after_start..].find(END)? + after_start;
    Some((
        &content[..start],
        &content[after_start..end],
        &content[end + END.len()..],
    ))
}

/// What one sentinel pair contributes.
///
/// A custom type writes a JSON array of blocks, and each element becomes a block of its own.
/// Anything else — text that never parsed, or an array with nothing in it — reaches the model
/// as the text it already was, which is upstream's fallback and keeps a malformed value
/// visible instead of dropping it.
fn embedded_parts(embedded: &str) -> Vec<LmPart> {
    let trimmed = embedded.trim();
    match parse_blocks(trimmed) {
        Some(blocks) if !blocks.is_empty() => blocks.iter().map(part_of_block).collect(),
        _ => vec![LmPart::text(trimmed)],
    }
}

/// The array a custom type wrote, read straight or through one layer of escaping.
///
/// A type nested in a list or dict is JSON-encoded twice on the way out, so its array arrives
/// with the quotes escaped and no quotes around the whole. Upstream reads that by wrapping the
/// text in quotes to make it a JSON string, decoding it to undo the escaping, then parsing what
/// that yields.
fn parse_blocks(trimmed: &str) -> Option<Vec<Value>> {
    if let Ok(blocks) = serde_json::from_str::<Vec<Value>>(trimmed) {
        return Some(blocks);
    }
    let unescaped: String = serde_json::from_str(&format!("\"{trimmed}\"")).ok()?;
    serde_json::from_str(&unescaped).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lm::Content;
    use serde_json::json;

    fn wrap(inner: &str) -> String {
        format!("{START}{inner}{END}")
    }

    fn image() -> String {
        wrap(r#"[{"type": "image_url", "image_url": {"url": "https://example.com/a.jpg"}}]"#)
    }

    fn user(content: &str) -> Vec<ChatTurn> {
        vec![ChatTurn::user(content)]
    }

    #[test]
    fn a_message_with_no_custom_type_stays_prose() {
        // Wrapping every text-only message in a single block would change the bytes every
        // provider receives for the whole crate's ordinary path.
        let turns = split_custom_types(user("[[ ## question ## ]]\nWhy?"));
        assert_eq!(turns[0].content.text(), Some("[[ ## question ## ]]\nWhy?"));
    }

    #[test]
    fn the_prose_either_side_becomes_its_own_block_unaltered() {
        let content = format!("[[ ## image ## ]]\n{}\n\nRespond with", image());
        let turns = split_custom_types(user(&content));
        let Content::Blocks(blocks) = &turns[0].content else {
            panic!("got: {:?}", turns[0].content)
        };

        // The whitespace is whatever the ordinary rendering produced: the leading block keeps
        // its newline and the trailing one opens with the blank line before the reminder.
        assert_eq!(blocks.len(), 3);
        assert_eq!(
            blocks[0],
            json!({ "type": "text", "text": "[[ ## image ## ]]\n" })
        );
        assert_eq!(
            blocks[1],
            json!({ "type": "image_url", "image_url": { "url": "https://example.com/a.jpg" } })
        );
        assert_eq!(
            blocks[2],
            json!({ "type": "text", "text": "\n\nRespond with" })
        );
    }

    #[test]
    fn an_embedded_array_contributes_each_of_its_blocks() {
        let content = wrap(r#"[{"type": "text", "text": "one"}, {"type": "text", "text": "two"}]"#);
        let turns = split_custom_types(user(&content));
        let Content::Blocks(blocks) = &turns[0].content else {
            panic!("expected blocks")
        };
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[1], json!({ "type": "text", "text": "two" }));
    }

    #[test]
    fn two_custom_types_in_one_message_each_split() {
        let content = format!("a{}b{}c", image(), image());
        let turns = split_custom_types(user(&content));
        let Content::Blocks(blocks) = &turns[0].content else {
            panic!("expected blocks")
        };
        // text, image, text, image, text
        assert_eq!(blocks.len(), 5);
        assert_eq!(blocks[0], json!({ "type": "text", "text": "a" }));
        assert_eq!(blocks[2], json!({ "type": "text", "text": "b" }));
        assert_eq!(blocks[4], json!({ "type": "text", "text": "c" }));
    }

    #[test]
    fn an_escaped_array_is_read_through_its_extra_layer() {
        // A custom type nested in a list or dict is JSON-encoded twice, so its array reaches
        // here with the quotes escaped and none around the whole — the shape upstream's
        // `_parse_doubly_quoted_json` exists for.
        let content = wrap(r#"[{\"type\": \"text\", \"text\": \"nested\"}]"#);
        let turns = split_custom_types(user(&content));
        let Content::Blocks(blocks) = &turns[0].content else {
            panic!("expected blocks")
        };
        assert_eq!(blocks[0], json!({ "type": "text", "text": "nested" }));
    }

    #[test]
    fn a_value_that_never_parsed_reaches_the_model_as_its_text() {
        // Dropping it would leave the model reading a request with a field silently missing.
        let turns = split_custom_types(user(&wrap("not json at all")));
        let Content::Blocks(blocks) = &turns[0].content else {
            panic!("expected blocks")
        };
        assert_eq!(
            blocks[0],
            json!({ "type": "text", "text": "not json at all" })
        );
    }

    /// Measured against dspy 3.2.1: a marker-split message renders as a list however few blocks
    /// it holds. 3.3 collapses a lone text part to a bare string, and borrowing that rule here
    /// silently changed what every such message sent.
    #[test]
    fn a_message_that_splits_to_one_text_block_stays_a_list() {
        let turns = split_custom_types(user(&wrap(r#"[{"type": "text", "text": "only"}]"#)));
        let Content::Blocks(blocks) = &turns[0].content else {
            panic!("a split message is blocks, got {:?}", turns[0].content)
        };
        assert_eq!(blocks, &[json!({ "type": "text", "text": "only" })]);
    }

    #[test]
    fn an_assistant_turn_is_left_alone() {
        // A custom type reaches a prompt as an input; an assistant turn carries the model's
        // own words, which upstream never splits.
        let turns = split_custom_types(vec![ChatTurn::assistant(image())]);
        assert!(turns[0].content.text().is_some());
    }

    #[test]
    fn an_unterminated_sentinel_leaves_the_message_alone() {
        // Upstream's pattern needs both sentinels to match, so a half-written one is prose.
        // Splitting at it would drop the opening marker and everything after it.
        let content = format!("before{START}oops");
        let turns = split_custom_types(user(&content));
        assert_eq!(turns[0].content.text(), Some(content.as_str()));
    }
}
