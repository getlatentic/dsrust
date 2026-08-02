//! `patternProperties`, matched over the literal-and-anchor subset upstream supports.
//!
//! Not a regex engine: anything past a literal with optional `^` and `$` is reported unsupported
//! and skipped, so `"^a.*z$"` matches nothing rather than matching by accident.

use crate::value::Value;

const UNSUPPORTED_REGEX_TOKENS: [char; 14] = [
    '.', '^', '$', '*', '+', '?', '{', '}', '[', ']', '|', '(', ')', '\\',
];

/// The schemas whose pattern matches `key`, and the patterns that were too complex to try.
pub(crate) fn match_pattern_properties<'a>(
    pattern_properties: &'a [(String, Value)],
    key: &str,
) -> (Vec<&'a Value>, Vec<&'a str>) {
    let mut matched = Vec::new();
    let mut unsupported = Vec::new();
    for (pattern, schema) in pattern_properties {
        let anchored_start = pattern.starts_with('^');
        let anchored_end = pattern.ends_with('$');
        // `pattern[1 if anchored_start else 0 : -1 if anchored_end else None]`. Both anchors are
        // one byte, and `anchored_end` cannot hold for an empty remainder — `"^"` does not end in
        // `$` — so neither slice needs a guard of its own.
        let literal = {
            let without_start = if anchored_start {
                &pattern[1..]
            } else {
                pattern.as_str()
            };
            match anchored_end {
                true => &without_start[..without_start.len() - 1],
                false => without_start,
            }
        };
        if literal.contains(UNSUPPORTED_REGEX_TOKENS) {
            unsupported.push(pattern.as_str());
            continue;
        }
        let is_match = match (anchored_start, anchored_end) {
            (true, true) => key == literal,
            (true, false) => key.starts_with(literal),
            (false, true) => key.ends_with(literal),
            (false, false) => key.contains(literal),
        };
        if is_match {
            matched.push(schema);
        }
    }
    (matched, unsupported)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn patterns() -> Vec<(String, Value)> {
        vec![
            ("^id".to_owned(), Value::Int(1)),
            ("_at$".to_owned(), Value::Int(2)),
            ("^exact$".to_owned(), Value::Int(3)),
            ("mid".to_owned(), Value::Int(4)),
            ("^a.*z$".to_owned(), Value::Int(5)),
        ]
    }

    #[test]
    fn each_anchoring_matches_its_own_way_and_a_real_regex_matches_nothing() {
        let all = patterns();
        assert_eq!(
            match_pattern_properties(&all, "id_mid").0,
            vec![&Value::Int(1), &Value::Int(4)]
        );
        assert_eq!(
            match_pattern_properties(&all, "created_at").0,
            vec![&Value::Int(2)]
        );
        assert_eq!(
            match_pattern_properties(&all, "exact").0,
            vec![&Value::Int(3)]
        );
        assert_eq!(
            match_pattern_properties(&all, "abcz").0,
            Vec::<&Value>::new()
        );
        assert_eq!(match_pattern_properties(&all, "abcz").1, vec!["^a.*z$"]);
    }
}
