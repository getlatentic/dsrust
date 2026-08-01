//! The three questions the string scan asks that are not about the string itself.
//!
//! Whether a bare word is `true`, `false` or `null`; whether a backtick run opens a fenced JSON
//! block; and — the one that decides most of the disagreements — whether a comma inside an
//! unterminated object value ends the member or belongs to the text.

use crate::parser::Parser;
use crate::value::Value;
use crate::{Result, STRING_DELIMITERS, pychar};

/// What a comma inside an unterminated object value turned out to be.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum CommaMeaning {
    /// It opens a container that belongs to the string.
    Container,
    /// It ends this member; the next object member starts after it.
    Member,
    /// It belongs to the string.
    Str,
    /// It belongs to the string, and no delimiter follows anywhere — so the rest does too.
    StrNoFutureDelimiter,
}

impl Parser {
    /// A bare `true`, `false` or `null`, or the empty string and no movement.
    pub(crate) fn parse_boolean_or_null(&mut self) -> Value {
        let first = self.char_here().map(pychar::lower).unwrap_or_default();
        let (word, value) = match first.as_str() {
            "t" => ("true", Value::Bool(true)),
            "f" => ("false", Value::Bool(false)),
            "n" => ("null", Value::Null),
            _ => return Value::Str(String::new()),
        };

        let starting_index = self.index;
        let mut matched = 0;
        let mut char = first;
        for expected in word.chars() {
            if char != expected.to_string() {
                break;
            }
            matched += 1;
            self.index += 1;
            char = self.char_here().map(pychar::lower).unwrap_or_default();
        }
        if matched == word.chars().count() {
            return value;
        }
        self.index = starting_index;
        Value::Str(String::new())
    }

    /// A ```` ```json ```` block, parsed as JSON.
    ///
    /// Answers `None` both when there is no block and when the block parsed to `false`, because
    /// upstream signals "no block" with the `False` singleton and cannot tell the two apart.
    pub(crate) fn parse_json_llm_block(&mut self) -> Result<Option<Value>> {
        if self.slice(self.index as isize, self.index as isize + 7) != "```json" {
            return Ok(None);
        }
        let i = self.skip_to_character(&['`'], 7);
        if self.slice(self.index as isize + i, self.index as isize + i + 3) != "```" {
            return Ok(None);
        }
        self.index += 7;
        let value = self.parse_json(None, "$")?;
        if value == Value::Bool(false) {
            return Ok(None);
        }
        Ok(Some(value))
    }

    /// Whether the comma at the cursor ends this object member.
    ///
    /// The test walks forward looking for a key — quoted, backticked or bare — with a colon after
    /// it. A bare key only counts when its *value* also looks recoverable, since `, floof:
    /// explanation` is more often prose than a member.
    pub(crate) fn classify_object_value_comma(
        &self,
        state: &mut super::StringParseState,
    ) -> CommaMeaning {
        let next_idx = self.scroll_whitespaces(1);
        let Some(next_c) = self.get_char_at(next_idx) else {
            return CommaMeaning::Member;
        };
        if next_c == '}' {
            return CommaMeaning::Member;
        }

        if STRING_DELIMITERS.contains(&next_c) {
            let key_end_idx = self.skip_to_character(&[next_c], next_idx + 1);
            if self.get_char_at(key_end_idx).is_none() {
                return CommaMeaning::Str;
            }
            let key_end_idx = self.scroll_whitespaces(key_end_idx + 1);
            return self.member_if_colon(key_end_idx);
        }

        if next_c == '`' {
            let bare_key_idx = self.scroll_whitespaces(self.scan_bare_key(next_idx + 1));
            return self.member_if_colon(bare_key_idx);
        }

        if pychar::is_alnum(next_c) || next_c == '_' {
            let bare_key_idx = self.scroll_whitespaces(self.scan_bare_key(next_idx));
            if self.get_char_at(bare_key_idx) == Some(':')
                && self.bare_member_has_recoverable_value(bare_key_idx + 1, state)
            {
                return CommaMeaning::Member;
            }
        }

        if next_c == '{' || next_c == '[' {
            return CommaMeaning::Container;
        }

        let targets = [STRING_DELIMITERS.as_slice(), &['{', '[']].concat();
        let next_special_idx = self.cached_skip_to_character(state, &targets, next_idx);
        let Some(next_special) = self.get_char_at(next_special_idx) else {
            return CommaMeaning::StrNoFutureDelimiter;
        };
        if next_special == '{' || next_special == '[' {
            return CommaMeaning::Str;
        }
        let key_end_idx =
            self.cached_skip_to_character(state, &[next_special], next_special_idx + 1);
        if self.get_char_at(key_end_idx).is_none() {
            return CommaMeaning::Str;
        }
        let key_end_idx = self.scroll_whitespaces(key_end_idx + 1);
        self.member_if_colon(key_end_idx)
    }

    fn member_if_colon(&self, idx: isize) -> CommaMeaning {
        if self.get_char_at(idx) == Some(':') {
            CommaMeaning::Member
        } else {
            CommaMeaning::Str
        }
    }

    /// Past a run of key characters, which are alphanumeric plus `_` and `-`.
    pub(crate) fn scan_bare_key(&self, mut idx: isize) -> isize {
        while self
            .get_char_at(idx)
            .is_some_and(|char| pychar::is_alnum(char) || char == '_' || char == '-')
        {
            idx += 1;
        }
        idx
    }

    /// Whether what follows a bare key is a value worth breaking the string for.
    fn bare_member_has_recoverable_value(
        &self,
        value_idx: isize,
        state: &mut super::StringParseState,
    ) -> bool {
        let value_start_idx = self.scroll_whitespaces(value_idx);
        let value_start = self.get_char_at(value_start_idx);
        if value_start.is_some_and(|char| {
            STRING_DELIMITERS.contains(&char) || matches!(char, '{' | '[' | '-')
        }) {
            return true;
        }
        if value_start.is_some_and(pychar::is_digit) {
            return true;
        }

        for literal in ["true", "false", "null"] {
            if literal.chars().enumerate().all(|(offset, char)| {
                self.get_char_at(value_start_idx + offset as isize) == Some(char)
            }) {
                let value_end = self.get_char_at(value_start_idx + literal.len() as isize);
                if value_end
                    .is_none_or(|char| pychar::is_space(char) || matches!(char, ',' | '}' | ']'))
                {
                    return true;
                }
            }
        }

        // An unquoted value is only a safe member boundary when its object closes before the
        // current string can.
        let targets = [STRING_DELIMITERS.as_slice(), &['}']].concat();
        let value_end_idx = self.cached_skip_to_character(state, &targets, value_start_idx);
        self.get_char_at(value_end_idx) == Some('}')
    }
}

/// Tracks a container the scan is keeping inside the string. Answers whether the pending flag
/// survives, and whether this character is one to keep.
pub(crate) fn update_inline_container_stack(
    char: char,
    pending_inline_container: bool,
    inline_container_stack: &mut Vec<char>,
) -> (bool, bool) {
    if char == '{' || char == '[' {
        if pending_inline_container {
            inline_container_stack.push(char);
            return (false, false);
        }
        if !inline_container_stack.is_empty() {
            inline_container_stack.push(char);
        }
    }

    let closes = match inline_container_stack.last() {
        Some('{') => char == '}',
        Some('[') => char == ']',
        _ => false,
    };
    if closes {
        inline_container_stack.pop();
        return (pending_inline_container, true);
    }
    (pending_inline_container, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_container_opened_while_pending_clears_the_flag_and_is_not_kept() {
        let mut stack = Vec::new();
        assert_eq!(
            update_inline_container_stack('{', true, &mut stack),
            (false, false)
        );
        assert_eq!(stack, vec!['{']);
        assert_eq!(
            update_inline_container_stack('[', false, &mut stack),
            (false, false)
        );
        assert_eq!(stack, vec!['{', '[']);
        assert_eq!(
            update_inline_container_stack(']', false, &mut stack),
            (false, true)
        );
        assert_eq!(
            update_inline_container_stack('}', false, &mut stack),
            (false, true)
        );
        assert!(stack.is_empty());
    }

    #[test]
    fn a_closer_with_nothing_open_is_not_kept() {
        let mut stack = Vec::new();
        assert_eq!(
            update_inline_container_stack('}', false, &mut stack),
            (false, false)
        );
    }
}
