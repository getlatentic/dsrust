//! `parse_string`: where a string starts, where it ends, and what to do when neither is marked.
//!
//! Every disagreement this crate had with dspy before the port came from here. A model writes
//! `{answer: "{}", "unknown: "7", reasoning": "[]"` and the quotes no longer pair up, so the end of
//! a string has to be *decided* rather than read: by what follows the candidate quote, by which
//! container the cursor is in, by whether a colon appears before the next comma. The rules have no
//! specification other than this code, which is why the port is line-for-line and the fixtures are
//! generated rather than written.

pub(crate) mod escape;
pub(crate) mod helpers;
pub(crate) mod lookahead;
pub(crate) mod scan;

use crate::parser::Parser;
use crate::parser::context::ContextValue;
use crate::value::Value;
use crate::{Result, STRING_DELIMITERS, pychar};

/// The sentinel pushed onto the delimiter stack while inside a `„ … ”` span, which is a quote pair
/// no other rule knows about.
pub(crate) const LOW_SMART_QUOTE_SENTINEL: char = '\0';

pub(crate) const INLINE_CONTAINER_OPENERS: [char; 3] = ['[', '{', '('];

pub(crate) fn inline_container_closer(opener: char) -> Option<char> {
    match opener {
        '[' => Some(']'),
        '{' => Some('}'),
        '(' => Some(')'),
        _ => None,
    }
}

/// A `“` closes with a `”`; every other delimiter closes with itself.
pub(crate) fn matching_string_delimiter(delimiter: char) -> char {
    if delimiter == '“' { '”' } else { delimiter }
}

#[derive(Default)]
pub(crate) struct StringParseState {
    pub(crate) missing_quotes: bool,
    pub(crate) doubled_quotes: bool,
    pub(crate) lstring_delimiter: char,
    /// A stack: the delimiter this string opened with, plus a sentinel per open `„` span.
    pub(crate) rstring_delimiter: Vec<char>,
    pub(crate) string_acc: Vec<char>,
    pub(crate) unmatched_delimiter: bool,
    pub(crate) pending_inline_container: bool,
    pub(crate) inline_container_stack: Vec<char>,
    pub(crate) object_value_has_no_future_delimiter: bool,
    pub(crate) lookahead_cache: Vec<(Vec<char>, isize, Option<isize>)>,
    pub(crate) object_value_unmatched_opening_braces: usize,
    pub(crate) regex_character_class_start: Option<usize>,
}

impl StringParseState {
    fn new() -> Self {
        Self {
            lstring_delimiter: '"',
            rstring_delimiter: vec!['"'],
            ..Default::default()
        }
    }

    pub(crate) fn outer_rstring_delimiter(&self) -> char {
        self.rstring_delimiter[0]
    }

    pub(crate) fn active_rstring_delimiter(&self) -> char {
        *self
            .rstring_delimiter
            .last()
            .expect("the stack always holds the opening delimiter")
    }

    pub(crate) fn in_low_smart_quote_span(&self) -> bool {
        self.active_rstring_delimiter() == LOW_SMART_QUOTE_SENTINEL
    }

    pub(crate) fn push_low_smart_quote_span(&mut self) {
        self.rstring_delimiter.push(LOW_SMART_QUOTE_SENTINEL);
    }

    pub(crate) fn pop_low_smart_quote_span(&mut self) {
        self.rstring_delimiter.pop();
    }

    pub(crate) fn last_acc(&self) -> Option<char> {
        self.string_acc.last().copied()
    }

    /// Appends, tracking the two things the scan reads back out of the accumulator: how many `{`
    /// are still open, and where the last `[` was.
    pub(crate) fn append(&mut self, content: &[char]) {
        let start_index = self.string_acc.len();
        self.string_acc.extend_from_slice(content);
        for (offset, char) in content.iter().enumerate() {
            match char {
                '{' => self.object_value_unmatched_opening_braces += 1,
                '}' if self.object_value_unmatched_opening_braces > 0 => {
                    self.object_value_unmatched_opening_braces -= 1;
                }
                '[' => self.regex_character_class_start = Some(start_index + offset + 1),
                ']' => self.regex_character_class_start = None,
                _ => {}
            }
        }
    }

    /// Recounts from scratch, which is what the escape handling needs after rewriting the tail.
    pub(crate) fn rebuild_unmatched_opening_braces(&mut self) {
        self.object_value_unmatched_opening_braces = 0;
        self.regex_character_class_start = None;
        for (index, char) in self.string_acc.iter().enumerate() {
            match char {
                '{' => self.object_value_unmatched_opening_braces += 1,
                '}' if self.object_value_unmatched_opening_braces > 0 => {
                    self.object_value_unmatched_opening_braces -= 1;
                }
                '[' => self.regex_character_class_start = Some(index + 1),
                ']' => self.regex_character_class_start = None,
                _ => {}
            }
        }
    }

    /// Drops the last character, which several escape rules do before appending a replacement.
    pub(crate) fn pop_acc(&mut self) {
        self.string_acc.pop();
    }
}

/// What `_prepare_string_entry` decided: a value already, or a state to scan with.
enum Entry {
    Done(Value),
    Scan(StringParseState),
}

impl Parser {
    pub(crate) fn parse_string(&mut self) -> Result<Value> {
        let mut state = match self.prepare_string_entry()? {
            Entry::Done(value) => return Ok(value),
            Entry::Scan(state) => state,
        };
        let char = self.scan_string_body(&mut state)?;
        Ok(Value::Str(self.finalize_string_result(&mut state, char)))
    }

    /// Where the string starts: which delimiter opened it, whether one did at all, and the three
    /// shapes that answer before any scanning happens.
    fn prepare_string_entry(&mut self) -> Result<Entry> {
        let mut char = self.char_here();
        if matches!(char, Some('#' | '/')) {
            return Ok(Entry::Done(self.parse_comment()?));
        }
        while let Some(current) = char {
            if STRING_DELIMITERS.contains(&current) || pychar::is_alnum(current) {
                break;
            }
            self.index += 1;
            char = self.char_here();
        }
        let Some(char) = char else {
            return Ok(Entry::Done(Value::Str(String::new())));
        };
        if let Some(value) = self.try_parse_simple_quoted_string() {
            return Ok(Entry::Done(Value::Str(value)));
        }

        let mut state = StringParseState::new();
        match char {
            '\'' => {
                state.lstring_delimiter = '\'';
                state.rstring_delimiter = vec!['\''];
            }
            '“' => {
                state.lstring_delimiter = '“';
                state.rstring_delimiter = vec!['”'];
            }
            char if pychar::is_alnum(char) => {
                if matches!(pychar::lower(char).as_str(), "t" | "f" | "n")
                    && !self.context.is(ContextValue::ObjectKey)
                {
                    let value = self.parse_boolean_or_null();
                    if !value.is_empty_string() {
                        return Ok(Entry::Done(value));
                    }
                }
                self.log("While parsing a string, we found a literal instead of a quote");
                state.missing_quotes = true;
            }
            _ => {}
        }

        if !state.missing_quotes {
            self.index += 1;
        }
        if self.char_here() == Some('`') {
            match self.parse_json_llm_block()? {
                Some(value) => return Ok(Entry::Done(value)),
                None => self.log(
                    "While parsing a string, we found code fences but they did not enclose valid JSON, continuing parsing the string",
                ),
            }
        }

        if self.char_here() == Some(state.lstring_delimiter)
            && let Some(value) = self.handle_doubled_opening_quote(&mut state)?
        {
            return Ok(Entry::Done(value));
        }
        Ok(Entry::Scan(state))
    }

    /// Two opening quotes: an empty value, a mistake to drop, or a genuinely doubled quote.
    fn handle_doubled_opening_quote(
        &mut self,
        state: &mut StringParseState,
    ) -> Result<Option<Value>> {
        let next = self.get_char_at(1);
        let empty_here = match self.context.current {
            Some(ContextValue::ObjectKey) => next == Some(':'),
            Some(ContextValue::ObjectValue) => matches!(next, Some(',' | '}')),
            Some(ContextValue::Array) => matches!(next, Some(',' | ']')),
            None => false,
        };
        if empty_here {
            self.index += 1;
            return Ok(Some(Value::Str(String::new())));
        }
        if next == Some(state.lstring_delimiter) {
            self.log("While parsing a string, we found a doubled quote and then a quote again, ignoring it");
            if self.strict {
                return Err(crate::Error::new(
                    "Found doubled quotes followed by another quote.",
                ));
            }
            return Ok(Some(Value::Str(String::new())));
        }

        let i = self.skip_to_character(&[state.outer_rstring_delimiter()], 1);
        if self.get_char_at(i + 1) == Some(state.outer_rstring_delimiter()) {
            self.log("While parsing a string, we found a valid starting doubled quote");
            state.doubled_quotes = true;
            self.index += 1;
            return Ok(None);
        }

        let i = self.scroll_whitespaces(1);
        let next_c = self.get_char_at(i);
        if next_c
            .is_some_and(|char| STRING_DELIMITERS.contains(&char) || char == '{' || char == '[')
        {
            self.log(
                "While parsing a string, we found a doubled quote but also another quote afterwards, ignoring it",
            );
            if self.strict {
                return Err(crate::Error::new(
                    "Found doubled quotes followed by another quote while parsing a string.",
                ));
            }
            self.index += 1;
            return Ok(Some(Value::Str(String::new())));
        }
        if !matches!(next_c, Some(',' | ']' | '}')) {
            self.log("While parsing a string, we found a doubled quote but it was a mistake, removing one quote");
            self.index += 1;
        }
        Ok(None)
    }

    /// The fast path: a plain `"..."` with no escapes and nothing surprising after it.
    fn try_parse_simple_quoted_string(&mut self) -> Option<String> {
        if self.char_here() != Some('"') {
            return None;
        }
        let start = self.index + 1;
        let end = self.json_str[start..]
            .iter()
            .position(|&char| char == '"')?
            + start;
        let value: String = self.json_str[start..end].iter().collect();
        if value.contains(['\\', '\n', '\r']) {
            return None;
        }

        let mut next_index = end + 1;
        while self
            .json_str
            .get(next_index)
            .is_some_and(|&char| pychar::is_space(char))
        {
            next_index += 1;
        }
        let next_char = self.json_str.get(next_index).copied();
        let follows = match self.context.current {
            Some(ContextValue::ObjectKey) => next_char == Some(':'),
            Some(ContextValue::ObjectValue) => matches!(next_char, Some(',' | '}') | None),
            Some(ContextValue::Array) => matches!(next_char, Some(',' | ']') | None),
            None => next_char.is_none(),
        };
        if !follows {
            return None;
        }
        self.index = end + 1;
        Some(value)
    }

    /// Where the string stops, and what trailing whitespace survives it.
    fn finalize_string_result(
        &mut self,
        state: &mut StringParseState,
        char: Option<char>,
    ) -> String {
        let outer = state.outer_rstring_delimiter();
        if char.is_some_and(pychar::is_space)
            && state.missing_quotes
            && self.context.is(ContextValue::ObjectKey)
        {
            self.log(
                "While parsing a string, handling an extreme corner case in which the LLM added a comment instead of valid string, invalidate the string and return an empty value",
            );
            self.skip_whitespaces();
            if !matches!(self.char_here(), Some(':' | ',')) {
                return String::new();
            }
        }

        if char != Some(outer) {
            if !self.stream_stable {
                self.log("While parsing a string, we missed the closing quote, ignoring");
                rstrip_acc(state);
            }
        } else {
            self.index += 1;
        }

        if !self.stream_stable && (state.missing_quotes || state.last_acc() == Some('\n')) {
            rstrip_acc(state);
        }
        state.string_acc.iter().collect()
    }
}

fn rstrip_acc(state: &mut StringParseState) {
    while state.last_acc().is_some_and(pychar::is_space) {
        state.string_acc.pop();
    }
}
