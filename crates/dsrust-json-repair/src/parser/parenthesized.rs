//! `(...)`: a Python tuple literal, or prose that happens to open a bracket.
//!
//! A model asked for JSON sometimes answers in Python, so `(1, 2)` is a value. But a reply also
//! reads `the answer (see below): {...}`, and treating that bracket as a value swallows the JSON
//! after it. The two are told apart by what the bracket holds and what surrounds it, and the test
//! is deliberately conservative at the top level and permissive once inside a container.

use crate::parser::Parser;
use crate::value::Value;
use crate::{Result, STRING_DELIMITERS, pychar};

/// Bracket depth while scanning a parenthesised run. Signed, because one of the two scans
/// decrements past zero and reads the negative count as "not at the top level".
#[derive(Default)]
struct Nesting {
    parentheses: i64,
    square_brackets: i64,
    braces: i64,
    quote: Option<char>,
    backslashes: usize,
}

/// What a character did to the scan before any bracket counting.
enum Quoting {
    /// A backslash, or a character inside a string: nothing else looks at it.
    Consumed,
    /// A quote that opened a string.
    Opened,
    /// An ordinary character.
    Bare,
}

impl Nesting {
    fn at_top_level(&self) -> bool {
        self.parentheses == 0 && self.square_brackets == 0 && self.braces == 0
    }

    fn quoting(&mut self, ch: char) -> Quoting {
        if ch == '\\' {
            self.backslashes += 1;
            return Quoting::Consumed;
        }
        if let Some(quote) = self.quote {
            if ch == quote && self.backslashes.is_multiple_of(2) {
                self.quote = None;
            }
            self.backslashes = 0;
            return Quoting::Consumed;
        }
        if STRING_DELIMITERS.contains(&ch) && self.backslashes.is_multiple_of(2) {
            self.quote = Some(ch);
            self.backslashes = 0;
            return Quoting::Opened;
        }
        self.backslashes = 0;
        Quoting::Bare
    }

    fn open_or_close(&mut self, ch: char) {
        match ch {
            '(' => self.parentheses += 1,
            '[' => self.square_brackets += 1,
            ']' if self.square_brackets > 0 => self.square_brackets -= 1,
            '{' => self.braces += 1,
            '}' if self.braces > 0 => self.braces -= 1,
            _ => {}
        }
    }
}

impl Parser {
    /// `(1, 2)` is a tuple and `(1)` is a grouped value, so one keeps its brackets and one does not.
    pub(crate) fn parse_parenthesized(
        &mut self,
        schema: Option<crate::Schema>,
        path: &str,
    ) -> Result<Value> {
        let explicit_tuple = self.parenthesized_is_explicit_tuple();
        self.index += 1;
        let mut items = self.parse_array_items(schema, path, ')')?;
        if explicit_tuple || items.len() != 1 {
            return Ok(Value::Array(items));
        }
        Ok(items.pop().expect("just checked the length"))
    }

    /// Whether this `(` opens a tuple: empty brackets count, a comma at the top level counts, a
    /// single grouped value does not.
    pub(crate) fn parenthesized_is_explicit_tuple(&self) -> bool {
        let mut nesting = Nesting::default();
        let mut saw_top_level_content = false;
        for position in self.index + 1..self.len() {
            let Some(ch) = self.json_str.at(position) else {
                break;
            };
            match nesting.quoting(ch) {
                Quoting::Consumed => continue,
                Quoting::Opened => {
                    saw_top_level_content = saw_top_level_content || nesting.at_top_level();
                    continue;
                }
                Quoting::Bare => {}
            }
            if !pychar::is_space(ch) && ch != ',' && ch != ')' && nesting.at_top_level() {
                saw_top_level_content = true;
            }
            if ch == ')' {
                if nesting.at_top_level() {
                    return !saw_top_level_content;
                }
                if nesting.parentheses > 0 {
                    nesting.parentheses -= 1;
                }
            } else if ch == ',' && nesting.at_top_level() {
                return true;
            } else {
                nesting.open_or_close(ch);
            }
        }
        !saw_top_level_content
    }

    /// Whether a `(` with nothing but whitespace before it on its line opens a value rather than
    /// an aside, judged by what follows it and by its closing bracket ending the line.
    pub(crate) fn top_level_parenthesized_can_start_value(&self) -> bool {
        let mut position = self.index as isize - 1;
        while position >= 0 {
            let Some(ch) = self.json_str.at(position as usize) else {
                break;
            };
            if ch == '\n' || ch == '\r' {
                break;
            }
            if !pychar::is_space(ch) {
                return false;
            }
            position -= 1;
        }

        if !self.parenthesized_opens_with_a_value() {
            return false;
        }

        let mut nesting = Nesting::default();
        for position in self.index + 1..self.len() {
            let Some(ch) = self.json_str.at(position) else {
                break;
            };
            if matches!(nesting.quoting(ch), Quoting::Consumed | Quoting::Opened) {
                continue;
            }
            if ch == ')' {
                if !nesting.at_top_level() {
                    nesting.parentheses -= 1;
                    continue;
                }
                return self
                    .json_str
                    .iter_from(position + 1)
                    .take_while(|&trailer| trailer != '\n' && trailer != '\r')
                    .all(pychar::is_space);
            }
            nesting.open_or_close(ch);
        }
        true
    }

    /// Whether the first thing inside the bracket could begin a JSON value at all.
    fn parenthesized_opens_with_a_value(&self) -> bool {
        let idx = self.scroll_whitespaces(1);
        let Some(first) = self.get_char_at(idx) else {
            return false;
        };
        let from = self.index as isize + idx;
        matches!(first, ')' | '{' | '[' | '(' | '-' | '.')
            || STRING_DELIMITERS.contains(&first)
            || pychar::is_digit(first)
            || self.slice(from, from + 4) == "true"
            || self.slice(from, from + 4) == "null"
            || self.slice(from, from + 5) == "false"
    }
}
