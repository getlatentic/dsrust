//! `parse_comment`: `#`, `//` and `/* */`, skipped over rather than parsed.
//!
//! Where a comment ends depends on where it was found: inside an array a `]` ends it, inside an
//! object value a `}` does, inside a key a `:` does. A model that annotates its JSON tends to run
//! the comment straight into the structure, and stopping at the structure is what saves the rest.

use crate::Result;
use crate::parser::Parser;
use crate::parser::context::ContextValue;
use crate::value::Value;

impl Parser {
    pub(crate) fn parse_comment(&mut self) -> Result<Value> {
        loop {
            match self.char_here() {
                Some('#') => self.skip_hash_comment(),
                Some('/') => self.skip_slash_comment(),
                _ => {}
            }
            if !self.context.empty {
                return Ok(Value::Str(String::new()));
            }
            // A long run of top-level comments would otherwise chain
            // parse_json -> parse_comment -> parse_json once per comment.
            self.skip_whitespaces();
            if matches!(self.char_here(), Some('#' | '/')) {
                continue;
            }
            return self.parse_json(None, "$");
        }
    }

    /// Where a line comment stops, which the surrounding context widens.
    fn terminates_comment(&self, char: char) -> bool {
        match char {
            '\n' | '\r' => true,
            ']' => self.context.contains(ContextValue::Array),
            '}' => self.context.contains(ContextValue::ObjectValue),
            ':' => self.context.contains(ContextValue::ObjectKey),
            _ => false,
        }
    }

    fn skip_hash_comment(&mut self) {
        let mut comment = String::new();
        while let Some(char) = self.char_here() {
            if self.terminates_comment(char) {
                break;
            }
            comment.push(char);
            self.index += 1;
        }
        self.log(&format!("Found line comment: {comment}, ignoring"));
    }

    /// `//` to the end of the line, `/* */` to its closer, and a lone `/` stepped over so the
    /// caller does not spin on it.
    fn skip_slash_comment(&mut self) {
        match self.get_char_at(1) {
            Some('/') => {
                let mut comment = String::from("//");
                self.index += 2;
                while let Some(char) = self.char_here() {
                    if char == '\n' || char == '\r' {
                        break;
                    }
                    comment.push(char);
                    self.index += 1;
                }
                self.log(&format!("Found line comment: {comment}, ignoring"));
            }
            Some('*') => {
                let mut comment = String::from("/*");
                self.index += 2;
                loop {
                    let Some(char) = self.char_here() else {
                        self.log("Reached end-of-string while parsing block comment; unclosed block comment.");
                        break;
                    };
                    comment.push(char);
                    self.index += 1;
                    if comment.ends_with("*/") {
                        break;
                    }
                }
                self.log(&format!("Found block comment: {comment}, ignoring"));
            }
            _ => self.index += 1,
        }
    }
}
