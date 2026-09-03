//! Where the reader is in the text, and the smallest things it reads: one character at a time with
//! expat's line and column bookkeeping, a literal it must find, and a name.
//!
//! Separated from the grammar because the two answer different questions — this one is "what is the
//! next character and where am I", `tokenizer.rs` is "what does this text mean" — and because a
//! position rule wrong here is wrong in every refusal at once.
use super::names::{NAME_CHAR, NAME_START, contains};
use super::namespaces::QName;
use super::tokenizer::Parser;
use super::tree::{INVALID, ParseError, Position, UNCLOSED};

/// XML's own whitespace: the four characters that may separate a tag's parts.
pub(super) fn is_whitespace(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\r' | '\n')
}

impl Parser {
    pub(super) fn peek(&self) -> Option<char> {
        self.chars.get(self.at).copied()
    }

    pub(super) fn starts_with(&self, text: &str) -> bool {
        text.chars()
            .enumerate()
            .all(|(i, c)| self.chars.get(self.at + i) == Some(&c))
    }

    /// The next character, a line end read as `\n`: expat normalises `\r\n` and a lone `\r`, and
    /// counts each as one line.
    pub(super) fn bump(&mut self) -> Option<char> {
        let c = *self.chars.get(self.at)?;
        self.at += 1;
        if c == '\r' || c == '\n' {
            if c == '\r' && self.chars.get(self.at) == Some(&'\n') {
                self.at += 1;
            }
            self.position.line += 1;
            self.position.column = 0;
            return Some('\n');
        }
        self.position.column += 1;
        Some(c)
    }

    pub(super) fn here(&self, kind: &'static str) -> ParseError {
        ParseError {
            kind,
            at: self.position,
        }
    }

    /// The text ahead spells `expected`, or the first character that does not is refused — and
    /// the text running out first is the token at `at` left unclosed.
    pub(super) fn literal(&mut self, expected: &str, at: Position) -> Result<(), ParseError> {
        for c in expected.chars() {
            match self.peek() {
                Some(found) if found == c => {
                    self.bump();
                }
                Some(_) => return Err(self.here(INVALID)),
                None => return Err(ParseError { kind: UNCLOSED, at }),
            }
        }
        Ok(())
    }

    pub(super) fn skip_whitespace(&mut self) -> bool {
        let mut any = false;
        while self.peek().is_some_and(is_whitespace) {
            self.bump();
            any = true;
        }
        any
    }

    pub(super) fn ncname(&mut self, token: Position) -> Result<String, ParseError> {
        let mut name = String::new();
        match self.peek() {
            Some(c) if contains(NAME_START, c) => {
                name.push(c);
                self.bump();
            }
            None => {
                return Err(ParseError {
                    kind: UNCLOSED,
                    at: token,
                });
            }
            Some(_) => return Err(self.here(INVALID)),
        }
        while let Some(c) = self.peek().filter(|c| contains(NAME_CHAR, *c)) {
            name.push(c);
            self.bump();
        }
        Ok(name)
    }

    /// `NCName` or `prefix:NCName` — with namespace processing on, a colon may appear once, and
    /// each side of it is a name of its own. `token` is where the tag began, which is where the
    /// text running out is reported.
    pub(super) fn qualified_name(&mut self, token: Position) -> Result<QName, ParseError> {
        let first = self.ncname(token)?;
        if self.peek() != Some(':') {
            return Ok(QName {
                prefix: None,
                local: first,
            });
        }
        self.bump();
        let local = self.ncname(token)?;
        Ok(QName {
            prefix: Some(first),
            local,
        })
    }
}
