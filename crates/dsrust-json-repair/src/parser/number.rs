//! `parse_number`: a run of number-ish characters, and what Python makes of it.
//!
//! The character set is wider than JSON's — it takes `/`, `,` and `_` — because the text it reads
//! is as likely to be `1_000`, `3/4` or `1,234` as a number. What each of those becomes is decided
//! by `int()` and `float()` failing, so the fallback is the text itself rather than an error.

use crate::parser::Parser;
use crate::parser::context::ContextValue;
use crate::pynum::{try_python_float, try_python_int};
use crate::value::Value;
use crate::{Result, pychar};

const NUMBER_CHARS: &str = "0123456789-.eE/,_";

impl Parser {
    pub(crate) fn parse_number(&mut self) -> Result<Value> {
        let mut number_str = String::new();
        let is_array = self.context.is(ContextValue::Array);
        while let Some(char) = self.char_here() {
            if !NUMBER_CHARS.contains(char) || (is_array && char == ',') {
                break;
            }
            if char != '_' {
                number_str.push(char);
            }
            self.index += 1;
        }
        if self.char_here().is_some_and(pychar::is_alpha) {
            // This was a string instead, sorry.
            self.index -= number_str.chars().count();
            return self.parse_string();
        }
        if number_str.ends_with(['-', 'e', 'E', '/', ',']) {
            // Ends on a character that is valid inside a number or a currency but not at the end
            // of one, so give it back.
            number_str.pop();
            self.index -= 1;
        }
        Ok(number_value(&number_str))
    }
}

/// `int(...)`, `float(...)`, or the text when neither conversion works.
fn number_value(number_str: &str) -> Value {
    if number_str.contains(',') {
        return Value::Str(number_str.to_owned());
    }
    if number_str.contains(['.', 'e', 'E']) {
        return match try_python_float(number_str) {
            Some(number) => Value::Float(number),
            None => Value::Str(number_str.to_owned()),
        };
    }
    try_python_int(number_str).unwrap_or_else(|| Value::Str(number_str.to_owned()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn what_the_two_conversions_refuse_comes_back_as_its_own_text() {
        assert_eq!(number_value("12"), Value::Int(12));
        assert_eq!(number_value("1.5"), Value::Float(1.5));
        assert_eq!(
            number_value("1,234"),
            Value::Str("1,234".into()),
            "a comma short-circuits"
        );
        assert_eq!(number_value("3/4"), Value::Str("3/4".into()));
        assert_eq!(number_value("1.2.3"), Value::Str("1.2.3".into()));
        assert_eq!(number_value(""), Value::Str(String::new()));
    }
}
