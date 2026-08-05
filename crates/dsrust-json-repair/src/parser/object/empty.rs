//! An object that consumed characters and produced no members, read again as something else.
//!
//! `{"a", "b"}` is a set, `{\"a\": 1}` is an object whose quotes were escaped by whatever wrote
//! it, and `{# only a comment\n}` is genuinely empty. Telling them apart is a scan of the text the
//! failed attempt covered, with comments removed, looking for a separator at the top level.

use crate::parser::Parser;
use crate::parser::context::ContextValue;
use crate::value::{Object, Value};
use crate::{Error, Result, STRING_DELIMITERS, Schema, pychar};

/// What the text the empty object covered should be read as instead.
enum Repair {
    /// Genuinely empty: keep the object.
    Keep,
    /// An object whose quotes arrived escaped, with the text to reparse.
    Object(String),
    /// A set-like body under a salvage schema that expects an object.
    SchemaSetObject,
    /// A set or a run of values: an array.
    Array,
}

impl Parser {
    pub(crate) fn repair_empty_object_result(
        &mut self,
        obj: &Object,
        start_index: usize,
        schema: Option<&Schema>,
        path: &str,
    ) -> Result<Option<Value>> {
        if !obj.is_empty() || self.index as isize - start_index as isize <= 2 {
            return Ok(None);
        }
        if self.strict {
            self.log("Parsed object is empty but contains extra characters in strict mode, raising an error");
            return Err(Error::new(
                "Parsed object is empty but contains extra characters in strict mode.",
            ));
        }

        match self.classify_empty_object_repair(start_index, schema) {
            Repair::Keep => Ok(None),
            Repair::Object(normalized) => {
                let end_index = self.index + 1;
                self.splice(start_index - 1, end_index, &normalized);
                self.index = start_index;
                let repaired = self.reparse_as(ContextValue::ObjectKey, |parser| {
                    parser.parse_object(schema.cloned(), path)
                })?;
                Ok(Some(repaired))
            }
            Repair::SchemaSetObject => {
                self.log(
                    "Parsed object is empty but salvage schema expects an object, reparsing set-like members as null-valued object keys",
                );
                self.index = start_index;
                let items = self.reparse_as(ContextValue::ObjectKey, |parser| {
                    parser.parse_array_items(None, "$", ']')
                })?;
                Ok(Some(set_items_as_object(items)))
            }
            Repair::Array => {
                self.log("Parsed object is empty, we will try to parse this as an array instead");
                self.index = start_index;
                let items = self.reparse_as(ContextValue::ObjectKey, |parser| {
                    parser.parse_array_items(None, "$", ']')
                })?;
                Ok(Some(Value::Array(items)))
            }
        }
    }

    /// Re-reads from `self.index` inside a key context, and leaves that context *deferred* so the
    /// value the caller goes on to parse is read in it too.
    fn reparse_as<T>(
        &mut self,
        context: ContextValue,
        parse: impl FnOnce(&mut Parser) -> Result<T>,
    ) -> Result<T> {
        self.context.set(context);
        let parsed = parse(self);
        self.context.reset();
        let parsed = parsed?;
        self.deferred_contexts.push(context);
        Ok(parsed)
    }

    fn splice(&mut self, start: usize, end: usize, replacement: &str) {
        let end = end.min(self.len());
        self.json_str.splice(start, end, replacement);
    }

    fn classify_empty_object_repair(&self, start_index: usize, schema: Option<&Schema>) -> Repair {
        let attempted_object = self.slice(start_index as isize - 1, self.index as isize + 1);
        let body: String = attempted_object.chars().skip(1).collect();
        let body = body.strip_suffix('}').unwrap_or(&body);
        let body = body.trim_start_matches(pychar::is_space);
        if body.is_empty() {
            return Repair::Keep;
        }
        if (body.starts_with("\\\"") && body.contains("\\\":"))
            || (body.starts_with("\\'") && body.contains("\\':"))
        {
            self.log(
                "Parsed object is empty but the input starts like an escaped object key, normalizing and reparsing it as an object",
            );
            return Repair::Object(attempted_object.replace("\\\"", "\"").replace("\\'", "'"));
        }
        let body = strip_comments(body);
        let body = body.trim_start_matches(pychar::is_space);
        if body.is_empty() {
            return Repair::Keep;
        }
        if has_top_level_separator(body) {
            self.log(
                "Parsed object is empty but the input still contains an object-style separator, keeping object repair",
            );
            return Repair::Keep;
        }
        if self.salvage_schema_expects_an_object(schema) {
            return Repair::SchemaSetObject;
        }
        Repair::Array
    }
}

/// A body whose members all came back as non-empty strings becomes keys with null values.
fn set_items_as_object(items: Vec<Value>) -> Value {
    let keys: Vec<&String> = items
        .iter()
        .filter_map(|item| match item {
            Value::Str(text) if !text.is_empty() => Some(text),
            _ => None,
        })
        .collect();
    if keys.len() != items.len() {
        return Value::Array(items);
    }
    Value::Object(
        keys.into_iter()
            .map(|key| (key.clone(), Value::Null))
            .collect(),
    )
}

/// A `:` outside any string, which says the failed attempt really was an object.
fn has_top_level_separator(body: &str) -> bool {
    let mut in_quote: Option<char> = None;
    let mut backslashes = 0_usize;
    for char in body.chars() {
        if char == '\\' {
            backslashes += 1;
            continue;
        }
        match in_quote {
            Some(quote) => {
                if char == quote && backslashes.is_multiple_of(2) {
                    in_quote = None;
                }
            }
            None if STRING_DELIMITERS.contains(&char) && backslashes.is_multiple_of(2) => {
                in_quote = Some(char);
            }
            None if char == ':' && backslashes.is_multiple_of(2) => return true,
            None => {}
        }
        backslashes = 0;
    }
    false
}

/// `#`, `//` and `/* */` removed, so a comment holding a colon does not read as a member.
fn strip_comments(body: &str) -> String {
    let chars: Vec<char> = body.chars().collect();
    let mut stripped = String::new();
    let mut in_quote: Option<char> = None;
    let mut backslashes = 0_usize;
    let mut index = 0;
    // This loop walks a local copy, out of reach of the source's read counter. Its inner skips
    // are iterator-shaped below, and the outer step count guards what remains.
    #[cfg(debug_assertions)]
    let mut steps: u64 = 0;
    while index < chars.len() {
        #[cfg(debug_assertions)]
        {
            steps += 1;
            assert!(
                steps <= chars.len() as u64 + 8,
                "{steps} steps over {} characters — the comment strip is not advancing",
                chars.len()
            );
        }
        let char = chars[index];
        let next = chars.get(index + 1).copied();

        if char == '\\' {
            backslashes += 1;
            stripped.push(char);
            index += 1;
            continue;
        }
        if let Some(quote) = in_quote {
            stripped.push(char);
            if char == quote && backslashes.is_multiple_of(2) {
                in_quote = None;
            }
            backslashes = 0;
            index += 1;
            continue;
        }
        if STRING_DELIMITERS.contains(&char) && backslashes.is_multiple_of(2) {
            in_quote = Some(char);
            stripped.push(char);
            backslashes = 0;
            index += 1;
            continue;
        }
        backslashes = 0;

        // Both skips are iterator-shaped: a scan with no cursor mutation has no hang site, and
        // six mutants held the index loops these replaced spinning for the full timeout.
        if char == '#' || (char == '/' && next == Some('/')) {
            let from = index + if char == '/' { 2 } else { 1 };
            let tail = chars.get(from..).unwrap_or_default();
            index = from
                + tail
                    .iter()
                    .position(|&ch| ch == '\n' || ch == '\r')
                    .unwrap_or(tail.len());
            continue;
        }
        if char == '/' && next == Some('*') {
            let from = index + 2;
            let tail = chars.get(from..).unwrap_or_default();
            index = match tail.windows(2).position(|pair| pair == ['*', '/']) {
                Some(close) => from + close + 2,
                None => chars.len(),
            };
            continue;
        }
        stripped.push(char);
        index += 1;
    }
    stripped
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_colon_inside_a_comment_is_not_an_object_separator() {
        assert!(has_top_level_separator("\"a\": 1"));
        assert!(!has_top_level_separator("\"a: 1\""));
        assert_eq!(
            strip_comments("\"a\" # note: here\n, \"b\""),
            "\"a\" \n, \"b\""
        );
        assert!(!has_top_level_separator(&strip_comments(
            "\"a\" # note: here\n, \"b\""
        )));
    }

    #[test]
    fn a_block_comment_with_no_closer_swallows_the_rest() {
        assert_eq!(strip_comments("a /* unterminated"), "a ");
        assert_eq!(strip_comments("a /* closed */ b"), "a  b");
    }
}
