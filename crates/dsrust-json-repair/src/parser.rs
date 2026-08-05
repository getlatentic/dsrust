//! `JSONParser`: the cursor every repair heuristic reads and moves.
//!
//! The text is held as code points rather than bytes because that is the unit Python indexes, and
//! every heuristic here is written in offsets — `get_char_at(-1)`, `skip_to_character(idx=i + 1)`,
//! a slice spliced back into the middle of the input. A byte cursor would agree on ASCII and
//! diverge on the first smart quote, which is exactly the input this library exists for.

pub(crate) mod array;
pub(crate) mod comment;
pub(crate) mod context;
pub(crate) mod number;
pub(crate) mod object;
pub(crate) mod parenthesized;
pub(crate) mod source;
pub(crate) mod string;

use std::rc::Rc;

use crate::schema::SchemaRepairer;
use crate::value::Value;
use crate::{Error, LogSink, Result, Schema};
use context::{ContextValue, JsonContext};

/// One entry of the repair log, when a caller asked for one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogEntry {
    /// What the parser did.
    pub text: String,
    /// The twenty code points around the cursor when it did.
    pub context: String,
}

pub(crate) struct Parser {
    /// The input, which the object repairs splice into as they go.
    pub(crate) json_str: source::Source,
    pub(crate) index: usize,
    pub(crate) context: JsonContext,
    pub(crate) deferred_contexts: Vec<ContextValue>,
    pub(crate) logger: LogSink,
    pub(crate) stream_stable: bool,
    pub(crate) strict: bool,
    pub(crate) try_valid_json_suffix: bool,
    pub(crate) has_tried_valid_json_suffix: bool,
    pub(crate) schema_repairer: Option<Rc<SchemaRepairer>>,
    /// How deep `parse_json` is nested. Python answers this with `RecursionError`; a Rust stack
    /// overflow is not catchable, so the limit is explicit — see [`crate::MAX_DEPTH`].
    pub(crate) depth: usize,
    /// Characters read so far, against what a terminating parse can need. Debug builds only.
    ///
    /// Forty-nine of this crate's loops advance a cursor with nothing checking that they do, so a
    /// mutation turning `+= 1` into `*= 1` leaves one spinning — and a test that spins does not
    /// fail, it simply never finishes. Eighty-two mutants were scored that way. Counting reads in
    /// the one place every scan passes through turns each of them into a panic, at a bound no
    /// terminating parse can reach.
    #[cfg(debug_assertions)]
    reads: std::cell::Cell<u64>,
    #[cfg(debug_assertions)]
    read_budget: u64,
}

impl Parser {
    pub(crate) fn new(json_str: &str, options: &crate::Repair, logger: LogSink) -> Self {
        Self {
            json_str: source::Source::of(json_str),
            index: 0,
            context: JsonContext::new(),
            deferred_contexts: Vec::new(),
            logger,
            stream_stable: options.stream_stable,
            strict: options.strict,
            try_valid_json_suffix: false,
            has_tried_valid_json_suffix: false,
            schema_repairer: None,
            depth: 0,
            #[cfg(debug_assertions)]
            reads: std::cell::Cell::new(0),
            // Every position may legitimately start one scan over the rest of the input — that is
            // what `cached_skip_to_character` exists to make cheap — so a terminating parse is
            // bounded by the square of the length. The floor covers short inputs, where the square
            // is smaller than the fixed work of a parse; saturation covers the other end, where a
            // long input's square does not fit.
            #[cfg(debug_assertions)]
            read_budget: {
                let length = json_str.chars().count() as u64;
                length.saturating_mul(length).saturating_add(1 << 20)
            },
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.json_str.len()
    }

    /// `self.json_str[self.index + count]`, including Python's wrap for a negative index — which
    /// `_should_split_duplicate_object` reaches with a lookback well below zero.
    pub(crate) fn get_char_at(&self, count: isize) -> Option<char> {
        #[cfg(debug_assertions)]
        {
            let read = self.reads.get() + 1;
            self.reads.set(read);
            assert!(
                read <= self.read_budget,
                "{read} character reads over an input of {} — a scan is not advancing. \
                 This bound is not reachable by a parse that ends.",
                self.len()
            );
        }
        let position = self.index as isize + count;
        let position = if position < 0 {
            position + self.len() as isize
        } else {
            position
        };
        usize::try_from(position)
            .ok()
            .and_then(|position| self.json_str.at(position))
    }

    pub(crate) fn char_here(&self) -> Option<char> {
        self.get_char_at(0)
    }

    /// `self.json_str[start:end]` with Python's clamping.
    pub(crate) fn slice(&self, start: isize, end: isize) -> String {
        let length = self.len() as isize;
        let clamp = |position: isize| {
            let position = if position < 0 {
                position + length
            } else {
                position
            };
            position.clamp(0, length) as usize
        };
        let (start, end) = (clamp(start), clamp(end));
        if start >= end {
            return String::new();
        }
        self.json_str.slice_string(start, end)
    }

    pub(crate) fn skip_whitespaces(&mut self) {
        self.index = self.json_str.scroll_spaces_from(self.index);
    }

    /// Whitespace from `self.index + idx` on, as an offset from `self.index`. Does not move.
    ///
    /// A negative `idx` cannot occur here — every caller scrolls forward from at or past the
    /// cursor — so the byte walk starts at a real position. The debug assert is what makes that
    /// sentence checkable rather than believed.
    pub(crate) fn scroll_whitespaces(&self, idx: isize) -> isize {
        let start = self.index as isize + idx;
        debug_assert!(start >= 0, "a whitespace scroll from before the input");
        let position = self.json_str.scroll_spaces_from(start.max(0) as usize);
        position as isize - self.index as isize
    }

    /// Advance from `self.index + idx` to the next *unescaped* target, or to the end.
    ///
    /// "Unescaped" counts the run of backslashes immediately before the character, so `\\"` closes
    /// a string and `\"` does not.
    pub(crate) fn skip_to_character(&self, targets: &[char], idx: isize) -> isize {
        let length = self.len() as isize;
        let mut position = self.index as isize + idx;
        let mut backslashes = 0_usize;
        while position < length {
            let Some(ch) = usize::try_from(position)
                .ok()
                .and_then(|at| self.json_str.at(at))
            else {
                break;
            };
            if ch == '\\' {
                backslashes += 1;
                position += 1;
                continue;
            }
            if targets.contains(&ch) && backslashes.is_multiple_of(2) {
                return position - self.index as isize;
            }
            backslashes = 0;
            position += 1;
        }
        length - self.index as isize
    }

    pub(crate) fn log(&self, text: &str) {
        let Some(logger) = &self.logger else { return };
        let window = 10_isize;
        let start = (self.index as isize - window).max(0);
        let end = (self.index as isize + window).min(self.len() as isize);
        logger.borrow_mut().push(LogEntry {
            text: text.to_owned(),
            context: self.slice(start, end),
        });
    }

    /// `parse()`: the whole input, however many values it turns out to hold.
    pub(crate) fn parse(&mut self) -> Result<Value> {
        self.parse_top_level(None)
    }

    /// `parse_with_schema()`: the same walk, with every nested value held to the schema.
    pub(crate) fn parse_with_schema(
        &mut self,
        repairer: Rc<SchemaRepairer>,
        schema: Schema,
    ) -> Result<Value> {
        self.schema_repairer = Some(repairer);
        self.parse_top_level(Some(schema))
    }

    /// Repeated top-level values, gathered into a list — and back out of it when there was one.
    fn parse_top_level(&mut self, schema: Option<Schema>) -> Result<Value> {
        let first = self.parse_json(schema.clone(), "$")?;
        if self.index >= self.len() {
            return Ok(first);
        }
        self.log("The parser returned early, checking if there's more json elements");
        let mut values = vec![first];
        while self.index < self.len() {
            self.context.clear();
            self.deferred_contexts.clear();
            let is_comma_separated = self.next_top_level_value_is_comma_separated();
            let element_start_index = self.index;
            let next = self.parse_json(schema.clone(), "$")?;
            if self.strict && self.index > element_start_index {
                self.log("Multiple top-level JSON elements found in strict mode, raising an error");
                return Err(Error::new(
                    "Multiple top-level JSON elements found in strict mode.",
                ));
            }
            if next.is_truthy() {
                let previous = values.last().expect("the first element is always there");
                if !is_comma_separated && previous.is_same_shape(&next) {
                    // Repeated objects read as updates: the newest wins.
                    values.pop();
                } else if !previous.is_truthy() {
                    values.pop();
                }
                values.push(next);
            } else {
                self.index += 1;
            }
        }
        if values.len() == 1 {
            self.log("There were no more elements, returning the element without the array");
            return Ok(values.pop().expect("just checked the length"));
        }
        Ok(Value::Array(values))
    }

    fn next_top_level_value_is_comma_separated(&self) -> bool {
        if self.get_char_at(self.scroll_whitespaces(0)) == Some(',') {
            return true;
        }
        let mut position = self.index as isize - 1;
        while position >= 0
            && self
                .json_str
                .at(position as usize)
                .is_some_and(crate::pychar::is_space)
        {
            position -= 1;
        }
        position >= 0 && self.json_str.at(position as usize) == Some(',')
    }

    /// The suffix fast path: once past a prefix, a value that is *already* valid JSON is decoded
    /// by CPython's own scanner rather than repaired.
    fn try_parse_valid_json_value(&mut self) -> Option<Value> {
        if !self.try_valid_json_suffix
            || self.has_tried_valid_json_suffix
            || !self.context.empty
            || self.index == 0
        {
            return None;
        }
        self.has_tried_valid_json_suffix = true;
        // The suffix goes to whichever strict scanner matches the storage; the two are one grammar,
        // held together by `tests/scanner_agreement.rs`. An `Ascii` suffix is valid UTF-8 as it
        // stands, and its byte positions are the code-point positions this index needs.
        let (value, end) = match &self.json_str {
            source::Source::Ascii(bytes) => {
                let suffix = std::str::from_utf8(&bytes[self.index..]).expect("ASCII is UTF-8");
                crate::strict_json::bytes::raw_decode(suffix).ok()?
            }
            source::Source::Wide(chars) => {
                crate::strict_json::raw_decode(&chars[self.index..]).ok()?
            }
        };
        self.index += end;
        Some(value)
    }

    pub(crate) fn parse_json(&mut self, schema: Option<Schema>, path: &str) -> Result<Value> {
        self.depth += 1;
        if self.depth > crate::MAX_DEPTH {
            self.depth -= 1;
            return Err(Error::new(
                "Input nesting exceeds the supported parser recursion depth.",
            ));
        }
        let parsed = self.parse_json_inner(schema, path);
        self.depth -= 1;
        parsed
    }

    fn parse_json_inner(&mut self, schema: Option<Schema>, path: &str) -> Result<Value> {
        if !self.deferred_contexts.is_empty() {
            let deferred = std::mem::take(&mut self.deferred_contexts);
            for value in &deferred {
                self.context.set(*value);
            }
            let parsed = self.parse_json(schema, path);
            for _ in &deferred {
                self.context.reset();
            }
            return parsed;
        }

        let (guided, schema) = self.resolve_schema_for_parse(schema)?;
        loop {
            let Some(char) = self.char_here() else {
                return Ok(Value::Str(String::new()));
            };
            if self.try_valid_json_suffix
                && matches!(char, '{' | '[')
                && let Some(value) = self.try_parse_valid_json_value()
            {
                return self.finalize(value, guided, schema.as_ref(), path);
            }
            let value = match char {
                '{' => {
                    self.index += 1;
                    self.parse_object(guided.then(|| schema.clone()).flatten(), path)?
                }
                '[' => {
                    self.index += 1;
                    self.parse_array(guided.then(|| schema.clone()).flatten(), path, ']')?
                }
                // A tuple literal, or prose that merely opens a bracket. Top-level detection stays
                // conservative so `note (clarification):` does not swallow the JSON after it.
                '(' if !self.context.empty || self.top_level_parenthesized_can_start_value() => {
                    self.parse_parenthesized(guided.then(|| schema.clone()).flatten(), path)?
                }
                '(' => {
                    self.index += 1;
                    continue;
                }
                '#' | '/' => self.parse_comment()?,
                char if !self.context.empty
                    && (crate::STRING_DELIMITERS.contains(&char)
                        || crate::pychar::is_alpha(char)) =>
                {
                    self.parse_string()?
                }
                char if !self.context.empty
                    && (crate::pychar::is_digit(char) || char == '-' || char == '.') =>
                {
                    self.parse_number()?
                }
                // Everything else is prose to step over.
                _ => {
                    self.index += 1;
                    continue;
                }
            };
            return self.finalize(value, guided, schema.as_ref(), path);
        }
    }

    /// Whether the repairer applies to this node, and the schema it resolved to.
    ///
    /// A schema of `None` or `True` constrains nothing, so the guided path is skipped rather than
    /// entered with a schema that would accept anything.
    fn resolve_schema_for_parse(
        &mut self,
        schema: Option<Schema>,
    ) -> Result<(bool, Option<Schema>)> {
        let Some(repairer) = self.schema_repairer.clone() else {
            return Ok((false, schema));
        };
        // `schema not in (None, True)`. A null schema resolves to `True` a line below, so it
        // needs no arm of its own here.
        match &schema {
            None | Some(Value::Bool(true)) => return Ok((false, schema)),
            _ => {}
        }
        let resolved = repairer.resolve_schema(schema.as_ref())?;
        match resolved {
            Value::Bool(true) => Ok((false, Some(resolved))),
            Value::Bool(false) => Err(Error::new("Schema does not allow any values.")),
            resolved => Ok((true, Some(resolved))),
        }
    }

    fn finalize(
        &mut self,
        value: Value,
        guided: bool,
        schema: Option<&Schema>,
        path: &str,
    ) -> Result<Value> {
        if !guided {
            return Ok(value);
        }
        let repairer = self
            .schema_repairer
            .clone()
            .expect("guided implies a repairer");
        repairer.repair_value(Some(value), schema, path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parser(text: &str) -> Parser {
        Parser::new(text, &crate::Repair::new(), None)
    }

    #[test]
    fn a_negative_lookback_wraps_the_way_a_python_index_does() {
        let parser = parser("abc");
        assert_eq!(
            parser.get_char_at(-1),
            Some('c'),
            "index 0 minus one is the last character"
        );
        assert_eq!(parser.get_char_at(0), Some('a'));
        assert_eq!(parser.get_char_at(3), None);
        assert_eq!(
            parser.get_char_at(-4),
            None,
            "past the front is out of range, not a wrap twice"
        );
    }

    #[test]
    fn skip_to_character_counts_the_backslash_run_before_the_target() {
        let mut parser = parser(r#"a\"b"c"#);
        parser.index = 1;
        // `\"` is escaped; the quote at 4 is not.
        assert_eq!(
            parser.index + parser.skip_to_character(&['"'], 0) as usize,
            4
        );
    }

    #[test]
    fn the_cursor_counts_code_points_rather_than_bytes() {
        let parser = parser("“a”");
        assert_eq!(parser.len(), 3);
        assert_eq!(parser.get_char_at(1), Some('a'));
        assert_eq!(parser.slice(0, 2), "“a");
    }
}
