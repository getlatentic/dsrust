//! A quote that matches the one the string opened with: does it close the string?
//!
//! This is where `{answer: "{}", "unknown: "7", reasoning": "[]"` is decided. The answer depends on
//! which container the cursor is in and on what lies between this quote and the next structural
//! character — a colon further on means the quote opened the next key rather than closing this
//! value, and an *even* number of quotes before the closing bracket of an array means the quoted
//! run inside was never a string at all.

use crate::parser::Parser;
use crate::parser::context::ContextValue;
use crate::parser::string::StringParseState;

/// Whether the quote was handled, the character to look at next, and whether the string ends here.
pub(crate) type Candidate = (bool, Option<char>, bool);

impl Parser {
    pub(crate) fn handle_right_delimiter_candidate(
        &mut self,
        state: &mut StringParseState,
        char: char,
    ) -> Candidate {
        let outer = state.outer_rstring_delimiter();

        if state.doubled_quotes && self.get_char_at(1) == Some(outer) {
            self.log("While parsing a string, we found a doubled quote, ignoring it");
            self.index += 1;
            return (true, Some(char), false);
        }

        if state.missing_quotes && self.context.is(ContextValue::ObjectValue) {
            return self.quote_in_unquoted_object_value(state, char);
        }

        if state.unmatched_delimiter {
            state.unmatched_delimiter = false;
            let next_char = self.append_quote(state, char);
            return (true, next_char, false);
        }

        let (i, next_c) = self.scan_to_structural_character(state);
        if next_c == Some(',') && self.context.is(ContextValue::ObjectValue) {
            return self.quote_before_comma(state, char, i);
        }
        if next_c == Some(outer) && self.get_char_at(i - 1) != Some('\\') {
            return self.quote_before_quote(state, char, i);
        }
        (false, Some(char), false)
    }

    /// Appends the quote as text and answers with the character after it.
    fn append_quote(&mut self, state: &mut StringParseState, char: char) -> Option<char> {
        state.append(&[char]);
        self.index += 1;
        self.char_here()
    }

    /// A string that never opened with a quote, in an object value. A quote here is the start of
    /// the next key when a colon follows it, and the value ends one character back.
    fn quote_in_unquoted_object_value(
        &mut self,
        state: &StringParseState,
        char: char,
    ) -> Candidate {
        let outer = state.outer_rstring_delimiter();
        let mut i = 1;
        let mut next_c = self.get_char_at(i);
        while next_c.is_some_and(|next| next != outer && next != state.lstring_delimiter) {
            i += 1;
            next_c = self.get_char_at(i);
        }
        if next_c.is_some() {
            let i = self.scroll_whitespaces(i + 1);
            if self.get_char_at(i) == Some(':') {
                self.index -= 1;
                let next_char = self.char_here();
                self.log(
                    "In a string with missing quotes and object value context, I found a delimeter but it turns out it was the beginning on the next key. Stopping here.",
                );
                return (false, next_char, true);
            }
        }
        (false, Some(char), false)
    }

    /// Forward to the first character that could end this value, and what it is.
    ///
    /// A comma only counts while nothing alphabetic has been seen — `"a, b" : ...` is prose, and
    /// `"a", "b": ...` is not.
    fn scan_to_structural_character(&self, state: &StringParseState) -> (isize, Option<char>) {
        let outer = state.outer_rstring_delimiter();
        let mut i = 1;
        let mut next_c = self.get_char_at(i);
        let mut check_comma_in_object_value = true;
        while let Some(next) = next_c {
            if next == outer || next == state.lstring_delimiter {
                break;
            }
            if check_comma_in_object_value && crate::pychar::is_alpha(next) {
                check_comma_in_object_value = false;
            }
            let ends_here = (self.context.contains(ContextValue::ObjectKey)
                && matches!(next, ':' | '}'))
                || (self.context.contains(ContextValue::ObjectValue) && next == '}')
                || (self.context.contains(ContextValue::Array) && matches!(next, ']' | ','))
                || (check_comma_in_object_value
                    && self.context.is(ContextValue::ObjectValue)
                    && next == ',');
            if ends_here {
                break;
            }
            i += 1;
            next_c = self.get_char_at(i);
        }
        (i, next_c)
    }

    /// A quote whose next structural character is a comma. If the run after it closes on `}` or
    /// another comma, the quote was inside the value.
    fn quote_before_comma(
        &mut self,
        state: &mut StringParseState,
        char: char,
        i: isize,
    ) -> Candidate {
        let outer = state.outer_rstring_delimiter();
        let i = self.skip_to_character(&[outer], i + 1);
        let i = self.scroll_whitespaces(i + 1);
        if matches!(self.get_char_at(i), Some('}' | ',')) {
            self.log(
                "While parsing a string, we found a misplaced quote that would have closed the string but has a different meaning here, ignoring it",
            );
            let next_char = self.append_quote(state, char);
            return (true, next_char, false);
        }
        (false, Some(char), false)
    }

    /// A quote whose next structural character is another quote — the shape asymmetric quoting
    /// produces, and the one each container answers differently.
    fn quote_before_quote(
        &mut self,
        state: &mut StringParseState,
        char: char,
        i: isize,
    ) -> Candidate {
        if self.only_whitespace_until(i)
            && !(self.context.is(ContextValue::ObjectValue) && self.quoted_object_member_follows(i))
        {
            return (false, Some(char), true);
        }
        match self.context.current {
            Some(ContextValue::ObjectValue) => {
                self.quote_before_quote_in_object_value(state, char, i)
            }
            Some(ContextValue::Array) => self.quote_before_quote_in_array(state, char, i),
            Some(ContextValue::ObjectKey) => {
                self.log(
                    "While parsing a string in Object Key context, we detected a quoted section that would have closed the string but has a different meaning here, ignoring it",
                );
                let next_char = self.append_quote(state, char);
                (true, next_char, false)
            }
            None => (false, Some(char), false),
        }
    }

    fn quote_before_quote_in_object_value(
        &mut self,
        state: &mut StringParseState,
        char: char,
        i: isize,
    ) -> Candidate {
        let outer = state.outer_rstring_delimiter();
        if self.quoted_object_member_follows(i) {
            self.log(
                "While parsing a string, we found a misplaced quote that would have closed the string but has a different meaning here, ignoring it",
            );
            let next_char = self.append_quote(state, char);
            return (true, next_char, false);
        }
        let mut i = self.skip_to_character(&[outer], i + 1) + 1;
        let mut next_c = self.get_char_at(i);
        while next_c.is_some_and(|next| next != ':') {
            let next = next_c.expect("just matched");
            if matches!(next, ',' | ']' | '}')
                || (next == outer && self.get_char_at(i - 1) != Some('\\'))
            {
                break;
            }
            i += 1;
            next_c = self.get_char_at(i);
        }
        if next_c != Some(':') {
            self.log(
                "While parsing a string, we found a misplaced quote that would have closed the string but has a different meaning here, ignoring it",
            );
            state.unmatched_delimiter = !state.unmatched_delimiter;
            let next_char = self.append_quote(state, char);
            return (true, next_char, false);
        }
        (false, Some(char), false)
    }

    /// In an array, an *even* number of quotes between here and the closing bracket means the
    /// quoted run belongs to this item rather than starting a new one.
    fn quote_before_quote_in_array(
        &mut self,
        state: &mut StringParseState,
        char: char,
        i: isize,
    ) -> Candidate {
        let outer = state.outer_rstring_delimiter();
        let mut i = i;
        let mut next_c = Some(outer);
        let mut even_delimiters = true;
        while next_c == Some(outer) {
            i = self.skip_to_character(&[outer, ']'], i + 1);
            next_c = self.get_char_at(i);
            if next_c != Some(outer) {
                even_delimiters = false;
                break;
            }
            i = self.skip_to_character(&[outer, ']'], i + 1);
            next_c = self.get_char_at(i);
        }
        if even_delimiters {
            self.log(
                "While parsing a string in Array context, we detected a quoted section that would have closed the string but has a different meaning here, ignoring it",
            );
            state.unmatched_delimiter = !state.unmatched_delimiter;
            let next_char = self.append_quote(state, char);
            return (true, next_char, false);
        }
        (false, Some(char), true)
    }
}
