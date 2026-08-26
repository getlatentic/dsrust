//! The scan: one character at a time until something says the string ended.
//!
//! The order of the checks is the specification. A `,` is tested for ending a member before the
//! container stack sees it; a `}` is tested against the brace count the accumulator carries before
//! anything looks ahead; an escape is normalised only after the character has already been
//! appended. Reordering any of them changes what comes out.

pub(crate) mod delimiter;

use crate::parser::Parser;
use crate::parser::context::ContextValue;
use crate::parser::string::helpers::{CommaMeaning, update_inline_container_stack};
use crate::parser::string::{INLINE_CONTAINER_OPENERS, StringParseState};
use crate::{Result, pychar};

/// What one rule decided about the character in hand.
enum Step {
    /// The string ends here, on the character named — which is not always the one the rule was
    /// handed. `handle_right_delimiter_candidate` steps the cursor *back* onto the character
    /// before a misplaced quote, and `finalize_string_result` then reads that character to decide
    /// whether the string closed on its delimiter or ran out. Breaking with the old one moves the
    /// cursor a character past where the next key starts.
    Stop(Option<char>),
    /// Handled; look at the character the rule left the cursor on.
    Again(Option<char>),
    /// No rule applied.
    Fall,
}

impl Parser {
    /// Appends the character, advances, and answers with the next one.
    fn append_literal_char(&mut self, state: &mut StringParseState, char: char) -> Option<char> {
        state.append(&[char]);
        self.index += 1;
        self.char_here()
    }

    pub(crate) fn scan_string_body(
        &mut self,
        state: &mut StringParseState,
    ) -> Result<Option<char>> {
        let outer = state.outer_rstring_delimiter();
        let mut char = self.char_here();
        while let Some(current) = char {
            if current == outer && !state.in_low_smart_quote_span() {
                break;
            }
            match self.before_appending(state, current)? {
                Step::Stop(next) => {
                    char = next;
                    break;
                }
                Step::Again(next) => {
                    char = next;
                    continue;
                }
                Step::Fall => {}
            }

            state.append(&[current]);
            self.index += 1;
            char = self.char_here();
            let Some(current) = char else {
                if self.stream_stable && state.last_acc() == Some('\\') {
                    state.pop_acc();
                    state.rebuild_unmatched_opening_braces();
                }
                break;
            };
            if state.last_acc() == Some('\\') {
                let (handled, next) = self.normalize_escape_sequence(state, current);
                if handled {
                    char = next;
                    continue;
                }
                char = next;
            }
            match self.after_appending(state, char)? {
                Step::Stop(next) => {
                    char = next;
                    break;
                }
                Step::Again(next) => char = next,
                Step::Fall => {}
            }
        }
        Ok(char)
    }

    /// The rules that look at a character before it joins the string.
    fn before_appending(&mut self, state: &mut StringParseState, char: char) -> Result<Step> {
        if state.missing_quotes {
            if self.context.is(ContextValue::ObjectKey) && (char == ':' || pychar::is_space(char)) {
                self.log(
                    "While parsing a string missing the left delimiter in object key context, we found a :, stopping here",
                );
                return Ok(Step::Stop(Some(char)));
            }
            if self.context.is(ContextValue::Array) && (char == ']' || char == ',') {
                self.log(
                    "While parsing a string missing the left delimiter in array context, we found a ] or ,, stopping here",
                );
                return Ok(Step::Stop(Some(char)));
            }
        }

        if char == '„' && state.last_acc() != Some('\\') {
            state.push_low_smart_quote_span();
            return Ok(Step::Again(self.append_literal_char(state, char)));
        }
        if state.in_low_smart_quote_span() && char == '”' {
            state.pop_low_smart_quote_span();
            return Ok(Step::Again(self.append_literal_char(state, char)));
        }

        if let Step::Again(next) = self.keep_inline_container(state, char) {
            return Ok(Step::Again(next));
        }
        if let step @ (Step::Stop(_) | Step::Again(_)) = self.object_value_comma(state, char) {
            return Ok(step);
        }

        let (pending, keep) = update_inline_container_stack(
            char,
            state.pending_inline_container,
            &mut state.inline_container_stack,
        );
        state.pending_inline_container = pending;
        if keep {
            return Ok(Step::Again(self.append_literal_char(state, char)));
        }

        if let step @ (Step::Stop(_) | Step::Again(_)) =
            self.object_value_closing_brace(state, char)
        {
            return Ok(step);
        }
        if !self.stream_stable
            && char == ']'
            && self.context.contains(ContextValue::Array)
            && state.last_acc() != Some(state.outer_rstring_delimiter())
            && self
                .get_char_at(self.skip_to_character(&[state.outer_rstring_delimiter()], 0))
                .is_none()
        {
            return Ok(Step::Stop(Some(char)));
        }
        Ok(self.closing_brace_before_code_fence(state, char))
    }

    /// The rules that look at the character the cursor moved onto.
    fn after_appending(
        &mut self,
        state: &mut StringParseState,
        char: Option<char>,
    ) -> Result<Step> {
        let Some(char) = char else {
            return Ok(Step::Fall);
        };
        let outer = state.outer_rstring_delimiter();

        if char == ':' && !state.missing_quotes && self.context.is(ContextValue::ObjectKey) {
            return Ok(self.object_key_colon(state, char));
        }
        if state.in_low_smart_quote_span() && char == '"' {
            state.pop_low_smart_quote_span();
            return Ok(Step::Again(self.append_literal_char(state, char)));
        }
        if char == outer
            && self.context.is(ContextValue::ObjectValue)
            && self.quote_belongs_to_regex_character_class(state)
        {
            self.log("While parsing a string, we found a bare quote inside a regex character class, keeping it");
            return Ok(Step::Again(self.append_literal_char(state, char)));
        }
        if char == outer && state.last_acc().is_some_and(|last| last != '\\') {
            let (handled, next, should_break) = self.handle_right_delimiter_candidate(state, char);
            if should_break {
                return Ok(Step::Stop(next));
            }
            if handled {
                return Ok(Step::Again(next));
            }
        }
        Ok(Step::Fall)
    }

    /// A balanced `{...}` or `[...]` that belongs to the value rather than to the structure.
    fn keep_inline_container(&mut self, state: &mut StringParseState, char: char) -> Step {
        let opens_a_kept_container = state.pending_inline_container
            || (self.context.is(ContextValue::ObjectValue)
                && char == '{'
                && self.get_char_at(-1) != Some('\\')
                && self.bare_key_is_followed_by_colon(self.scroll_whitespaces(1)));
        if !opens_a_kept_container
            || !INLINE_CONTAINER_OPENERS.contains(&char)
            || state.last_acc() == Some('\\')
        {
            return Step::Fall;
        }
        let Some(container_end_idx) = self.skip_inline_container(0) else {
            return Step::Fall;
        };
        self.log(
            "While parsing a string in object value context, we found a balanced inline container that belongs to the string, keeping it",
        );
        state.pending_inline_container = false;
        state.inline_container_stack.clear();
        let span: Vec<char> = self
            .json_str
            .chars_vec(self.index, self.index + container_end_idx as usize);
        state.append(&span);
        self.index += container_end_idx as usize;
        Step::Again(self.char_here())
    }

    /// A `,` inside an unterminated object value: the end of the member, or part of the text.
    fn object_value_comma(&mut self, state: &mut StringParseState, char: char) -> Step {
        if self.stream_stable
            || !self.context.is(ContextValue::ObjectValue)
            || char != ','
            || state.pending_inline_container
            || !state.inline_container_stack.is_empty()
        {
            return Step::Fall;
        }
        let meaning = if state.object_value_has_no_future_delimiter {
            CommaMeaning::Str
        } else {
            self.classify_object_value_comma(state)
        };
        if meaning == CommaMeaning::Member {
            self.log(
                "While parsing a string missing the right delimiter in object value context, we found a comma that starts the next object member. Stopping here",
            );
            return Step::Stop(Some(char));
        }
        if meaning == CommaMeaning::StrNoFutureDelimiter {
            state.object_value_has_no_future_delimiter = true;
        }
        state.pending_inline_container = meaning == CommaMeaning::Container;
        self.log("While parsing a string in object value context, we found a comma that belongs to the string, keeping it");
        Step::Again(self.append_literal_char(state, char))
    }

    /// A `}` inside an unterminated object value, judged by whether a closing quote can be found
    /// anywhere that would make this brace the object's rather than the string's.
    fn object_value_closing_brace(&mut self, state: &mut StringParseState, char: char) -> Step {
        let outer = state.outer_rstring_delimiter();
        if self.stream_stable
            || !self.context.is(ContextValue::ObjectValue)
            || char != '}'
            || state.last_acc() == Some(outer)
        {
            return Step::Fall;
        }
        if state.object_value_unmatched_opening_braces > 0 {
            return Step::Again(self.append_literal_char(state, char));
        }

        let mut rstring_delimiter_missing = true;
        self.skip_whitespaces();
        if self.get_char_at(1) == Some('\\') {
            rstring_delimiter_missing = false;
        }
        let mut i = self.cached_skip_to_character(state, &[outer], 1);
        if self.get_char_at(i).is_some() {
            i = self.scroll_whitespaces(i + 1);
            match self.get_char_at(i) {
                None | Some(',' | '}') => rstring_delimiter_missing = false,
                _ => {
                    i = self.skip_to_character(&[state.lstring_delimiter], i);
                    if self.get_char_at(i).is_none() {
                        rstring_delimiter_missing = false;
                    } else {
                        i = self.scroll_whitespaces(i + 1);
                        if self.get_char_at(i).is_some_and(|next| next != ':') {
                            rstring_delimiter_missing = false;
                        }
                    }
                }
            }
        } else {
            let i = self.skip_to_character(&[':'], 1);
            if self.get_char_at(i).is_some() {
                return Step::Stop(Some(char));
            }
            let i = self.scroll_whitespaces(1);
            let j = self.skip_to_character(&['}'], i);
            if j - i > 1 {
                rstring_delimiter_missing = false;
            }
        }
        if rstring_delimiter_missing {
            self.log(
                "While parsing a string missing the left delimiter in object value context, we found a , or } and we couldn't determine that a right delimiter was present. Stopping here",
            );
            return Step::Stop(Some(char));
        }
        Step::Fall
    }

    /// A `}` right before ```` ``` ````: either the object closing before a wrapper fence, or a
    /// fenced snippet written inside the value.
    fn closing_brace_before_code_fence(
        &mut self,
        state: &mut StringParseState,
        char: char,
    ) -> Step {
        if !self.context.is(ContextValue::ObjectValue) || char != '}' {
            return Step::Fall;
        }
        let i = self.scroll_whitespaces(1);
        let next_c = self.get_char_at(i);
        if next_c == Some('`')
            && self.get_char_at(i + 1) == Some('`')
            && self.get_char_at(i + 2) == Some('`')
        {
            if self.brace_before_code_fence_belongs_to_string(state, i) {
                self.log(
                    "While parsing a string in object value context, we found a literal fenced snippet after }, keeping it in the string",
                );
                return Step::Again(self.append_literal_char(state, char));
            }
            self.log(
                "While parsing a string in object value context, we found a } that closes the object before code fences, stopping here",
            );
            return Step::Stop(Some(char));
        }
        if next_c.is_none() {
            self.log("While parsing a string in object value context, we found a } that closes the object, stopping here");
            return Step::Stop(Some(char));
        }
        Step::Fall
    }

    /// A `:` inside what was supposed to be a key: the key ended before it, unless a quoted value
    /// and a separator can still be found after it.
    fn object_key_colon(&mut self, state: &mut StringParseState, char: char) -> Step {
        let outer = state.outer_rstring_delimiter();
        let mut i = self.skip_to_character(&[state.lstring_delimiter], 1);
        if self.get_char_at(i).is_none() {
            self.log(
                "While parsing a string missing the right delimiter in object key context, we found a :, stopping here",
            );
            return Step::Stop(Some(char));
        }
        i = self.skip_to_character(&[outer], i + 1);
        if self.get_char_at(i).is_none() {
            return Step::Fall;
        }
        i = self.scroll_whitespaces(i + 1);
        if let Some(ch @ (',' | '}')) = self.get_char_at(i) {
            self.log(&format!(
                "While parsing a string missing the right delimiter in object key context, we found a {ch} stopping here"
            ));
            return Step::Stop(Some(char));
        }
        Step::Fall
    }
}
