//! What the scan looks *forward* to before deciding a character ends the string.
//!
//! Each of these answers one question about text the cursor has not reached: does a balanced
//! container start here, does an object member start here, does a quote further on close a string
//! or open the next key. They never move the cursor.

use crate::parser::Parser;
use crate::parser::string::{
    INLINE_CONTAINER_OPENERS, StringParseState, inline_container_closer, matching_string_delimiter,
};
use crate::{STRING_DELIMITERS, pychar};

impl Parser {
    /// A cached lookahead answer, checked against the scan it stands in for. Debug builds only.
    ///
    /// This cache is a memoisation and nothing else, so its whole contract is one equality: what it
    /// returns must be what `skip_to_character` would have returned. Every line of it runs on
    /// ordinary input — its Python counterpart sits at 100% of statements — and eighteen mutants
    /// still survived, because running a line is not the same as checking what it produced. The
    /// bounds can be widened, the guards inverted and the arithmetic shifted, and a corpus only
    /// notices where the wrong answer happens to change a parse.
    ///
    /// Stating the invariant catches all of that at the point it breaks, and it costs release
    /// builds nothing — which is the whole reason the cache exists.
    #[inline]
    fn answers_as_the_scan_does(&self, cached: isize, targets: &[char], idx: isize) -> isize {
        debug_assert_eq!(
            cached,
            self.skip_to_character(targets, idx),
            "the lookahead cache answered {cached} where the scan it caches disagrees, for \
             {targets:?} at offset {idx}"
        );
        cached
    }

    /// `skip_to_character`, remembering the answer per target set for the length of one string.
    ///
    /// The scan asks the same lookahead question once per character of a long unterminated value,
    /// and the answer only moves when the cursor passes it.
    pub(crate) fn cached_skip_to_character(
        &self,
        state: &mut StringParseState,
        targets: &[char],
        idx: isize,
    ) -> isize {
        let start_index = self.index as isize + idx;
        if let Some((_, cached_start, cached_match)) = state
            .lookahead_cache
            .iter()
            .find(|(key, _, _)| key == targets)
        {
            match cached_match {
                None if start_index >= *cached_start => {
                    return self.answers_as_the_scan_does(
                        self.len() as isize - self.index as isize,
                        targets,
                        idx,
                    );
                }
                Some(cached_match)
                    if *cached_start <= start_index && start_index <= *cached_match =>
                {
                    return self.answers_as_the_scan_does(
                        cached_match - self.index as isize,
                        targets,
                        idx,
                    );
                }
                _ => {}
            }
        }

        let match_offset = self.skip_to_character(targets, idx);
        let remember = |state: &mut StringParseState, entry: (isize, Option<isize>)| match state
            .lookahead_cache
            .iter_mut()
            .find(|(key, _, _)| key == targets)
        {
            Some(slot) => (slot.1, slot.2) = entry,
            None => state
                .lookahead_cache
                .push((targets.to_vec(), entry.0, entry.1)),
        };
        if self.get_char_at(match_offset).is_none() {
            remember(state, (start_index, None));
            return match_offset;
        }
        let match_index = self.index as isize + match_offset;
        if match_index == 0 || self.get_char_at(match_offset - 1) != Some('\\') {
            remember(state, (start_index, Some(match_index)));
        }
        match_offset
    }

    /// Whether the quote at the cursor sits inside a compact `[...]` character class, as a regex
    /// written into a string does.
    pub(crate) fn quote_belongs_to_regex_character_class(&self, state: &StringParseState) -> bool {
        let Some(start) = state.regex_character_class_start else {
            return false;
        };
        if state.string_acc[start.min(state.string_acc.len())..]
            .iter()
            .copied()
            .any(pychar::is_space)
        {
            return false;
        }
        let closing_bracket_idx = self.skip_to_character(&[']'], 1);
        self.get_char_at(closing_bracket_idx) == Some(']')
    }

    /// Whether a `}` immediately before a code fence closes the object or belongs to the string.
    pub(crate) fn brace_before_code_fence_belongs_to_string(
        &self,
        state: &mut StringParseState,
        fence_idx: isize,
    ) -> bool {
        let mut quote_search_idx = fence_idx + 3;
        let next_content_idx = self.scroll_comment_prefixed_member_start(quote_search_idx);
        let mut keep_post_fence_container = false;
        if self
            .get_char_at(next_content_idx)
            .is_some_and(|char| INLINE_CONTAINER_OPENERS.contains(&char))
            && let Some(container_end_idx) = self.skip_inline_container(next_content_idx)
        {
            if self.post_fence_container_starts_next_member(container_end_idx) {
                return false;
            }
            keep_post_fence_container = true;
            quote_search_idx = container_end_idx;
        }

        let outer = state.outer_rstring_delimiter();
        let mut quote_idx = self.skip_to_character(&[outer], quote_search_idx);
        while self.get_char_at(quote_idx) == Some(outer) {
            let after_quote_idx = self.scroll_whitespaces(quote_idx + 1);
            if matches!(
                self.get_char_at(after_quote_idx),
                Some(',' | '}' | ']') | None
            ) {
                if keep_post_fence_container {
                    state.pending_inline_container = true;
                }
                return true;
            }
            quote_idx = self.skip_to_character(&[outer], quote_idx + 1);
        }
        false
    }

    /// Whether a bare word at `key_idx` is a key: alphanumeric or `_` to start, then a colon.
    pub(crate) fn bare_key_is_followed_by_colon(&self, key_idx: isize) -> bool {
        let Some(key_char) = self.get_char_at(key_idx) else {
            return false;
        };
        if !(pychar::is_alnum(key_char) || key_char == '_') {
            return false;
        }
        let key_idx = self.scroll_whitespaces(self.scan_bare_key(key_idx));
        self.get_char_at(key_idx) == Some(':')
    }

    /// Whether what follows a container after a code fence is the next member rather than more of
    /// the same value.
    fn post_fence_container_starts_next_member(&self, container_end_idx: isize) -> bool {
        let after_container_idx = self.scroll_whitespaces(container_end_idx);
        match self.get_char_at(after_container_idx) {
            Some('}') | None => return true,
            Some(',') => {}
            _ => return false,
        }
        let next_member_idx = self.scroll_comment_prefixed_member_start(after_container_idx + 1);
        matches!(self.get_char_at(next_member_idx), Some('}') | None)
            || self.object_member_starts_at(next_member_idx)
    }

    /// Whether the bracket at `idx` opens a container nested in a value, rather than closing one
    /// piece of prose and opening another.
    fn starts_nested_inline_container(&self, idx: isize) -> bool {
        let opening_delimiter = self.get_char_at(idx);
        let mut prev_idx = idx - 1;
        while prev_idx >= 0 {
            let Some(prev_char) = self.get_char_at(prev_idx) else {
                return true;
            };
            if !pychar::is_space(prev_char) {
                if INLINE_CONTAINER_OPENERS.contains(&prev_char) {
                    return true;
                }
                if prev_char != ',' && prev_char != ':' {
                    return false;
                }
                let next_idx = self.scroll_whitespaces(idx + 1);
                let next_char = self.get_char_at(next_idx);
                if matches!(opening_delimiter, Some('[' | '(')) {
                    return next_char.is_some_and(|char| {
                        matches!(char, ']' | ')' | '-' | 't' | 'f' | 'n')
                            || STRING_DELIMITERS.contains(&char)
                            || INLINE_CONTAINER_OPENERS.contains(&char)
                            || pychar::is_digit(char)
                    });
                }
                if opening_delimiter != Some('{') {
                    return false;
                }
                if next_char.is_some_and(|char| char == '}' || STRING_DELIMITERS.contains(&char)) {
                    return true;
                }
                return prev_char == ':' && self.bare_key_is_followed_by_colon(next_idx);
            }
            prev_idx -= 1;
        }
        true
    }

    /// Past a balanced container starting at `idx`, or `None` when it never closes.
    pub(crate) fn skip_inline_container(&self, idx: isize) -> Option<isize> {
        let opening_delimiter = self.get_char_at(idx);
        let Some(closer) = opening_delimiter.and_then(inline_container_closer) else {
            return Some(idx);
        };

        let mut stack = vec![closer];
        let mut i = idx + 1;
        while !stack.is_empty() {
            let char = self.get_char_at(i)?;
            if STRING_DELIMITERS.contains(&char) {
                let end_delimiter = matching_string_delimiter(char);
                i = self.skip_to_character(&[end_delimiter], i + 1);
                if self.get_char_at(i) != Some(end_delimiter) {
                    return None;
                }
            } else if inline_container_closer(char).is_some()
                && self.starts_nested_inline_container(i)
            {
                stack.push(inline_container_closer(char).expect("just checked it opens one"));
            } else if Some(&char) == stack.last() {
                stack.pop();
                if stack.is_empty() {
                    return Some(i + 1);
                }
            }
            i += 1;
        }
        None
    }

    /// Past whitespace and any comments, to where the next member could begin.
    pub(crate) fn scroll_comment_prefixed_member_start(&self, idx: isize) -> isize {
        let mut idx = self.scroll_whitespaces(idx);
        loop {
            match self.get_char_at(idx) {
                Some('#') => {
                    idx = self.scroll_whitespaces(self.skip_to_line_end(idx));
                }
                Some('/') => match self.get_char_at(idx + 1) {
                    Some('/') => {
                        idx = self.scroll_whitespaces(self.skip_to_line_end(idx + 2));
                    }
                    Some('*') => {
                        idx += 2;
                        loop {
                            match self.get_char_at(idx) {
                                None => return idx,
                                Some('*') if self.get_char_at(idx + 1) == Some('/') => {
                                    idx += 2;
                                    break;
                                }
                                _ => idx += 1,
                            }
                        }
                        idx = self.scroll_whitespaces(idx);
                    }
                    _ => return idx,
                },
                _ => return idx,
            }
        }
    }

    fn skip_to_line_end(&self, mut idx: isize) -> isize {
        while self
            .get_char_at(idx)
            .is_some_and(|char| char != '\n' && char != '\r')
        {
            idx += 1;
        }
        idx
    }

    /// Whether a quote at `quote_idx` is followed by a comma and then another object member.
    pub(crate) fn quoted_object_member_follows(&self, quote_idx: isize) -> bool {
        let comma_idx = self.scroll_whitespaces(quote_idx + 1);
        if self.get_char_at(comma_idx) != Some(',') {
            return false;
        }
        let next_member_idx = self.scroll_comment_prefixed_member_start(comma_idx + 1);
        self.object_member_starts_at(next_member_idx)
    }

    /// Whether a key — quoted or bare — with a colon after it starts at `next_member_idx`.
    pub(crate) fn object_member_starts_at(&self, next_member_idx: isize) -> bool {
        let Some(next_member) = self.get_char_at(next_member_idx) else {
            return false;
        };
        if next_member == '}' {
            return false;
        }
        if STRING_DELIMITERS.contains(&next_member) {
            let key_end_delimiter = matching_string_delimiter(next_member);
            let key_end_idx = self.skip_to_character(&[key_end_delimiter], next_member_idx + 1);
            if self.get_char_at(key_end_idx) != Some(key_end_delimiter) {
                return false;
            }
            let after_key_idx = self.scroll_whitespaces(key_end_idx + 1);
            return self.get_char_at(after_key_idx) == Some(':');
        }
        if pychar::is_alnum(next_member) || next_member == '_' {
            return self.bare_key_is_followed_by_colon(next_member_idx);
        }
        false
    }

    /// Whether everything between the cursor and `end` is whitespace.
    pub(crate) fn only_whitespace_until(&self, end: isize) -> bool {
        (1..end).all(|offset| self.get_char_at(offset).is_none_or(pychar::is_space))
    }
}
