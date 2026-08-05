//! A backslash already in the accumulator, and the character after it.
//!
//! The rules run wider than JSON's: `\x41` is read, an even run of backslashes is halved, and a
//! delimiter that had no business being escaped loses its escape. What is *not* here is as
//! deliberate — an unrecognised escape is left alone and the backslash stays in the string.

use crate::STRING_DELIMITERS;
use crate::parser::Parser;
use crate::parser::string::StringParseState;

/// Whether the escape was handled, and the character the scan should look at next.
pub(crate) type Handled = (bool, Option<char>);

impl Parser {
    pub(crate) fn normalize_escape_sequence(
        &mut self,
        state: &mut StringParseState,
        char: char,
    ) -> Handled {
        self.log("Found a stray escape sequence, normalizing it");
        let active = state.active_rstring_delimiter();

        if state.in_low_smart_quote_span() && char == '"' {
            state.pop_acc();
            state.append(&[char]);
            state.rebuild_unmatched_opening_braces();
            state.pop_low_smart_quote_span();
            self.index += 1;
            return (true, self.char_here());
        }

        if char == '\\' && self.halve_backslash_run(state, active) {
            return (true, self.char_here());
        }

        if char == active || matches!(char, 't' | 'n' | 'r' | 'b' | '\\') {
            self.apply_short_escape(state, char, active);
            return (true, self.char_here());
        }

        if char == 'u' || char == 'x' {
            let width = if char == 'u' { 4 } else { 2 };
            let digits = self.slice(self.index as isize + 1, self.index as isize + 1 + width);
            if digits.chars().count() == width as usize
                && digits.chars().all(|digit| digit.is_ascii_hexdigit())
            {
                self.log("Found a unicode escape sequence, normalizing it");
                let code_point = u32::from_str_radix(&digits, 16).expect("four hex digits");
                state.pop_acc();
                state.append(&[char::from_u32(code_point).unwrap_or(crate::LONE_SURROGATE)]);
                state.rebuild_unmatched_opening_braces();
                self.index += 1 + width as usize;
                return (true, self.char_here());
            }
        } else if char == '„' || (STRING_DELIMITERS.contains(&char) && char != active) {
            self.log(
                "Found a delimiter that was escaped but shouldn't be escaped, removing the escape",
            );
            state.pop_acc();
            state.append(&[char]);
            state.rebuild_unmatched_opening_braces();
            self.index += 1;
            return (true, self.char_here());
        }

        (false, Some(char))
    }

    /// An even run of backslashes that does not end at the closing quote is halved: `\\\\` in the
    /// input is two backslashes in the value.
    fn halve_backslash_run(&mut self, state: &mut StringParseState, active: char) -> bool {
        let run_start = self.index - 1;
        let mut run_end = self.index + 1;
        while self.json_str.at(run_end) == Some('\\') {
            run_end += 1;
        }
        let run_length = run_end - run_start;
        let next_char = self.get_char_at(run_end as isize - self.index as isize);
        if !run_length.is_multiple_of(2) || next_char == Some(active) {
            return false;
        }
        state.pop_acc();
        let halved = vec!['\\'; run_length / 2];
        state.append(&halved);
        state.rebuild_unmatched_opening_braces();
        self.index = run_end;
        true
    }

    /// `\t`, `\n`, `\r`, `\b`, an escaped delimiter or an escaped backslash — and then any run of
    /// the same that the replacement exposed.
    fn apply_short_escape(&mut self, state: &mut StringParseState, char: char, active: char) {
        state.pop_acc();
        let replacement = match char {
            't' => '\t',
            'n' => '\n',
            'r' => '\r',
            'b' => '\u{8}',
            other => other,
        };
        state.append(&[replacement]);
        state.rebuild_unmatched_opening_braces();
        self.index += 1;

        while let Some(next_char) = self.char_here() {
            if state.last_acc() != Some('\\') || !(next_char == active || next_char == '\\') {
                break;
            }
            state.pop_acc();
            state.append(&[next_char]);
            state.rebuild_unmatched_opening_braces();
            self.index += 1;
        }
    }
}
