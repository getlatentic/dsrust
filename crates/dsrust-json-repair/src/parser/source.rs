//! The input, stored as whichever cell size the text actually needs.
//!
//! The parser thinks in code points because Python does: every index it keeps, every ±10 window
//! the repair log records, every negative wrap is a code-point number. `Vec<char>` honoured that
//! at four bytes per character — most of them zero, since the JSON a model emits is
//! overwhelmingly ASCII, where a code point *is* a byte. So the storage is chosen per input: an
//! all-ASCII text is kept as its bytes and indexed directly, and anything wider keeps the char
//! vector. Every position means the same thing in both, which is what keeps this a storage swap
//! rather than a re-derivation of 3,500 lines of index arithmetic.
//!
//! The one mutation the parser makes — splicing a `{` back into the input when a duplicate key
//! closes an object early — inserts an ASCII character, so it cannot move an `Ascii` input out of
//! its variant.

/// The characters of the input, at one byte each when the input allows it.
pub(crate) enum Source {
    /// Every character is ASCII, so a byte holds a code point and positions are byte offsets.
    Ascii(Vec<u8>),
    /// The general case, one `char` per code point.
    Wide(Vec<char>),
}

impl Source {
    pub(crate) fn of(text: &str) -> Self {
        if text.is_ascii() {
            Source::Ascii(text.as_bytes().to_vec())
        } else {
            Source::Wide(text.chars().collect())
        }
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            Source::Ascii(bytes) => bytes.len(),
            Source::Wide(chars) => chars.len(),
        }
    }

    /// The character at an absolute code-point position.
    #[inline]
    pub(crate) fn at(&self, position: usize) -> Option<char> {
        match self {
            Source::Ascii(bytes) => bytes.get(position).map(|&byte| byte as char),
            Source::Wide(chars) => chars.get(position).copied(),
        }
    }

    /// `text[start..end]` as an owned string, both ends already clamped by the caller.
    pub(crate) fn slice_string(&self, start: usize, end: usize) -> String {
        match self {
            Source::Ascii(bytes) => {
                String::from_utf8(bytes[start..end].to_vec()).expect("ASCII is UTF-8")
            }
            Source::Wide(chars) => chars[start..end].iter().collect(),
        }
    }

    /// The first `needle` at or past `start`, as an absolute position.
    pub(crate) fn find_from(&self, start: usize, needle: char) -> Option<usize> {
        match self {
            Source::Ascii(bytes) => {
                if !needle.is_ascii() {
                    return None;
                }
                bytes[start..]
                    .iter()
                    .position(|&byte| byte == needle as u8)
                    .map(|offset| start + offset)
            }
            Source::Wide(chars) => chars[start..]
                .iter()
                .position(|&ch| ch == needle)
                .map(|offset| start + offset),
        }
    }

    /// Insert one character at a code-point position — the duplicate-key splice, always ASCII.
    pub(crate) fn insert(&mut self, position: usize, ch: char) {
        match self {
            Source::Ascii(bytes) => {
                debug_assert!(ch.is_ascii(), "splicing {ch:?} would widen an ASCII source");
                bytes.insert(position, ch as u8);
            }
            Source::Wide(chars) => chars.insert(position, ch),
        }
    }

    /// `text[..start] + replacement + text[end..]` — the comment-strip rebuild, which may write
    /// any text back. A replacement past ASCII widens an `Ascii` source to `Wide`, since the
    /// storage is a claim about every character it holds.
    pub(crate) fn splice(&mut self, start: usize, end: usize, replacement: &str) {
        match self {
            Source::Ascii(bytes) if replacement.is_ascii() => {
                bytes.splice(start..end, replacement.bytes());
            }
            Source::Ascii(bytes) => {
                let mut rebuilt: Vec<char> =
                    bytes[..start].iter().map(|&byte| byte as char).collect();
                rebuilt.extend(replacement.chars());
                rebuilt.extend(bytes[end..].iter().map(|&byte| byte as char));
                *self = Source::Wide(rebuilt);
            }
            Source::Wide(chars) => {
                chars.splice(start..end, replacement.chars());
            }
        }
    }

    /// `text[start..end]` as the characters it holds, for the span a string keeps.
    pub(crate) fn chars_vec(&self, start: usize, end: usize) -> Vec<char> {
        match self {
            Source::Ascii(bytes) => bytes[start..end].iter().map(|&byte| byte as char).collect(),
            Source::Wide(chars) => chars[start..end].to_vec(),
        }
    }

    /// The end of the whitespace run starting at `position`, by CPython's `str.isspace()`.
    ///
    /// The `Ascii` arm walks bytes against a 128-entry table instead of taking the enum branch
    /// and the range lookup per character — the whitespace scrolls sit inside every member parse.
    pub(crate) fn scroll_spaces_from(&self, position: usize) -> usize {
        // Iterator-shaped rather than an index loop: there is no cursor mutation here for a
        // mutant to stall, so this scan cannot be made to hang — the failure the read counters
        // exist to catch simply has no site. `position` may sit past the end; an empty tail
        // scrolls nowhere.
        match self {
            Source::Ascii(bytes) => {
                let tail = bytes.get(position..).unwrap_or_default();
                position
                    + tail
                        .iter()
                        .position(|&byte| !ASCII_SPACE[byte as usize])
                        .unwrap_or(tail.len())
            }
            Source::Wide(chars) => {
                let tail = chars.get(position..).unwrap_or_default();
                position
                    + tail
                        .iter()
                        .position(|&ch| !crate::pychar::is_space(ch))
                        .unwrap_or(tail.len())
            }
        }
    }

    /// The characters from `position` on, for the callers that walk a tail.
    pub(crate) fn iter_from(&self, position: usize) -> impl Iterator<Item = char> + '_ {
        let (ascii, wide) = match self {
            Source::Ascii(bytes) => (Some(bytes[position..].iter()), None),
            Source::Wide(chars) => (None, Some(chars[position..].iter())),
        };
        ascii
            .into_iter()
            .flatten()
            .map(|&byte| byte as char)
            .chain(wide.into_iter().flatten().copied())
    }
}

/// `str.isspace()` restricted to ASCII: the CPython table's low 128 entries, precomputed.
const ASCII_SPACE: [bool; 128] = {
    let mut table = [false; 128];
    let mut byte = 0;
    while byte < 128 {
        table[byte] = matches!(byte as u8, 9..=13 | 28..=31 | 32);
        byte += 1;
    }
    table
};
