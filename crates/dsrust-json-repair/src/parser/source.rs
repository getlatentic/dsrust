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
pub(crate) struct Source {
    cells: Cells,
    /// Reads so far against what a terminating parse can need — debug builds only, and here
    /// rather than on the parser because half a dozen scans read positions directly. When the
    /// counter lived in `get_char_at`, every one of them was a loop a mutation could stall into a
    /// silent two-minute timeout; storing it at the storage makes an uncounted read something a
    /// reviewer has to construct rather than something a refactor gets for free.
    #[cfg(debug_assertions)]
    reads: std::cell::Cell<u64>,
    #[cfg(debug_assertions)]
    budget: u64,
}

pub(crate) enum Cells {
    /// Every character is ASCII, so a byte holds a code point and positions are byte offsets.
    Ascii(Vec<u8>),
    /// The general case, one `char` per code point.
    Wide(Vec<char>),
}

impl Source {
    pub(crate) fn of(text: &str) -> Self {
        let cells = if text.is_ascii() {
            Cells::Ascii(text.as_bytes().to_vec())
        } else {
            Cells::Wide(text.chars().collect())
        };
        Source {
            // Every position may legitimately start one scan over the rest of the input — that is
            // what the lookahead cache exists to make cheap — so a terminating parse is bounded by
            // the square of the length, with a floor for short inputs whose fixed work exceeds it.
            // The floor is the tighter kind: at a megabyte it let a stalled cursor spin for the
            // full two-minute mutation timeout on a twenty-character input, because each of its
            // iterations did a string-accumulator's worth of work per counted read — the timeout
            // fired first and the failure read as a hang. No terminating parse of a short input
            // approaches sixty-four thousand reads.
            #[cfg(debug_assertions)]
            budget: {
                let length = cells_len(&cells) as u64;
                length.saturating_mul(length).saturating_add(1 << 16)
            },
            #[cfg(debug_assertions)]
            reads: std::cell::Cell::new(0),
            cells,
        }
    }

    pub(crate) fn len(&self) -> usize {
        cells_len(&self.cells)
    }

    /// The character at an absolute code-point position.
    #[inline]
    pub(crate) fn at(&self, position: usize) -> Option<char> {
        #[cfg(debug_assertions)]
        {
            let read = self.reads.get() + 1;
            self.reads.set(read);
            assert!(
                read <= self.budget,
                "{read} character reads over an input of {} — a scan is not advancing. \
                 This bound is not reachable by a parse that ends.",
                self.len()
            );
        }
        match &self.cells {
            Cells::Ascii(bytes) => bytes.get(position).map(|&byte| byte as char),
            Cells::Wide(chars) => chars.get(position).copied(),
        }
    }

    /// `text[start..end]` as an owned string, both ends already clamped by the caller.
    pub(crate) fn slice_string(&self, start: usize, end: usize) -> String {
        match &self.cells {
            Cells::Ascii(bytes) => {
                String::from_utf8(bytes[start..end].to_vec()).expect("ASCII is UTF-8")
            }
            Cells::Wide(chars) => chars[start..end].iter().collect(),
        }
    }

    /// The first `needle` at or past `start`, as an absolute position.
    pub(crate) fn find_from(&self, start: usize, needle: char) -> Option<usize> {
        match &self.cells {
            Cells::Ascii(bytes) => {
                if !needle.is_ascii() {
                    return None;
                }
                bytes[start..]
                    .iter()
                    .position(|&byte| byte == needle as u8)
                    .map(|offset| start + offset)
            }
            Cells::Wide(chars) => chars[start..]
                .iter()
                .position(|&ch| ch == needle)
                .map(|offset| start + offset),
        }
    }

    /// Insert one character at a code-point position — the duplicate-key splice, always ASCII.
    pub(crate) fn insert(&mut self, position: usize, ch: char) {
        match &mut self.cells {
            Cells::Ascii(bytes) => {
                debug_assert!(ch.is_ascii(), "splicing {ch:?} would widen an ASCII source");
                bytes.insert(position, ch as u8);
            }
            Cells::Wide(chars) => chars.insert(position, ch),
        }
    }

    /// `text[..start] + replacement + text[end..]` — the comment-strip rebuild, which may write
    /// any text back. A replacement past ASCII widens an `Ascii` source to `Wide`, since the
    /// storage is a claim about every character it holds.
    pub(crate) fn splice(&mut self, start: usize, end: usize, replacement: &str) {
        match &mut self.cells {
            Cells::Ascii(bytes) if replacement.is_ascii() => {
                bytes.splice(start..end, replacement.bytes());
            }
            Cells::Ascii(bytes) => {
                let mut rebuilt: Vec<char> =
                    bytes[..start].iter().map(|&byte| byte as char).collect();
                rebuilt.extend(replacement.chars());
                rebuilt.extend(bytes[end..].iter().map(|&byte| byte as char));
                self.cells = Cells::Wide(rebuilt);
            }
            Cells::Wide(chars) => {
                chars.splice(start..end, replacement.chars());
            }
        }
    }

    /// `text[start..end]` as the characters it holds, for the span a string keeps.
    pub(crate) fn chars_vec(&self, start: usize, end: usize) -> Vec<char> {
        match &self.cells {
            Cells::Ascii(bytes) => bytes[start..end].iter().map(|&byte| byte as char).collect(),
            Cells::Wide(chars) => chars[start..end].to_vec(),
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
        match &self.cells {
            Cells::Ascii(bytes) => {
                let tail = bytes.get(position..).unwrap_or_default();
                position
                    + tail
                        .iter()
                        .position(|&byte| !ASCII_SPACE[byte as usize])
                        .unwrap_or(tail.len())
            }
            Cells::Wide(chars) => {
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
    /// Which cell size this source holds, for the one caller that picks a strict scanner by it.
    pub(crate) fn cells(&self) -> &Cells {
        &self.cells
    }

    pub(crate) fn iter_from(&self, position: usize) -> impl Iterator<Item = char> + '_ {
        let (ascii, wide) = match &self.cells {
            Cells::Ascii(bytes) => (Some(bytes[position..].iter()), None),
            Cells::Wide(chars) => (None, Some(chars[position..].iter())),
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
    // `const` evaluation has no iterators to own the progress, so the bound is the literal the
    // condition names.
    // ast-grep-ignore: cursor-arithmetic-loop
    while byte < 128 {
        table[byte] = matches!(byte as u8, 9..=13 | 28..=31 | 32);
        byte += 1;
    }
    table
};

fn cells_len(cells: &Cells) -> usize {
    match cells {
        Cells::Ascii(bytes) => bytes.len(),
        Cells::Wide(chars) => chars.len(),
    }
}
