//! Which keys a half-written JSON object has actually given a value to.
//!
//! dspy's JSON stream listener decides one field has ended by partial-parsing everything it has
//! accumulated and asking whether a *second* key appeared:
//!
//! ```text
//! parsed = jiter.from_json(accumulated, partial_mode="trailing-strings")
//! if len(parsed) > 1: ...   # the next field started, so ours is done
//! ```
//!
//! That predicate is the whole of what the listener needs, so this answers it directly rather than
//! reproducing a partial parser. [`json_repair`](dsrust_json_repair) cannot stand in: it
//! reproduces Python's `json_repair`, a different library, and the two disagree exactly where the
//! decision is made — `{"answer": "x", "judgement":` is one key to jiter and two to the repairer,
//! which would close the field a delta early.
//!
//! jiter's rule, read off 342 recorded prefixes rather than off its source: **a key counts once its
//! value has begun.** A string counts at its opening quote, a number at its first digit, an object
//! or array at its bracket. A *literal* is the exception — `nul` counts for nothing, because it is
//! not yet parseable as anything.

/// The keys of a half-written object that have had a value begin, in order.
///
/// `None` when the text is not an object at all, which is jiter raising rather than answering.
pub fn keys_with_values(text: &str) -> Option<Vec<String>> {
    let bytes = text.as_bytes();
    let mut at = skip_space(bytes, 0);
    if bytes.get(at) != Some(&b'{') {
        return None;
    }
    at = skip_space(bytes, at + 1);
    let mut keys = Vec::new();
    loop {
        match bytes.get(at) {
            None | Some(b'}') => return Some(keys),
            Some(b'"') => {}
            // Anything else where a key belongs is malformed, which jiter reports rather than
            // guessing at — and the listener treats as "no second key yet" either way.
            Some(_) => return Some(keys),
        }
        let (key, after) = string_at(bytes, at)?;
        at = skip_space(bytes, after);
        if bytes.get(at) != Some(&b':') {
            return Some(keys);
        }
        at = skip_space(bytes, at + 1);
        let Some(after_value) = value_at(bytes, at) else {
            // The value has not begun, so the key does not count yet. This is the whole
            // disagreement with `json_repair`, which fills it with an empty string.
            return Some(keys);
        };
        keys.push(key);
        at = skip_space(bytes, after_value);
        match bytes.get(at) {
            Some(b',') => at = skip_space(bytes, at + 1),
            _ => return Some(keys),
        }
    }
}

/// Whether the text is a complete JSON value — jiter parsing it with no partial mode at all, which
/// is how the listener knows the object closed rather than merely gained a key.
pub fn is_complete(text: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(text).is_ok()
}

fn skip_space(bytes: &[u8], mut at: usize) -> usize {
    while matches!(bytes.get(at), Some(b' ' | b'\t' | b'\n' | b'\r')) {
        at += 1;
    }
    at
}

/// The string starting at `at`, and where it ends. A string that runs off the end still yields what
/// it has — trailing-strings mode reads a half-written key as far as it goes.
fn string_at(bytes: &[u8], at: usize) -> Option<(String, usize)> {
    let mut out = String::new();
    let mut index = at + 1;
    while let Some(&byte) = bytes.get(index) {
        match byte {
            b'"' => return Some((out, index + 1)),
            b'\\' => {
                // Whatever the escape names, it cannot end the string, and the listener only ever
                // compares the key by name — so the byte after is taken as itself.
                index += 1;
                if let Some(&escaped) = bytes.get(index) {
                    out.push(escaped as char);
                    index += 1;
                }
            }
            _ => {
                out.push(byte as char);
                index += 1;
            }
        }
    }
    Some((out, index))
}

/// Where the value starting at `at` ends, or `None` if no value has begun there.
fn value_at(bytes: &[u8], at: usize) -> Option<usize> {
    match bytes.get(at)? {
        b'"' => string_at(bytes, at).map(|(_, end)| end),
        b'{' | b'[' => Some(balanced(bytes, at)),
        b'-' | b'0'..=b'9' => Some(number_end(bytes, at)),
        // A literal counts only once it is whole: `nul` is not `null`, and jiter reads it as
        // nothing rather than as a value in progress.
        byte => literal_end(bytes, at, *byte),
    }
}

fn literal_end(bytes: &[u8], at: usize, first: u8) -> Option<usize> {
    let word: &[u8] = match first {
        b't' => b"true",
        b'f' => b"false",
        b'n' => b"null",
        _ => return None,
    };
    bytes
        .get(at..at + word.len())
        .filter(|found| *found == word)
        .map(|_| at + word.len())
}

fn number_end(bytes: &[u8], at: usize) -> usize {
    let mut index = at + 1;
    while matches!(
        bytes.get(index),
        Some(b'0'..=b'9' | b'.' | b'e' | b'E' | b'+' | b'-')
    ) {
        index += 1;
    }
    index
}

/// Where the object or array starting at `at` closes, or the end of the text if it has not — a
/// brace inside a string does not close anything, which is why this cannot be a bracket count.
fn balanced(bytes: &[u8], at: usize) -> usize {
    let mut depth = 0usize;
    let mut index = at;
    while let Some(&byte) = bytes.get(index) {
        match byte {
            b'"' => {
                index = string_at(bytes, index).map_or(index + 1, |(_, end)| end);
                continue;
            }
            b'{' | b'[' => depth += 1,
            b'}' | b']' => {
                depth -= 1;
                if depth == 0 {
                    return index + 1;
                }
            }
            _ => {}
        }
        index += 1;
    }
    index
}
