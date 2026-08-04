//! CPython's `str` predicates, which are not Rust's.
//!
//! `json_repair` decides whether a run of text is a bare object key, a number or prose by asking
//! `char.isalnum()` and friends, so a reply in Chinese takes a different branch from a reply in
//! English. Rust's own predicates answer differently in both directions: `char::is_whitespace`
//! follows White_Space and so refuses `\x1c`-`\x1f`, which CPython calls spaces, while
//! `char::is_alphabetic` follows Alphabetic and so accepts combining marks and Nl, which
//! `str.isalpha()` refuses.
//!
//! The tables are generated from CPython itself rather than taken from a Rust crate's idea of
//! Unicode, whose version moves independently: `unicode-general-category` 1.1.0 is Unicode 16.0
//! where CPython 3.13 is 15.1, which is several thousand code points of disagreement.

/// The ranges each predicate accepts, generated from CPython.
const DATA: &str = include_str!("pychar_data.txt");

/// Inclusive code point ranges, in ascending order, for one predicate.
struct Class(Vec<(u32, u32)>);

impl Class {
    /// The range holding `code_point`, if any.
    fn containing(&self, code_point: u32) -> Option<(u32, u32)> {
        self.0
            .binary_search_by(|&(lo, hi)| {
                if code_point < lo {
                    std::cmp::Ordering::Greater
                } else if code_point > hi {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Equal
                }
            })
            .ok()
            .map(|at| self.0[at])
    }

    fn accepts(&self, ch: char) -> bool {
        let code_point = ch as u32;
        self.0
            .binary_search_by(|&(lo, hi)| {
                if code_point < lo {
                    std::cmp::Ordering::Greater
                } else if code_point > hi {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Equal
                }
            })
            .is_ok()
    }
}

/// The four tables, parsed once.
struct Classes {
    space: Class,
    alpha: Class,
    digit: Class,
    alnum: Class,
    decimal: Class,
}

fn parse(name: &str) -> Class {
    let line = DATA
        .lines()
        .find(|line| line.split_whitespace().next() == Some(name))
        .unwrap_or_else(|| panic!("pychar_data.txt has no `{name}` line"));
    Class(
        line.split_whitespace()
            .skip(1)
            .map(|pair| {
                let (lo, hi) = pair.split_once('-').expect("a lo-hi range");
                (
                    u32::from_str_radix(lo, 16).expect("a hex code point"),
                    u32::from_str_radix(hi, 16).expect("a hex code point"),
                )
            })
            .collect(),
    )
}

fn classes() -> &'static Classes {
    static CLASSES: std::sync::OnceLock<Classes> = std::sync::OnceLock::new();
    CLASSES.get_or_init(|| Classes {
        space: parse("space"),
        alpha: parse("alpha"),
        digit: parse("digit"),
        alnum: parse("alnum"),
        decimal: parse("decimal"),
    })
}

/// CPython's `str.isspace()` for one character.
pub(crate) fn is_space(ch: char) -> bool {
    classes().space.accepts(ch)
}

/// CPython's `str.isalpha()` for one character.
pub(crate) fn is_alpha(ch: char) -> bool {
    classes().alpha.accepts(ch)
}

/// CPython's `str.isdigit()` for one character.
pub(crate) fn is_digit(ch: char) -> bool {
    classes().digit.accepts(ch)
}

/// CPython's `str.isalnum()` for one character.
///
/// Public because Python's `\w` is defined as this plus `_`, and a caller reproducing one of
/// dspy's regexes needs the same answer CPython gives rather than `char::is_alphanumeric`, which
/// accepts the combining marks `Alphabetic` carries and `str.isalnum()` refuses.
pub fn is_alnum(ch: char) -> bool {
    classes().alnum.accepts(ch)
}

/// CPython's `str.isdecimal()` for one character: category Nd, and the only digits `int()` and
/// `float()` accept. Wider than nothing and narrower than [`is_digit`], which takes `²`.
pub(crate) fn is_decimal(ch: char) -> bool {
    classes().decimal.accepts(ch)
}

/// The value of a decimal digit, `0` to `9`.
///
/// Every run in the table is a whole number of aligned `0`-`9` blocks — the generator refuses one
/// that is not — so a digit's value is its offset within its block.
pub(crate) fn decimal_value(ch: char) -> Option<u32> {
    let code_point = ch as u32;
    let (lo, _) = classes().decimal.containing(code_point)?;
    Some((code_point - lo) % 10)
}

/// `str.lower()` where the answer is a single character, or `None` where it widens or stays
/// beyond one — the allocation-free form of [`lower`] for callers comparing against ASCII words.
///
/// A character whose lowering is exactly one `char` compares as that `char`; `'İ'`, whose lowering
/// is two code points, answers `None` and can never equal an ASCII letter anyway. Built on the
/// same `char::to_lowercase` as [`lower`], so the two cannot disagree on what a character lowers
/// to — only on how the answer is spelled.
pub(crate) fn lowered_single(ch: char) -> Option<char> {
    let mut lowered = ch.to_lowercase();
    let first = lowered.next();
    match lowered.next() {
        None => first,
        Some(_) => None,
    }
}

/// CPython's `str.lower()` for one character, which may widen — `'İ'.lower()` is two code points,
/// so the caller compares strings rather than characters.
pub(crate) fn lower(ch: char) -> String {
    ch.to_lowercase().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The code points `predicate` accepts, as the inclusive ranges the fixture records.
    fn ranges(limit: u32, predicate: impl Fn(char) -> bool) -> Vec<(u32, u32)> {
        let mut found = Vec::new();
        let mut start: Option<u32> = None;
        for code_point in 0..limit {
            // A surrogate is a code point CPython answers for and Rust has no `char` for. It is in
            // none of these classes on either side, so rejecting it keeps the ranges aligned.
            let accepted = char::from_u32(code_point).is_some_and(&predicate);
            match (accepted, start) {
                (true, None) => start = Some(code_point),
                (false, Some(from)) => {
                    found.push((from, code_point - 1));
                    start = None;
                }
                _ => {}
            }
        }
        if let Some(from) = start {
            found.push((from, limit - 1));
        }
        found
    }

    #[test]
    fn each_class_matches_cpython_across_every_code_point() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/conformance/python_char_classes.json");
        let text = std::fs::read_to_string(&path).unwrap_or_else(|error| {
            panic!(
                "{}: {error} — run scripts/generate_python_char_tables.py",
                path.display()
            )
        });
        let fixture: serde_json::Value = serde_json::from_str(&text).expect("the fixture is JSON");
        let limit = fixture["limit"].as_u64().expect("a limit") as u32;
        let classes = fixture["classes"].as_object().expect("the classes");

        for (name, recorded) in classes {
            let expected: Vec<(u32, u32)> = recorded
                .as_array()
                .expect("ranges")
                .iter()
                .map(|pair| {
                    let pair = pair.as_array().expect("a pair");
                    (
                        pair[0].as_u64().expect("lo") as u32,
                        pair[1].as_u64().expect("hi") as u32,
                    )
                })
                .collect();
            let predicate: fn(char) -> bool = match name.as_str() {
                "space" => is_space,
                "alpha" => is_alpha,
                "digit" => is_digit,
                "alnum" => is_alnum,
                "decimal" => is_decimal,
                other => panic!("the fixture records a class this module does not have: {other}"),
            };
            let ours = ranges(limit, predicate);
            let first_difference = ours
                .iter()
                .zip(&expected)
                .find(|(left, right)| left != right);
            assert_eq!(
                ours,
                expected,
                "{name}: {} ranges against CPython\'s {}, first difference {first_difference:?}",
                ours.len(),
                expected.len(),
            );
        }
        assert_eq!(
            classes.len(),
            5,
            "the fixture stopped covering one of the five classes"
        );
    }

    #[test]
    fn the_classes_rust_disagrees_with_follow_cpython() {
        // Each of these is a code point where Rust's own predicate answers the other way, which is
        // the whole reason this module exists rather than a call to `char::is_whitespace`.
        assert!(
            is_space('\u{1c}'),
            "CPython calls the file separator a space"
        );
        assert!(
            !'\u{1c}'.is_whitespace(),
            "Rust does not, or this test proves nothing"
        );

        assert!(
            !is_alpha('\u{345}'),
            "CPython refuses the combining ypogegrammeni"
        );
        assert!(
            '\u{345}'.is_alphabetic(),
            "Rust accepts it, or this test proves nothing"
        );

        assert!(
            !is_alpha('\u{2160}'),
            "CPython refuses Roman numeral one: it is Nl, not a letter"
        );
        assert!(is_alnum('\u{2160}'), "but it is alphanumeric");

        assert!(is_digit('²'), "superscript two has Numeric_Type=Digit");
        assert!(!is_digit('½'), "a vulgar fraction does not");
        assert!(is_alnum('½'), "though it is still alphanumeric");
    }

    #[test]
    fn a_zero_width_space_is_not_a_space() {
        assert!(is_space('\u{a0}'));
        assert!(!is_space('\u{200b}'));
    }
}
