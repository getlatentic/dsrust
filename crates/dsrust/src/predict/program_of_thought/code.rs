//! dspy `ProgramOfThought._parse_code`: the runnable code out of the field the model wrote.
//!
//! Its own module because it is a text problem rather than a loop problem — three cuts, a fence
//! matcher and an assignment matcher, each mirroring one regex — and because every one of them is
//! a place a hand-written matcher parts company with the regex it stands for. The golden beside
//! them (`generate_pot_fixture.py`) chooses its inputs there deliberately.

use serde_json::Value;

use crate::example::Example;

/// dspy `_parse_code`: the runnable code out of the field the model wrote, or why it is not
/// runnable.
///
/// Upstream cuts the field at the first `---` or blank-blank-line, prefers a fenced ```python
/// block if there is one, and — where the last line assigns a name — appends that name so the
/// value becomes the program's result.
pub(in crate::predict) fn parse_generated_code(written: &Example) -> (String, Option<String>) {
    let code = written
        .get("generated_code")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let code = code.split("---").next().unwrap_or_default();
    let code = code.split("\n\n\n").next().unwrap_or_default();
    let block = fenced_python(code).unwrap_or(code);
    if block.is_empty() {
        return (
            code.to_owned(),
            Some("Error: Empty code after parsing.".to_owned()),
        );
    }
    if !block.contains('\n') && block.matches('=').count() > 1 {
        return (
            code.to_owned(),
            Some("Error: Code format is not correct.".to_owned()),
        );
    }
    let lines: Vec<&str> = block.split('\n').collect();
    let mut block = block.to_owned();
    if let Some(assigned) = assigned_name(lines.last().unwrap_or(&"").trim())
        && lines.len() > 1
    {
        block.push('\n');
        block.push_str(assigned);
    }
    (block, None)
}

/// The body of the first ```` ```python ```` block, matching upstream's
/// `` ```python[ \n](.*?)[ \n]```? `` — one space or newline after the opener, the shortest body,
/// and a closing fence of two backticks or three.
fn fenced_python(code: &str) -> Option<&str> {
    let opened = code.find("```python")? + "```python".len();
    let rest = &code[opened..];
    if !rest.starts_with([' ', '\n']) {
        return None;
    }
    let body = &rest[1..];
    // The shortest body: the earliest separator that is followed by a closing fence.
    let mut at = 0;
    while at < body.len() {
        let next = body[at..].find([' ', '\n'])? + at;
        if body[next + 1..].starts_with("``") {
            return Some(&body[..next]);
        }
        at = next + 1;
    }
    None
}

/// The name a line assigns to, matching upstream's `^(\w+)\s*=`. A `==` is not an assignment, and
/// upstream's regex agrees: `\s*=` matches the first `=`, leaving the second unread.
fn assigned_name(line: &str) -> Option<&str> {
    let name_end = line.find(|c: char| !(c.is_alphanumeric() || c == '_'))?;
    if name_end == 0 {
        return None;
    }
    let (name, rest) = line.split_at(name_end);
    rest.trim_start().starts_with('=').then_some(name)
}
