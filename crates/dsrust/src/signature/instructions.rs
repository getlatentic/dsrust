//! What a signature's instructions become: upstream's `inspect.cleandoc`, and the default.
//!
//! dspy stores no instructions field: `Signature.instructions` is a property that reads `__doc__`
//! and returns `inspect.cleandoc` of it, so the normalisation happens on every read and nothing a
//! caller sets escapes it. Text set through `with_instructions` is therefore not the text the
//! prompt carries whenever the two differ.
//!
//! A port that keeps the string it was handed agrees with upstream wherever the instructions are
//! rendered directly — an adapter re-indents them anyway — and diverges wherever anything
//! *composes* them further. `ReActV2` appends four lines to the signature's instructions, and a
//! trailing newline that upstream had already stripped becomes a blank line in the agent's prompt.

/// The instructions a signature carries once a caller has set them to `given`.
///
/// Empty text is not instructions with nothing in them: dspy keeps them in `__doc__`, where empty
/// is indistinguishable from unset, and a signature with an empty docstring states the default
/// objective. Any other text is cleaned.
pub fn stated(given: &str, inputs: &[&str], outputs: &[&str]) -> String {
    match given.is_empty() {
        true => super::default_instructions(inputs, outputs),
        false => cleandoc(given),
    }
}

/// Strip a docstring's common indentation and its leading and trailing blank lines.
///
/// CPython's algorithm, in its order: expand tabs, take the smallest indent of any non-blank line
/// *after the first*, strip the first line outright and that margin from the rest, then drop
/// wholly empty lines from each end. A line of spaces is not an empty line — it survives at the
/// end where a truly empty one is dropped.
fn cleandoc(text: &str) -> String {
    let expanded = expand_tabs(text);
    let mut lines: Vec<String> = expanded.split('\n').map(str::to_owned).collect();

    let margin = lines
        .iter()
        .skip(1)
        .filter(|line| !line.trim_start().is_empty())
        .map(|line| line.len() - line.trim_start().len())
        .min();

    if let Some(first) = lines.first_mut() {
        *first = first.trim_start().to_owned();
    }
    if let Some(margin) = margin {
        for line in lines.iter_mut().skip(1) {
            *line = line.chars().skip(margin).collect();
        }
    }
    while lines.last().is_some_and(String::is_empty) {
        lines.pop();
    }
    while lines.first().is_some_and(String::is_empty) {
        lines.remove(0);
    }
    lines.join("\n")
}

/// Python's `str.expandtabs()`: a tab advances to the next multiple of eight *columns*, which is
/// not the same as eight spaces, and the column count restarts at every line break.
fn expand_tabs(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut column = 0;
    for ch in text.chars() {
        match ch {
            '\t' => {
                let stop = 8 - (column % 8);
                out.extend(std::iter::repeat_n(' ', stop));
                column += stop;
            }
            '\n' | '\r' => {
                out.push(ch);
                column = 0;
            }
            _ => {
                out.push(ch);
                column += 1;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {

    use serde_json::Value;

    /// What dspy answers, recorded by running it. Regenerate with
    /// `scripts/generate_signature_fixture.py`.
    fn golden() -> Value {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/conformance/signature/signature.json");
        let text = std::fs::read_to_string(&path).expect("the signature golden is committed");
        serde_json::from_str(&text).expect("the golden parses")
    }

    /// Every string upstream was handed, against what it answered for it.
    ///
    /// Set through `Signature::with_instructions`, not through `cleandoc` alone: the empty string
    /// restores the default objective rather than cleaning to nothing, and that decision belongs
    /// to the same seam.
    #[test]
    fn normalises_instructions_as_dspy_does() {
        let cases = golden()["instructions"].as_array().expect("cases").clone();
        assert!(!cases.is_empty(), "the golden records no instructions");
        let signature: crate::signature::Signature = "q -> a".parse().expect("parses");
        for case in cases {
            let given = case["given"].as_str().expect("given");
            let want = case["instructions"].as_str().expect("instructions");
            assert_eq!(
                signature.with_instructions(given).instructions,
                want,
                "for {given:?}"
            );
        }
    }
}
