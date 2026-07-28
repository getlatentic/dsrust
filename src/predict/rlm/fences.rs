//! dspy `_strip_code_fences`: the Python out of whatever the model wrote around it.

use anyhow::{Result, bail};

/// dspy `_PYTHON_FENCE_LANGS`: the language tags a fence may carry and still be Python. The empty
/// one is a bare ``` fence, which upstream reads as Python rather than refusing.
const PYTHON_FENCE_LANGS: [&str; 5] = ["python", "py", "python3", "py3", ""];

/// Five rules, in upstream's order. Text with no fence is its own code. Decorative fence pairs
/// wrapping the whole thing are peeled off in a loop. The first fence in what remains is the one
/// read, so prose before it is skipped. A tag that is not Python is an error rather than a guess —
/// it reaches the model as the next turn's output, which is how it learns to write Python. And a
/// fence whose opener has no newline after it is left alone entirely, since there is no body to
/// take.
pub(crate) fn strip_code_fences(code: &str) -> Result<String> {
    let code = code.trim();
    if !code.contains("```") {
        return Ok(code.to_owned());
    }

    // Peel decorative pairs: ```\n```python\n…\n```\n``` and deeper.
    let mut lines: Vec<&str> = code.lines().collect();
    while lines.len() >= 2 && lines[0].trim() == "```" && lines[lines.len() - 1].trim() == "```" {
        lines.remove(0);
        lines.pop();
    }
    let peeled = lines.join("\n");
    let code = peeled.trim();
    if !code.contains("```") {
        return Ok(code.to_owned());
    }

    let opened = code.find("```").expect("the fence just checked for") + 3;
    // No newline after the opener means no body, and upstream hands back what it was given.
    let Some((language_line, body)) = code[opened..].split_once('\n') else {
        return Ok(code.to_owned());
    };

    // The first word of the language line, lowercased; a line of only whitespace reads as bare.
    let language = language_line.split_whitespace().next().unwrap_or_default().to_lowercase();
    if !PYTHON_FENCE_LANGS.contains(&language.as_str()) {
        bail!("Expected Python code but got ```{language} fence. Write Python code, not {language}.");
    }

    Ok(match body.find("```") {
        // An unterminated fence keeps everything after the opener.
        None => body.trim().to_owned(),
        Some(closed) => body[..closed].trim().to_owned(),
    })
}

#[cfg(test)]
mod conformance {
    use super::*;

    /// Every shape dspy was given, to the same code or the same refusal.
    ///
    /// The golden is what upstream returned for inputs chosen at the edges of its five rules —
    /// where a reimplementation guesses rather than agrees.
    #[test]
    fn it_strips_the_fences_dspy_strips() {
        for case in super::super::golden()["strip_code_fences"].as_array().expect("cases") {
            let written = case["written"].as_str().expect("written");
            match case["error"].as_str() {
                None => {
                    let code = strip_code_fences(written).expect("parses");
                    assert_eq!(code, case["code"].as_str().expect("code"), "code for {written:?}");
                }
                Some(error) => {
                    let refused = strip_code_fences(written).expect_err("refuses");
                    assert_eq!(refused.to_string(), error, "error for {written:?}");
                }
            }
        }
    }
}
