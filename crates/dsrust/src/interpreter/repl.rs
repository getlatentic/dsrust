//! dspy `primitives/repl_types.py`: what a REPL session shows the model about itself.
//!
//! Three values, all of which reach a prompt: the variables the code can reach, one interaction,
//! and the run so far. Each renders through the [`Type`] seam — upstream
//! declares them as `dspy.Type`s whose `serialize_model` *is* their `format`, so a field holding
//! one carries the prose rather than a JSON dump of its fields.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::adapter::python_json::{json_dumps_indented, python_type_name};
use crate::adapter::types::base::{Formatted, Type};

/// dspy's default cap on how much of an output reaches a prompt.
///
/// A cap rather than a truncation the caller applies: sandboxed code can print a megabyte, and the
/// next turn of the loop puts that output in front of a model. Named so a caller writing their own
/// interpreter can cut at the same place upstream cuts.
///
/// ```
/// assert_eq!(dsrust::interpreter::repl::MAX_OUTPUT_CHARS, 10_000);
/// ```
pub const MAX_OUTPUT_CHARS: usize = 10_000;

/// dspy's default cap on how much of a variable's value is previewed.
pub(crate) const PREVIEW_CHARS: usize = 1_000;

/// dspy `REPLVariable`: what the model is told about one value it can reach from its code.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ReplVariable {
    pub name: String,
    /// The name of the value's type as the sandbox would print it — Python's `type(value).__name__`.
    pub type_name: String,
    #[serde(default)]
    pub desc: String,
    #[serde(default)]
    pub constraints: String,
    /// How long the whole value is, which the model is told even when the preview is cut.
    pub total_length: usize,
    pub preview: String,
}

impl ReplVariable {
    /// dspy `REPLVariable.from_value`: what the model is told about a value it can reach.
    ///
    /// Two writers decide the text, and they are not the one the adapters use. A container goes
    /// through `json.dumps(indent=2)` — an entry per line, and non-ASCII escaped, since upstream
    /// leaves `ensure_ascii` at its default. Anything else is what Python's `str` prints, so a
    /// bool reads `True` and a null reads `None`. The length reported beside the preview counts
    /// that text, escapes and all.
    pub fn from_value(name: impl Into<String>, value: &Value) -> Self {
        Self::from_value_previewed(name, value, PREVIEW_CHARS)
    }

    /// As [`Self::from_value`], with dspy's `preview_chars` stated rather than defaulted.
    pub fn from_value_previewed(
        name: impl Into<String>,
        value: &Value,
        preview_chars: usize,
    ) -> Self {
        let text = stringified(value);
        Self {
            name: name.into(),
            type_name: python_type_name(value).to_owned(),
            desc: String::new(),
            constraints: String::new(),
            total_length: text.chars().count(),
            preview: preview(&text, preview_chars),
        }
    }

    /// A variable whose value is this text, previewed the way dspy previews it.
    ///
    /// The type name is the caller's, which is what a value living in the sandbox needs: only the
    /// sandbox knows what it became on its side, so upstream's `SandboxSerializable` hook has the
    /// holder state it. [`Self::from_value`] is the path for a value the caller still holds.
    pub fn new(name: impl Into<String>, type_name: impl Into<String>, value: &str) -> Self {
        Self {
            name: name.into(),
            type_name: type_name.into(),
            desc: String::new(),
            constraints: String::new(),
            total_length: value.chars().count(),
            preview: preview(value, PREVIEW_CHARS),
        }
    }

    pub fn desc(mut self, desc: impl Into<String>) -> Self {
        self.desc = desc.into();
        self
    }

    pub fn constraints(mut self, constraints: impl Into<String>) -> Self {
        self.constraints = constraints.into();
        self
    }
}

/// dspy's middle-out preview: over the budget, the halves either side of an ellipsis.
fn preview(value: &str, budget: usize) -> String {
    let characters: Vec<char> = value.chars().collect();
    if characters.len() <= budget {
        return value.to_owned();
    }
    // Python's `//`, and both halves are taken from the *same* budget, so an odd budget loses a
    // character rather than gaining one.
    let half = budget / 2;
    let head: String = characters[..half].iter().collect();
    format!("{head}...{}", tail(&characters, half))
}

/// Python's `value[-half:]`, which is not `value[len - half..]` at zero: `-0 == 0`, so a budget
/// with no half to spare hands back the *whole* value rather than nothing.
fn tail(characters: &[char], half: usize) -> String {
    match half {
        0 => characters.iter().collect(),
        _ => characters[characters.len() - half..].iter().collect(),
    }
}

/// dspy `REPLVariable.from_value`'s stringifier: `json.dumps(indent=2)` for a container, Python's
/// `str` for anything else.
fn stringified(value: &Value) -> String {
    match value {
        Value::Object(_) | Value::Array(_) => json_dumps_indented(value),
        Value::String(text) => text.clone(),
        Value::Null => "None".to_owned(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        number => number.to_string(),
    }
}

/// Python's `f"{n:,}"`: a thousands separator every three digits. It reaches the prompt, so the
/// grouping is part of the bytes rather than a display choice.
fn grouped(number: usize) -> String {
    let digits = number.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (position, digit) in digits.chars().enumerate() {
        if position > 0 && (digits.len() - position) % 3 == 0 {
            out.push(',');
        }
        out.push(digit);
    }
    out
}

impl Type for ReplVariable {
    /// dspy `REPLVariable.format`: the name and type always, the description and constraints only
    /// where they were given, then the length and the preview in a fence.
    fn format(&self) -> Formatted {
        let mut lines = vec![
            format!("Variable: `{}` (access it in your code)", self.name),
            format!("Type: {}", self.type_name),
        ];
        if !self.desc.is_empty() {
            lines.push(format!("Description: {}", self.desc));
        }
        if !self.constraints.is_empty() {
            lines.push(format!("Constraints: {}", self.constraints));
        }
        lines.push(format!(
            "Total length: {} characters",
            grouped(self.total_length)
        ));
        lines.push(format!("Preview:\n```\n{}\n```", self.preview));
        Formatted::Text(lines.join("\n"))
    }
}

/// dspy `REPLEntry`: one turn of the loop — what the model was thinking, what it ran, and what came
/// back.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ReplEntry {
    #[serde(default)]
    pub reasoning: String,
    pub code: String,
    pub output: String,
}

impl ReplEntry {
    pub fn new(
        reasoning: impl Into<String>,
        code: impl Into<String>,
        output: impl Into<String>,
    ) -> Self {
        Self {
            reasoning: reasoning.into(),
            code: code.into(),
            output: output.into(),
        }
    }

    /// dspy `REPLEntry.format_output`: the true length in the header, and a middle cut where the
    /// output ran past the cap — so the model is never misled about how much it is not seeing.
    pub fn format_output(output: &str, max_output_chars: usize) -> String {
        let characters: Vec<char> = output.chars().collect();
        let raw_len = characters.len();
        let shown = match raw_len > max_output_chars {
            false => output.to_owned(),
            true => {
                let half = max_output_chars / 2;
                let head: String = characters[..half].iter().collect();
                let omitted = raw_len - max_output_chars;
                format!(
                    "{head}\n\n... ({} characters omitted) ...\n\n{}",
                    grouped(omitted),
                    tail(&characters, half)
                )
            }
        };
        format!("Output ({} chars):\n{shown}", grouped(raw_len))
    }

    /// This entry as the model reads it back, numbered from one.
    pub fn format_at(&self, index: usize, max_output_chars: usize) -> String {
        let reasoning = match self.reasoning.is_empty() {
            true => String::new(),
            false => format!("Reasoning: {}\n", self.reasoning),
        };
        format!(
            "=== Step {} ===\n{reasoning}Code:\n```python\n{}\n```\n{}",
            index + 1,
            self.code,
            Self::format_output(&self.output, max_output_chars)
        )
    }
}

/// dspy `REPLHistory`: the run so far, as the next turn is shown it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplHistory {
    #[serde(default)]
    pub entries: Vec<ReplEntry>,
    pub max_output_chars: usize,
}

impl Default for ReplHistory {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            max_output_chars: MAX_OUTPUT_CHARS,
        }
    }
}

impl ReplHistory {
    pub fn new(max_output_chars: usize) -> Self {
        Self {
            entries: Vec::new(),
            max_output_chars,
        }
    }

    /// dspy's `append` answers with a new history rather than mutating: the value is frozen, and a
    /// turn's history is a snapshot the prompt already rendered.
    pub fn append(&self, entry: ReplEntry) -> Self {
        let mut entries = self.entries.clone();
        entries.push(entry);
        Self {
            entries,
            max_output_chars: self.max_output_chars,
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Type for ReplHistory {
    /// dspy `REPLHistory.format`: every entry in order, and a sentence rather than a blank where
    /// the run has not started.
    fn format(&self) -> Formatted {
        if self.entries.is_empty() {
            return Formatted::Text(
                "You have not interacted with the REPL environment yet.".to_owned(),
            );
        }
        Formatted::Text(
            self.entries
                .iter()
                .enumerate()
                .map(|(index, entry)| entry.format_at(index, self.max_output_chars))
                .collect::<Vec<_>>()
                .join("\n"),
        )
    }
}

#[cfg(test)]
mod conformance;

#[cfg(test)]
mod tests {
    use super::*;

    /// Appending answers with a new history rather than mutating — a Rust property, since dspy
    /// gets it from a frozen pydantic model.
    #[test]
    fn appending_does_not_mutate() {
        let history = ReplHistory::default();
        let appended = history.append(ReplEntry::new("", "print(1)", "1"));
        assert!(history.is_empty(), "the original is unchanged");
        assert_eq!(appended.len(), 1);
    }
}
