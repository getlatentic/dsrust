//! dspy `primitives/repl_types.py`: what a REPL session shows the model about itself.
//!
//! Three values, all of which reach a prompt: the variables the code can reach, one interaction,
//! and the run so far. Each renders through the [`Type`](crate::adapter::Type) seam — upstream
//! declares them as `dspy.Type`s whose `serialize_model` *is* their `format`, so a field holding
//! one carries the prose rather than a JSON dump of its fields.

use serde::{Deserialize, Serialize};

use crate::adapter::types::base::{Formatted, Type};

/// dspy's default cap on how much of an output reaches a prompt.
pub const MAX_OUTPUT_CHARS: usize = 10_000;

/// dspy's default cap on how much of a variable's value is previewed.
pub const PREVIEW_CHARS: usize = 1_000;

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
    /// A variable whose value is this text, previewed the way dspy previews it.
    ///
    /// The type name is the caller's: only the sandbox knows what the value became on its side, and
    /// a Rust caller states it rather than a Python `type()` guessing at it.
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

    pub fn with_desc(mut self, desc: impl Into<String>) -> Self {
        self.desc = desc.into();
        self
    }

    pub fn with_constraints(mut self, constraints: impl Into<String>) -> Self {
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
    let tail: String = characters[characters.len() - half..].iter().collect();
    format!("{head}...{tail}")
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
        lines.push(format!("Total length: {} characters", grouped(self.total_length)));
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
    pub fn new(reasoning: impl Into<String>, code: impl Into<String>, output: impl Into<String>) -> Self {
        Self { reasoning: reasoning.into(), code: code.into(), output: output.into() }
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
                let tail: String = characters[raw_len - half..].iter().collect();
                let omitted = raw_len - max_output_chars;
                format!("{head}\n\n... ({} characters omitted) ...\n\n{tail}", grouped(omitted))
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
        Self { entries: Vec::new(), max_output_chars: MAX_OUTPUT_CHARS }
    }
}

impl ReplHistory {
    pub fn new(max_output_chars: usize) -> Self {
        Self { entries: Vec::new(), max_output_chars }
    }

    /// dspy's `append` answers with a new history rather than mutating: the value is frozen, and a
    /// turn's history is a snapshot the prompt already rendered.
    pub fn append(&self, entry: ReplEntry) -> Self {
        let mut entries = self.entries.clone();
        entries.push(entry);
        Self { entries, max_output_chars: self.max_output_chars }
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
mod tests {
    use super::*;

    #[test]
    fn a_variable_states_its_name_type_length_and_preview() {
        let variable = ReplVariable::new("context", "str", "hello");
        let Formatted::Text(rendered) = variable.format() else {
            panic!("a variable renders as text");
        };
        assert_eq!(
            rendered,
            "Variable: `context` (access it in your code)\n\
             Type: str\n\
             Total length: 5 characters\n\
             Preview:\n```\nhello\n```"
        );
    }

    /// The description and constraints appear only where they were given, between the type and the
    /// length.
    #[test]
    fn a_variable_states_a_description_and_constraints_only_when_it_has_them() {
        let variable = ReplVariable::new("n", "int", "42").with_desc("how many").with_constraints("> 0");
        let Formatted::Text(rendered) = variable.format() else {
            panic!("a variable renders as text");
        };
        assert!(rendered.contains("Type: int\nDescription: how many\nConstraints: > 0\nTotal length:"));
    }

    /// Python's thousands separator reaches the prompt, so it is reproduced rather than left plain.
    #[test]
    fn lengths_carry_pythons_thousands_separator() {
        assert_eq!(grouped(0), "0");
        assert_eq!(grouped(999), "999");
        assert_eq!(grouped(1_000), "1,000");
        assert_eq!(grouped(12_345), "12,345");
        assert_eq!(grouped(1_234_567), "1,234,567");
    }

    /// A value past the budget is cut in the middle, both halves off the same budget.
    #[test]
    fn a_long_value_is_previewed_head_and_tail() {
        let value: String = std::iter::repeat_n('x', 10).collect();
        assert_eq!(preview(&value, 10), value, "at the budget nothing is cut");
        assert_eq!(preview("abcdefghij", 4), "ab...ij");
        // An odd budget loses a character to Python's floor division, rather than gaining one.
        assert_eq!(preview("abcdefghij", 5), "ab...ij");
    }

    /// The header states the *true* length even when the body was cut.
    #[test]
    fn an_output_keeps_its_true_length_in_the_header() {
        assert_eq!(ReplEntry::format_output("hi", 10), "Output (2 chars):\nhi");
        let long: String = std::iter::repeat_n('y', 20).collect();
        let formatted = ReplEntry::format_output(&long, 10);
        assert!(formatted.starts_with("Output (20 chars):\n"), "got: {formatted}");
        assert!(formatted.contains("\n\n... (10 characters omitted) ...\n\n"), "got: {formatted}");
    }

    /// An entry is numbered from one, and its reasoning line appears only when there is one.
    #[test]
    fn an_entry_is_numbered_from_one() {
        let entry = ReplEntry::new("", "print(1)", "1");
        assert_eq!(
            entry.format_at(0, MAX_OUTPUT_CHARS),
            "=== Step 1 ===\nCode:\n```python\nprint(1)\n```\nOutput (1 chars):\n1"
        );
        let reasoned = ReplEntry::new("look first", "print(1)", "1");
        assert!(reasoned.format_at(1, MAX_OUTPUT_CHARS).starts_with("=== Step 2 ===\nReasoning: look first\nCode:"));
    }

    /// An empty history says so rather than rendering blank, and appending answers with a new one.
    #[test]
    fn an_empty_history_says_so_and_appending_does_not_mutate() {
        let history = ReplHistory::default();
        assert_eq!(
            history.format(),
            Formatted::Text("You have not interacted with the REPL environment yet.".to_owned())
        );
        let appended = history.append(ReplEntry::new("", "print(1)", "1"));
        assert!(history.is_empty(), "the original is unchanged");
        assert_eq!(appended.len(), 1);
        let Formatted::Text(rendered) = appended.format() else {
            panic!("a history renders as text");
        };
        assert!(rendered.starts_with("=== Step 1 ==="));
    }
}
