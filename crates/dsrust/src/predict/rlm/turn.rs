//! What crosses between one REPL turn and the next: the values going in, and the text coming out.
//!
//! Its own module because the loop's job is *deciding* — submit, feed back, extract — and these
//! two are the loop's vocabulary rather than its decisions. Both reach a prompt: the variable
//! descriptions are what the model is told it can touch, and a turn's output is what it reads back
//! before writing the next line.

use serde_json::Value;

use super::Rlm;
use crate::adapter::Type;
use crate::adapter::python_json::json_dumps;
use crate::adapter::types::base::Formatted;
use crate::example::Example;
use crate::interpreter::{ReplVariable, constraints};

impl Rlm {
    /// dspy `_build_variables`: what the model is told about each input it can reach.
    pub(super) fn variables(&self, inputs: &Example) -> Vec<String> {
        self.signature
            .inputs
            .iter()
            .filter_map(|field| {
                let stated = field.constraints.clone().unwrap_or_default();
                let variable = match self.sandboxed.get(&field.name) {
                    Some(held) => constraints(held.as_ref(), &field.name, &field.desc, &stated),
                    None => {
                        let mut built =
                            ReplVariable::from_value(&field.name, inputs.get(&field.name)?);
                        built.desc = field.desc.clone();
                        built.constraints = stated;
                        built
                    }
                };
                match Type::format(&variable) {
                    Formatted::Text(rendered) => Some(rendered),
                    Formatted::Blocks(_) => None,
                }
            })
            .collect()
    }
}

/// dspy `_format_output`: silence is reported as such, since a turn that printed nothing is
/// almost always a turn that forgot to.
pub(super) fn printed_output(printed: &Value) -> String {
    let output = match printed {
        Value::Null => String::new(),
        // dspy joins a list of output lines with newlines.
        Value::Array(lines) => lines
            .iter()
            .map(|line| match line {
                Value::String(text) => text.clone(),
                other => json_dumps(other),
            })
            .collect::<Vec<_>>()
            .join("\n"),
        Value::String(text) => text.clone(),
        other => json_dumps(other),
    };
    match output.is_empty() {
        true => "(no output - did you forget to print?)".to_owned(),
        false => output,
    }
}

pub(super) fn string_field(example: &Example, name: &str) -> String {
    example
        .get(name)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned()
}
