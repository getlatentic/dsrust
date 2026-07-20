//! The record of an episode: the turns the agent took, and how they reach the model.
//!
//! dspy holds the trajectory as one flat dict of numbered fields, rendered into the prompt on
//! every turn and handed back to the caller at the end. Both renderings live beside the turns
//! they read.

use serde_json::{Value, json};

use crate::adapter::python_json::format_value;

/// One turn of the loop, kept so the next turn can read what already happened.
#[derive(Debug, Clone, PartialEq)]
pub struct Step {
    pub thought: String,
    pub tool: String,
    pub args: Value,
    pub observation: String,
}

/// What the agent did, in order. A failed tool call stays in the trajectory rather than being
/// dropped: the model needs to see the error to recover from it, which is the whole point of
/// interleaving observations with reasoning.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Trajectory {
    pub steps: Vec<Step>,
}

impl Trajectory {
    /// The trajectory as prompt text, one labelled block per field, matching how dspy renders
    /// it: `format_user_message_content` over a signature built from the trajectory's own
    /// keys, which joins the blocks with a blank line and strips the result.
    pub fn rendered(&self) -> String {
        let mut blocks = Vec::new();
        for (index, step) in self.steps.iter().enumerate() {
            blocks.push(format!("[[ ## thought_{index} ## ]]\n{}", step.thought));
            blocks.push(format!("[[ ## tool_name_{index} ## ]]\n{}", step.tool));
            blocks.push(format!(
                "[[ ## tool_args_{index} ## ]]\n{}",
                format_value(&step.args)
            ));
            blocks.push(format!(
                "[[ ## observation_{index} ## ]]\n{}",
                step.observation
            ));
        }
        blocks.join("\n\n").trim().to_owned()
    }

    /// The flat `thought_0`/`tool_name_0`/`tool_args_0`/`observation_0` map dspy accumulates
    /// during the loop and hands back beside the task's outputs.
    pub fn as_value(&self) -> Value {
        let mut fields = serde_json::Map::new();
        for (index, step) in self.steps.iter().enumerate() {
            fields.insert(format!("thought_{index}"), json!(step.thought));
            fields.insert(format!("tool_name_{index}"), json!(step.tool));
            fields.insert(format!("tool_args_{index}"), step.args.clone());
            fields.insert(format!("observation_{index}"), json!(step.observation));
        }
        Value::Object(fields)
    }

    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn the_trajectory_renders_each_step_as_labelled_blocks() {
        let trajectory = Trajectory {
            steps: vec![Step {
                thought: "I should look it up".to_owned(),
                tool: "get_weather".to_owned(),
                args: json!({ "city": "Tokyo" }),
                observation: "sunny".to_owned(),
            }],
        };
        assert_eq!(
            trajectory.rendered(),
            "[[ ## thought_0 ## ]]\nI should look it up\n\n\
             [[ ## tool_name_0 ## ]]\nget_weather\n\n\
             [[ ## tool_args_0 ## ]]\n{\"city\": \"Tokyo\"}\n\n\
             [[ ## observation_0 ## ]]\nsunny"
        );
    }

    #[test]
    fn an_empty_trajectory_renders_as_nothing_at_all() {
        // dspy strips the joined blocks, so the first turn's trajectory field is empty rather
        // than a run of blank lines.
        assert_eq!(Trajectory::default().rendered(), "");
    }

    #[test]
    fn tool_arguments_render_with_pythons_json_spacing() {
        // dspy formats the argument object with `json.dumps`, which puts a space after every
        // colon and comma; serde_json's own `Display` puts neither.
        let args_block = |args| {
            Trajectory {
                steps: vec![Step {
                    thought: String::new(),
                    tool: "get_weather".to_owned(),
                    args,
                    observation: String::new(),
                }],
            }
            .rendered()
        };
        let spaced = args_block(json!({ "city": "Tokyo", "days": [1, 2] }));
        assert!(
            spaced.contains("[[ ## tool_args_0 ## ]]\n{\"city\": \"Tokyo\", \"days\": [1, 2]}"),
            "got: {spaced}"
        );
        assert!(args_block(json!({})).contains("[[ ## tool_args_0 ## ]]\n{}"));
    }
}
