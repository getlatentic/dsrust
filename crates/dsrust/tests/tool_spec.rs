//! Every tool `scripts/generate_tool_spec_fixture.py` recorded from dspy, re-created with `#[tool]`
//! and held to what dspy printed: the argument schema, the description, and the function spec a
//! provider that calls tools natively is sent.

use dsrust::adapter::native_tools::spec_of;
use dsrust::{Tool, tool};
use serde_json::{Map, Value, json};

fn fixture() -> Vec<Value> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/conformance/react/tool_spec.json"
    );
    let fixture: Value =
        serde_json::from_str(&std::fs::read_to_string(path).expect("committed")).expect("parses");
    fixture["tools"].as_array().expect("tools").clone()
}

/// Look one term up.
///
///     query is a phrase, not a sentence.
/// Answers with a summary.
///
#[tool]
fn indented_docstring(term: String) -> anyhow::Result<String> {
    Ok(term)
}

#[tool]
fn undocumented(x: i64) -> anyhow::Result<String> {
    Ok(x.to_string())
}

/// Look something up.
#[tool]
fn every_argument_required(query: String, limit: i64) -> anyhow::Result<String> {
    Ok(format!("{query}{limit}"))
}

/// Append one practice question and the answer the learner checks it against.
#[tool]
fn a_default_of_empty_string(
    prompt: String,
    answer: String,
    #[tool(default = "")] worked_solution: String,
) -> anyhow::Result<String> {
    Ok(format!("{prompt}{answer}{worked_solution}"))
}

/// Four arguments, each optional by default.
#[tool]
fn defaults_of_every_scalar(
    #[tool(default = 1)] a: i64,
    #[tool(default = false)] b: bool,
    #[tool(default = 0.5)] c: f64,
    #[tool(default = "x")] d: String,
) -> anyhow::Result<String> {
    Ok(format!("{a}{b}{c}{d}"))
}

// The recorded docstring has a blank line between its paragraphs and closes its quotes on a line
// of their own, so Python 3.13's `__doc__` — dedented at compile time — ends with a newline.
/// An argument may be optional by type or by default, and only one of those exempts it.
///
/// `by_type` has no default, so upstream requires it however nullable its schema is.
///
#[tool]
fn optional_by_type_and_by_default(
    by_type: Option<String>,
    #[tool(default = null)] by_default: Option<String>,
) -> anyhow::Result<String> {
    Ok(format!("{by_type:?}{by_default:?}"))
}

/// Read the whole draft as written so far.
#[tool]
fn no_arguments() -> anyhow::Result<String> {
    Ok("read".to_owned())
}

/// A mutable default is still a default.
#[tool]
fn a_container_with_a_default(
    names: Vec<String>,
    #[tool(default = {})] tags: Map<String, Value>,
) -> anyhow::Result<String> {
    Ok(format!("{names:?}{tags:?}"))
}

fn ours() -> Vec<Box<dyn Tool>> {
    vec![
        Box::new(IndentedDocstring),
        Box::new(Undocumented),
        Box::new(EveryArgumentRequired),
        Box::new(ADefaultOfEmptyString),
        Box::new(DefaultsOfEveryScalar),
        Box::new(OptionalByTypeAndByDefault),
        Box::new(NoArguments),
        Box::new(AContainerWithADefault),
    ]
}

#[test]
fn every_recorded_tool_is_re_created_here() {
    let recorded: Vec<&str> = fixture()
        .iter()
        .map(|t| t["name"].as_str().unwrap().to_owned())
        .collect::<Vec<_>>()
        .leak()
        .iter()
        .map(String::as_str)
        .collect();
    let mut here: Vec<String> = ours().iter().map(|t| t.name().to_owned()).collect();
    let mut theirs: Vec<String> = recorded.iter().map(|s| (*s).to_owned()).collect();
    here.sort();
    theirs.sort();
    assert_eq!(
        here, theirs,
        "the fixture and this file name the same tools"
    );
}

#[test]
fn the_argument_schema_is_the_one_dspy_prints() {
    for entry in fixture() {
        let name = entry["name"].as_str().unwrap();
        let tool = ours()
            .into_iter()
            .find(|t| t.name() == name)
            .unwrap_or_else(|| panic!("{name} has no counterpart"));
        assert_eq!(
            serde_json::to_string(tool.args()).unwrap(),
            serde_json::to_string(&entry["args"]).unwrap(),
            "{name}: args"
        );
    }
}

#[test]
fn the_description_is_dspys_including_none() {
    for entry in fixture() {
        let name = entry["name"].as_str().unwrap();
        let tool = ours().into_iter().find(|t| t.name() == name).unwrap();
        let theirs = entry["desc"].as_str().unwrap_or("");
        assert_eq!(tool.description(), theirs, "{name}: description");
    }
}

#[test]
fn the_native_function_spec_is_dspys() {
    for entry in fixture() {
        let name = entry["name"].as_str().unwrap();
        let tool = ours().into_iter().find(|t| t.name() == name).unwrap();
        assert_eq!(
            spec_of(tool.as_ref()).to_openai(),
            entry["native"],
            "{name}: native spec"
        );
    }
    let _ = json!(null);
}
