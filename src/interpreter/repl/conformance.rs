//! The REPL types held to what dspy's own rendered, rather than to a reading of its source.
//!
//! These bytes reach the prompt on every RLM iteration, and each carries a rule a reimplementation
//! gets wrong quietly: Python's thousands separator, a middle-out cut taken in code points off a
//! floor-divided budget, `str()` versus `json.dumps(indent=2)` for the value, `ensure_ascii`
//! escaping every non-ASCII character and the length that counts the escapes, and `s[-0:]` — which
//! is the whole string, not the empty tail.
//!
//! The golden is `tests/conformance/primitives/repl_types.json`; see
//! `scripts/generate_repl_types_fixture.py`.

use serde_json::Value;

use super::*;

fn golden() -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/conformance/primitives/repl_types.json");
    let text = std::fs::read_to_string(&path).expect("the golden is committed");
    serde_json::from_str(&text).expect("the golden parses")
}

fn text_of(formatted: Formatted) -> String {
    match formatted {
        Formatted::Text(text) => text,
        Formatted::Blocks(_) => panic!("a REPL type renders as text"),
    }
}

fn str_at<'a>(case: &'a Value, key: &str) -> &'a str {
    case[key]
        .as_str()
        .unwrap_or_else(|| panic!("{key} is a string"))
}

fn usize_at(case: &Value, key: &str) -> usize {
    case[key]
        .as_u64()
        .unwrap_or_else(|| panic!("{key} is a number")) as usize
}

/// Every value dspy was shown a variable for, to the same type, length, preview and prose.
#[test]
fn a_variable_describes_what_dspy_described() {
    for case in golden()["variables"].as_array().expect("cases") {
        let label = str_at(case, "label");
        let mut variable = ReplVariable::from_value_previewed(
            str_at(case, "name"),
            &case["value"],
            usize_at(case, "preview_chars"),
        );
        // dspy reads the description and constraints off the signature field; the crate's RLM
        // does the same, so the golden's are set here rather than re-derived.
        variable.desc = str_at(case, "desc").to_owned();
        variable.constraints = str_at(case, "constraints").to_owned();

        assert_eq!(
            variable.type_name,
            str_at(case, "type_name"),
            "type for {label}"
        );
        assert_eq!(
            variable.total_length,
            usize_at(case, "total_length"),
            "length for {label}"
        );
        assert_eq!(
            variable.preview,
            str_at(case, "preview"),
            "preview for {label}"
        );
        assert_eq!(
            text_of(variable.format()),
            str_at(case, "formatted"),
            "rendering for {label}"
        );
        // What a field holding one carries: dspy's `serialize_model` *is* its `format`.
        assert_eq!(
            text_of(Type::format(&variable)),
            str_at(case, "serialized"),
            "serialization for {label}"
        );
    }
}

/// Every output dspy formatted, to the same header and the same cut.
#[test]
fn an_output_is_cut_where_dspy_cut_it() {
    for case in golden()["outputs"].as_array().expect("cases") {
        let label = str_at(case, "label");
        assert_eq!(
            ReplEntry::format_output(str_at(case, "output"), usize_at(case, "max_output_chars")),
            str_at(case, "formatted"),
            "output for {label}"
        );
    }
}

/// Every entry dspy rendered, at the index it rendered it under.
#[test]
fn an_entry_renders_as_dspy_rendered_it() {
    for case in golden()["entries"].as_array().expect("cases") {
        let label = str_at(case, "label");
        let entry = ReplEntry::new(
            str_at(case, "reasoning"),
            str_at(case, "code"),
            str_at(case, "output"),
        );
        assert_eq!(
            entry.format_at(usize_at(case, "index"), usize_at(case, "max_output_chars")),
            str_at(case, "formatted"),
            "entry for {label}"
        );
    }
}

/// Every history dspy rendered, including the sentence an empty one stands in with.
#[test]
fn a_history_renders_as_dspy_rendered_it() {
    let golden = golden();
    let entries: Vec<&Value> = golden["entries"]
        .as_array()
        .expect("entries")
        .iter()
        .collect();
    let entry_of = |label: &str| {
        let case = entries
            .iter()
            .find(|case| str_at(case, "label") == label)
            .unwrap_or_else(|| panic!("no entry labelled {label}"));
        ReplEntry::new(
            str_at(case, "reasoning"),
            str_at(case, "code"),
            str_at(case, "output"),
        )
    };

    for case in golden["histories"].as_array().expect("cases") {
        let label = str_at(case, "label");
        let mut history = ReplHistory::new(usize_at(case, "max_output_chars"));
        for name in case["entries"].as_array().expect("entry labels") {
            history = history.append(entry_of(name.as_str().expect("a label")));
        }
        assert_eq!(history.len(), usize_at(case, "len"), "length for {label}");
        assert_eq!(
            !history.is_empty(),
            case["truthy"].as_bool().expect("truthy"),
            "truthiness for {label}"
        );
        assert_eq!(
            text_of(history.format()),
            str_at(case, "formatted"),
            "history for {label}"
        );
        assert_eq!(
            text_of(Type::format(&history)),
            str_at(case, "serialized"),
            "serialization for {label}"
        );
    }
}
