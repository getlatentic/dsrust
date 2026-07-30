//! `interpreter/sandbox.rs` against dspy's own `build_repl_variable`.
//!
//! Upstream's `test_sandbox_serializable.py` runs green in the bridge and is declared not-crossing,
//! honestly: every test there asserts on the Python ABC — that an incomplete subclass cannot be
//! instantiated, that a duck-typed class fails `isinstance`, that the pydantic hook passes through.
//! A Rust trait has no runtime `isinstance` and no pydantic, so none of it reaches this crate.
//!
//! What does reach it is the behaviour, and this holds that to the oracle:
//! `scripts/generate_sandbox_serializable_fixture.py` runs dspy over four subclasses and records
//! every field of the `REPLVariable` it builds.

use std::path::Path;

use dsrust::interpreter::{SandboxSerializable, with_constraints};
use serde_json::Value;

/// One golden case, replayed through the Rust trait.
struct Recorded {
    setup: String,
    assignment: String,
    preview: String,
    type_name: String,
}

impl SandboxSerializable for Recorded {
    fn sandbox_setup(&self) -> String {
        self.setup.clone()
    }

    fn to_sandbox(&self) -> Vec<u8> {
        Vec::new()
    }

    /// The golden recorded `sandbox_assignment(name, "_raw_<name>")`, so replaying it means
    /// answering with that same text rather than reconstructing Python from a template.
    fn sandbox_assignment(&self, _var_name: &str, _data_expr: &str) -> String {
        self.assignment.clone()
    }

    fn rlm_preview(&self, _max_chars: usize) -> String {
        self.preview.clone()
    }

    fn type_name(&self) -> &str {
        &self.type_name
    }
}

/// Every field dspy's `build_repl_variable` decided, decided the same way here.
#[test]
fn build_repl_variable_matches_dspys() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/conformance/interpreter/sandbox_serializable.json");
    let golden: Value =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("the golden is committed"))
            .expect("the golden parses");
    let cases = golden["cases"].as_array().expect("cases");
    assert!(!cases.is_empty(), "the golden records no cases");

    for case in cases {
        let label = case["label"].as_str().expect("a label");
        let value = Recorded {
            setup: case["sandbox_setup"].as_str().expect("setup").to_owned(),
            assignment: case["sandbox_assignment"]
                .as_str()
                .expect("assignment")
                .to_owned(),
            preview: case["rlm_preview"].as_str().expect("preview").to_owned(),
            type_name: case["type_name"].as_str().expect("a type name").to_owned(),
        };
        // The golden's `desc` is what dspy *produced*, so the input is what it was given: the
        // `${...}` placeholder for the case that tests dropping one, and the leading line of the
        // produced text otherwise.
        let produced = case["desc"].as_str().expect("a desc");
        let given = match label.contains("placeholder") {
            true => format!("${{{}}}", case["name"].as_str().expect("a name")),
            false => produced
                .strip_suffix(&format!(
                    "Sandbox imports available:\n{}",
                    value.setup.trim()
                ))
                .unwrap_or(produced)
                .trim_end_matches('\n')
                .to_owned(),
        };
        let constraints = case["constraints"].as_str().expect("constraints");

        let built = with_constraints(
            &value,
            case["name"].as_str().expect("a name"),
            &given,
            constraints,
        );

        assert_eq!(
            built.name,
            case["name"].as_str().expect("a name"),
            "name, {label}"
        );
        assert_eq!(built.type_name, value.type_name, "type_name, {label}");
        assert_eq!(built.desc, produced, "desc, {label}");
        assert_eq!(built.constraints, constraints, "constraints, {label}");
        assert_eq!(built.preview, value.preview, "preview, {label}");
        assert_eq!(
            built.total_length,
            case["total_length"].as_u64().expect("a length") as usize,
            "total_length, {label}"
        );
    }
}
