//! dspy `primitives/sandbox_serializable.py`: a value that reaches the REPL as its own
//! reconstruction rather than as JSON.
//!
//! The other half of the bring-your-own-sandbox story. [`CodeInterpreter`](super::CodeInterpreter)
//! says *where* code runs; this says how a value the caller holds gets *into* it — a dataframe as
//! parquet bytes and a `pd.read_parquet` line, rather than a million characters of JSON in a
//! prompt. The model is shown a short description and reaches the real thing by name.

use serde_json::{Map, Value, json};

use super::repl::ReplVariable;

/// dspy's default cap on the description a sandbox value contributes.
pub const PREVIEW_MAX_CHARS: usize = 500;

/// A value that enters the sandbox as code that rebuilds it.
///
/// The four methods are upstream's, and each answers a different question: what has to be in scope
/// first, what the bytes are, how to turn them back into the value, and what the model should be
/// told it has.
pub trait SandboxSerializable: Send + Sync {
    /// Python the sandbox runs once before the assignment — imports, normally.
    ///
    /// It is also shown to the model in the variable's description, so it knows which names are in
    /// scope. Nothing is a valid answer, and then nothing is added to the description.
    fn sandbox_setup(&self) -> String {
        String::new()
    }

    /// The value as bytes. Text crosses as itself; anything else is base64'd on the way in, so a
    /// binary format needs no special handling here.
    fn to_sandbox(&self) -> Vec<u8>;

    /// Python that binds `var_name` from `data_expr`, which names the variable the bytes arrived in.
    fn sandbox_assignment(&self, var_name: &str, data_expr: &str) -> String;

    /// A short description for the model — the shape of a frame, the length of a corpus. This
    /// stands in for the preview a JSON value would get, and is what its reported length counts.
    fn rlm_preview(&self, max_chars: usize) -> String;

    /// What the sandbox would call this value's type, which the model is shown.
    ///
    /// Python reads `type(value).__name__`; Rust cannot behind a trait object, so the holder says
    /// it — the same reason [`ReplVariable::new`] takes one.
    fn type_name(&self) -> &str;
}

/// dspy `build_repl_variable`: what the model is told about a value living in the sandbox.
///
/// Unlike [`ReplVariable::from_value`], the preview is the value's own description rather than a
/// slice of its text, and the reported length is that description's — the model is not being told
/// how big the value is, which it cannot be shown anyway, but what it is.
pub fn build_repl_variable(
    value: &dyn SandboxSerializable,
    name: &str,
    desc: &str,
) -> ReplVariable {
    with_constraints(value, name, desc, "")
}

/// As [`build_repl_variable`], with the field's constraint text as well.
///
/// Upstream reads both out of one `json_schema_extra`, so a field stating a constraint carries it
/// into the sandbox description. Rust has no `FieldInfo` to read, so the caller passes it.
pub fn with_constraints(
    value: &dyn SandboxSerializable,
    name: &str,
    desc: &str,
    constraints: &str,
) -> ReplVariable {
    let preview = value.rlm_preview(PREVIEW_MAX_CHARS);
    let setup = value.sandbox_setup().trim().to_owned();
    ReplVariable {
        name: name.to_owned(),
        type_name: value.type_name().to_owned(),
        desc: described(stated(desc), &setup),
        constraints: constraints.to_owned(),
        // Python's `len` over a `str` counts code points, which is what `chars` counts.
        total_length: preview.chars().count(),
        preview,
    }
}

/// A description someone actually wrote, or nothing.
///
/// dspy fills an unstated field description with the placeholder `${name}` and then drops anything
/// starting `${` on the way into a variable. Passing one through would show the model a literal
/// `${answer}` as though it were documentation.
fn stated(desc: &str) -> &str {
    match desc.starts_with("${") {
        true => "",
        false => desc,
    }
}

/// The field's own description with the setup appended under upstream's heading, either alone.
fn described(desc: &str, setup: &str) -> String {
    if setup.is_empty() {
        return desc.to_owned();
    }
    let note = format!("Sandbox imports available:\n{setup}");
    match desc.is_empty() {
        true => note,
        false => format!("{desc}\n{note}"),
    }
}

/// dspy `_prepare_serializable_vars`: the code that puts one value in the sandbox, and the
/// variables that code reads its bytes from.
///
/// Text crosses as itself. Bytes that are not UTF-8 cross base64'd, with the two lines upstream
/// prepends to decode them — so a parquet blob needs nothing from the caller but `to_sandbox`.
pub fn injection(value: &dyn SandboxSerializable, name: &str) -> (String, Map<String, Value>) {
    let raw = format!("_raw_{name}");
    let payload = value.to_sandbox();
    let mut lines = Vec::new();
    let mut variables = Map::new();

    match String::from_utf8(payload) {
        Ok(text) => {
            variables.insert(raw.clone(), json!(text));
        }
        Err(error) => {
            let encoded = format!("{raw}_base64");
            variables.insert(encoded.clone(), json!(base64(error.as_bytes())));
            lines.push("import base64".to_owned());
            lines.push(format!("{raw} = base64.b64decode({encoded})"));
        }
    }

    let setup = value.sandbox_setup();
    if !setup.is_empty() {
        lines.push(setup);
    }
    lines.push(value.sandbox_assignment(name, &raw));
    (lines.join("\n"), variables)
}

/// Standard base64, which is what Python's `b64encode` writes.
fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let packed = chunk
            .iter()
            .enumerate()
            .fold(0u32, |packed, (position, byte)| {
                packed | (u32::from(*byte) << (16 - 8 * position))
            });
        for slot in 0..4 {
            match slot <= chunk.len() {
                true => out.push(ALPHABET[(packed >> (18 - 6 * slot)) as usize & 0x3f] as char),
                false => out.push('='),
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::types::base::{Formatted, Type};

    /// A frame-shaped value, in the shape upstream's own docstring uses as the example.
    struct Frame {
        rows: usize,
        payload: Vec<u8>,
    }

    impl SandboxSerializable for Frame {
        fn sandbox_setup(&self) -> String {
            "import pandas as pd\nimport base64\nimport io".to_owned()
        }

        fn to_sandbox(&self) -> Vec<u8> {
            self.payload.clone()
        }

        fn sandbox_assignment(&self, var_name: &str, data_expr: &str) -> String {
            format!("{var_name} = pd.read_parquet(io.BytesIO(base64.b64decode({data_expr})))")
        }

        fn rlm_preview(&self, _max_chars: usize) -> String {
            format!("DataFrame: {} rows x 2 columns", self.rows)
        }

        fn type_name(&self) -> &str {
            "DataFrame"
        }
    }

    fn frame() -> Frame {
        Frame {
            rows: 3,
            payload: b"col_a,col_b".to_vec(),
        }
    }

    /// The preview is the value's own description, and the length counts *that* rather than the
    /// value — the model is told what it has, not how big it is.
    #[test]
    fn the_description_stands_in_for_the_preview_and_its_length() {
        let variable = build_repl_variable(&frame(), "sales", "");
        assert_eq!(variable.preview, "DataFrame: 3 rows x 2 columns");
        assert_eq!(
            variable.total_length,
            "DataFrame: 3 rows x 2 columns".chars().count()
        );
        assert_eq!(variable.type_name, "DataFrame");
    }

    /// The setup is shown to the model under upstream's heading, after the field's own description.
    #[test]
    fn the_setup_reaches_the_model_under_its_heading() {
        let alone = build_repl_variable(&frame(), "sales", "");
        assert_eq!(
            alone.desc,
            "Sandbox imports available:\nimport pandas as pd\nimport base64\nimport io"
        );
        let after = build_repl_variable(&frame(), "sales", "last quarter");
        assert!(
            after
                .desc
                .starts_with("last quarter\nSandbox imports available:\n")
        );
        // And it reaches the prompt, since the description is a line of the rendered variable.
        let Formatted::Text(rendered) = Type::format(&after) else {
            panic!("a variable renders as text");
        };
        assert!(rendered.contains("Description: last quarter\nSandbox imports available:"));
    }

    /// A value with no setup contributes no note, and no blank line either.
    #[test]
    fn no_setup_adds_nothing_to_the_description() {
        struct Bare;
        impl SandboxSerializable for Bare {
            fn to_sandbox(&self) -> Vec<u8> {
                b"x".to_vec()
            }
            fn sandbox_assignment(&self, var_name: &str, data_expr: &str) -> String {
                format!("{var_name} = {data_expr}")
            }
            fn rlm_preview(&self, _max_chars: usize) -> String {
                "one character".to_owned()
            }
            fn type_name(&self) -> &str {
                "str"
            }
        }
        assert_eq!(build_repl_variable(&Bare, "x", "").desc, "");
        assert_eq!(build_repl_variable(&Bare, "x", "a thing").desc, "a thing");
    }

    /// Text crosses as itself: the setup, then the assignment, reading one bound variable.
    #[test]
    fn text_bytes_cross_as_themselves() {
        let (code, variables) = injection(&frame(), "sales");
        assert_eq!(variables["_raw_sales"], json!("col_a,col_b"));
        assert_eq!(variables.len(), 1, "no base64 hop for text");
        assert_eq!(
            code,
            "import pandas as pd\nimport base64\nimport io\n\
             sales = pd.read_parquet(io.BytesIO(base64.b64decode(_raw_sales)))"
        );
    }

    /// Bytes that are not text cross base64'd, with the two lines that decode them prepended.
    #[test]
    fn binary_bytes_cross_base64_encoded() {
        let binary = Frame {
            rows: 1,
            payload: vec![0x50, 0x41, 0x52, 0xff, 0xfe],
        };
        let (code, variables) = injection(&binary, "sales");
        assert!(
            variables.contains_key("_raw_sales_base64"),
            "the encoded name is bound"
        );
        assert!(
            !variables.contains_key("_raw_sales"),
            "the raw name is built in the sandbox"
        );
        assert!(
            code.starts_with("import base64\n_raw_sales = base64.b64decode(_raw_sales_base64)\n")
        );
        assert!(
            code.ends_with("sales = pd.read_parquet(io.BytesIO(base64.b64decode(_raw_sales)))")
        );
    }

    /// The encoder is Python's, padding and all.
    #[test]
    fn the_encoder_writes_what_python_writes() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foo"), "Zm9v");
        assert_eq!(base64(b"foob"), "Zm9vYg==");
        assert_eq!(base64(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64(b"foobar"), "Zm9vYmFy");
        assert_eq!(base64(&[0xff, 0xfe, 0xfd]), "//79");
    }
}
