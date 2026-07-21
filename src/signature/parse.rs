//! dspy's string signature (`signatures/signature.py`): `"question -> answer"` declares a task
//! without declaring a type.
//!
//! Upstream reaches the field names through `ast.parse`, which is how it accepts any annotation
//! Python can spell. The structure around them is not Python-specific and is decided here: how
//! many arrows a signature may have, which names each side claims, what a name means when no
//! type follows it, and what the instructions read when nobody wrote any.
//!
//! An annotation this crate has no kind for travels as its own text, the same way a reflected
//! one does. Nothing is rejected for being unfamiliar: dspy prints the annotation it was given,
//! and so does this.

use anyhow::{Result, anyhow};

use super::{FieldKind, InField, OutField, Signature};
use crate::signature::field_type::JsonType;

/// dspy `Signature("question -> answer")`, and the typed form
/// `"question: str, context: list[str] -> answer: int"`.
pub fn parse(signature: &str) -> Result<Signature> {
    let arrows = signature.matches("->").count();
    if arrows != 1 {
        return Err(anyhow!(
            "Invalid signature format: '{signature}', must contain exactly one '->'."
        ));
    }
    let (before, after) = signature.split_once("->").expect("one arrow, just counted");
    let inputs = declarations(before)?;
    let outputs = declarations(after)?;
    refuse_duplicates(&inputs, &outputs)?;

    Ok(Signature {
        instructions: default_instructions(&inputs, &outputs),
        inputs: inputs
            .iter()
            .map(|(name, kind)| InField {
                name: name.clone(),
                kind: kind.clone(),
                ..Default::default()
            })
            .collect(),
        outputs: outputs
            .iter()
            .map(|(name, kind)| OutField {
                name: name.clone(),
                kind: kind.clone(),
                ..Default::default()
            })
            .collect(),
    })
}

/// dspy's `_default_instructions`, which stands in for a docstring nobody wrote.
fn default_instructions(inputs: &[(String, FieldKind)], outputs: &[(String, FieldKind)]) -> String {
    let quoted = |fields: &[(String, FieldKind)]| {
        fields
            .iter()
            .map(|(name, _)| format!("`{name}`"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    format!(
        "Given the fields {}, produce the fields {}.",
        quoted(inputs),
        quoted(outputs)
    )
}

/// dspy sorts the names it found on both sides, so the message does not depend on which side
/// was walked first.
fn refuse_duplicates(
    inputs: &[(String, FieldKind)],
    outputs: &[(String, FieldKind)],
) -> Result<()> {
    let mut shared: Vec<&str> = inputs
        .iter()
        .filter(|(name, _)| outputs.iter().any(|(other, _)| other == name))
        .map(|(name, _)| name.as_str())
        .collect();
    shared.sort_unstable();
    shared.dedup();
    if shared.is_empty() {
        return Ok(());
    }
    Err(anyhow!(
        "Input and output fields must have distinct names, but found duplicates: '{}'.",
        shared.join(", ")
    ))
}

/// One side of the arrow: `name`, or `name: annotation`, separated by commas.
fn declarations(side: &str) -> Result<Vec<(String, FieldKind)>> {
    split_top_level(side, ',')
        .into_iter()
        .filter(|part| !part.trim().is_empty())
        .map(|part| declaration(part.trim()))
        .collect()
}

fn declaration(part: &str) -> Result<(String, FieldKind)> {
    let (name, annotation) = match part.split_once(':') {
        // An unannotated field is a string, which upstream notes it would rather be explicit
        // about and cannot be without breaking programs that leave the type off.
        None => (part.trim(), None),
        Some((name, annotation)) => (name.trim(), Some(annotation.trim())),
    };
    if name.is_empty() {
        return Err(anyhow!("Invalid signature format: a field has no name."));
    }
    Ok((name.to_owned(), kind_of(annotation)))
}

/// The scalar kinds name themselves; everything else travels as the annotation dspy would print.
fn kind_of(annotation: Option<&str>) -> FieldKind {
    match annotation {
        None | Some("str") => FieldKind::Str,
        Some("int") => FieldKind::Int,
        Some("float") => FieldKind::Float,
        Some("bool") => FieldKind::Bool,
        Some(other) => FieldKind::Json(JsonType {
            annotation: canonical(other),
            ..Default::default()
        }),
    }
}

/// The spelling Python prints for an annotation, which is not always the one it was given.
///
/// Upstream resolves an annotation into a `typing` object and prints that object afterwards, so
/// a PEP 604 union comes back as the older spelling: `int | None` reads `Optional[int]`, and
/// anything else reads `Union[...]` with `None` written `NoneType`. It applies at every depth,
/// because the members are resolved before the type around them is built.
fn canonical(annotation: &str) -> String {
    let annotation = annotation.trim();
    let members = split_top_level(annotation, '|');
    if members.len() > 1 {
        return canonical_union(&members);
    }
    let Some((head, rest)) = annotation.split_once('[') else {
        return annotation.to_owned();
    };
    let arguments: Vec<String> = split_top_level(rest.strip_suffix(']').unwrap_or(rest), ',')
        .iter()
        .map(|argument| canonical(argument))
        .collect();
    format!("{}[{}]", head.trim(), arguments.join(", "))
}

fn canonical_union(members: &[&str]) -> String {
    let members: Vec<String> = members.iter().map(|member| canonical(member)).collect();
    // Exactly two, one of them `None`, is the shape `Optional` exists to spell.
    if members.len() == 2 {
        if let Some(at) = members.iter().position(|member| member == "None") {
            return format!("Optional[{}]", members[1 - at]);
        }
    }
    let spelled: Vec<&str> = members
        .iter()
        .map(|member| if member == "None" { "NoneType" } else { member })
        .collect();
    format!("Union[{}]", spelled.join(", "))
}

/// Split on the separators outside every bracket: `dict[str, int]` is one field, not two, and
/// the `|` in `dict[str, int | None]` belongs to the dict's value rather than to the field.
fn split_top_level(text: &str, separator: char) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0usize;
    let mut start = 0;
    for (at, character) in text.char_indices() {
        match character {
            '[' | '(' | '{' => depth += 1,
            ']' | ')' | '}' => depth = depth.saturating_sub(1),
            _ if character == separator && depth == 0 => {
                parts.push(&text[start..at]);
                start = at + character.len_utf8();
            }
            _ => {}
        }
    }
    parts.push(&text[start..]);
    parts
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `Signature` carries no `Debug`, so `expect_err` is out of reach.
    fn unwrap_err(parsed: Result<Signature>, spelling: &str) -> String {
        match parsed {
            Err(error) => error.to_string(),
            Ok(_) => panic!("{spelling:?} should have been refused"),
        }
    }

    fn names(signature: &Signature) -> (Vec<&str>, Vec<&str>) {
        (
            signature.inputs.iter().map(|f| f.name.as_str()).collect(),
            signature.outputs.iter().map(|f| f.name.as_str()).collect(),
        )
    }

    #[test]
    fn reads_the_shape_of_an_untyped_signature() {
        let signature = parse("email -> sentiment").expect("parses");
        assert_eq!(names(&signature), (vec!["email"], vec!["sentiment"]));
        assert!(signature.inputs.iter().all(|f| f.kind == FieldKind::Str));
    }

    #[test]
    fn keeps_the_order_the_fields_were_written_in() {
        let signature = parse("question, context -> reasoning, answer").expect("parses");
        assert_eq!(
            names(&signature),
            (vec!["question", "context"], vec!["reasoning", "answer"])
        );
    }

    #[test]
    fn reads_the_scalar_annotations_by_name() {
        let signature = parse("a: int, b: float, c: bool -> d: str").expect("parses");
        let kinds: Vec<&FieldKind> = signature.inputs.iter().map(|f| &f.kind).collect();
        assert_eq!(
            kinds,
            vec![&FieldKind::Int, &FieldKind::Float, &FieldKind::Bool]
        );
        assert_eq!(signature.outputs[0].kind, FieldKind::Str);
    }

    /// A comma inside brackets separates type arguments, not fields.
    #[test]
    fn a_generic_annotation_stays_one_field() {
        let signature = parse("ctx: list[str], weights: dict[str, int] -> answer").expect("parses");
        assert_eq!(names(&signature), (vec!["ctx", "weights"], vec!["answer"]));
        assert_eq!(
            signature.inputs[1].kind,
            FieldKind::Json(JsonType {
                annotation: "dict[str, int]".to_owned(),
                ..Default::default()
            })
        );
    }

    #[test]
    fn writes_the_instructions_dspy_writes_when_nobody_wrote_any() {
        let signature = parse("question, context -> answer").expect("parses");
        assert_eq!(
            signature.instructions,
            "Given the fields `question`, `context`, produce the fields `answer`."
        );
    }

    #[test]
    fn refuses_a_signature_without_exactly_one_arrow() {
        for spelling in ["", "question", "a -> b -> c"] {
            let error = unwrap_err(parse(spelling), spelling);
            assert!(
                error.contains("must contain exactly one '->'"),
                "{spelling:?} gave {error}"
            );
        }
    }

    /// Every recorded signature string, against what dspy made of it.
    ///
    /// The annotations are compared as dspy *prints* them, not as they were written, which is
    /// what makes the union canonicalisation part of the comparison rather than an assumption.
    #[test]
    fn parses_the_signatures_dspy_parses() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/conformance/signature/signature.json");
        let text = std::fs::read_to_string(&path).expect("the signature golden is committed");
        let golden: serde_json::Value = serde_json::from_str(&text).expect("the golden parses");
        let cases = golden["parse"].as_array().expect("parse cases");
        assert!(!cases.is_empty(), "the golden records no signatures");

        for case in cases {
            let spelling = case["signature"].as_str().expect("a signature");
            if let Some(expected) = case["error"].as_str() {
                assert_eq!(unwrap_err(parse(spelling), spelling), expected);
                continue;
            }
            let signature = match parse(spelling) {
                Ok(signature) => signature,
                Err(error) => panic!("{spelling:?} should have parsed, got {error}"),
            };
            assert_eq!(
                signature.instructions,
                case["instructions"].as_str().expect("instructions"),
                "instructions for {spelling:?}"
            );
            let ours: Vec<(String, String)> = signature
                .inputs
                .iter()
                .map(|f| (f.name.clone(), f.annotation()))
                .chain(
                    signature
                        .outputs
                        .iter()
                        .map(|f| (f.name.clone(), f.annotation())),
                )
                .collect();
            let expected: Vec<(String, String)> = ["inputs", "outputs"]
                .iter()
                .flat_map(|side| case[side].as_array().expect("fields"))
                .map(|field| {
                    (
                        field["name"].as_str().expect("a name").to_owned(),
                        field["annotation"]
                            .as_str()
                            .expect("an annotation")
                            .to_owned(),
                    )
                })
                .collect();
            assert_eq!(ours, expected, "fields for {spelling:?}");
        }
    }

    #[test]
    fn refuses_a_name_claimed_by_both_sides() {
        let error = unwrap_err(parse("b, a -> a, b"), "b, a -> a, b");
        assert!(
            error.contains("found duplicates: 'a, b'"),
            "duplicates should be named in sorted order, got {error}"
        );
    }
}

#[cfg(test)]
mod macro_tests {
    use crate::signature;

    /// `predict!("subject -> haiku")` builds the module, the way dspy's `Predict(spelling)` does.
    #[test]
    fn the_string_form_builds_a_module() {
        let mut haiku = crate::predict!("subject -> haiku");
        let named: Vec<String> = crate::module::Module::named_predictors(&mut haiku)
            .into_iter()
            .map(|predictor| predictor.signature.outputs[0].name.clone())
            .collect();
        assert_eq!(named, ["haiku"]);
    }

    /// The spelling a caller writes, checked while this crate compiles.
    #[test]
    fn the_macro_reads_the_same_signature_the_parser_reads() {
        let built = signature!("subject -> haiku");
        assert_eq!(built.inputs[0].name, "subject");
        assert_eq!(built.outputs[0].name, "haiku");
        let parsed: crate::Signature = "subject -> haiku".parse().expect("parses");
        assert!(built == parsed, "the macro and the parser should agree");
    }
}
