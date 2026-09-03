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
use serde_json::Value;

use super::{FieldKind, InField, LiteralValue, OutField, Signature};
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
    let inputs = collapsed(declarations(before, "input")?);
    let outputs = collapsed(declarations(after, "output")?);
    refuse_duplicates(&inputs, &outputs)?;

    Ok(Signature {
        instructions: declared_instructions(&inputs, &outputs),
        inputs: inputs
            .iter()
            .map(|declared| InField {
                name: declared.name.clone(),
                kind: declared.kind.clone(),
                values: declared.values.clone(),
                ..Default::default()
            })
            .collect(),
        outputs: outputs
            .iter()
            .map(|declared| OutField {
                name: declared.name.clone(),
                kind: declared.kind.clone(),
                values: declared.values.clone(),
                schema: declared.schema.clone(),
                ..Default::default()
            })
            .collect(),
    })
}

/// dspy's `_default_instructions`, which stands in for a docstring nobody wrote.
///
/// Public because the derive needs it too: a `dspy.Signature` subclass with no docstring is
/// ordinary — the `conversation_history` tutorial writes one — and it gets these instructions, not
/// an error. Reachable rather than reimplemented, so the two spellings cannot drift.
pub fn default_instructions(inputs: &[&str], outputs: &[&str]) -> String {
    let quoted = |fields: &[&str]| {
        fields
            .iter()
            .map(|name| format!("`{name}`"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    format!(
        "Given the fields {}, produce the fields {}.",
        quoted(inputs),
        quoted(outputs)
    )
}

fn declared_instructions(inputs: &[Declared], outputs: &[Declared]) -> String {
    fn names(fields: &[Declared]) -> Vec<&str> {
        fields
            .iter()
            .map(|declared| declared.name.as_str())
            .collect()
    }
    default_instructions(&names(inputs), &names(outputs))
}

/// One name declared twice on the same side is one field — Python's dict semantics, which are
/// upstream's whole implementation here.
///
/// dspy collects each side into a `dict` keyed by name, so a repeat *overwrites* rather than adding:
/// `"q: int, ctx: str, q: str -> a"` is two inputs, `q` still first and now a `str`. **First
/// position, last value**, measured on 3.3.0.
///
/// A `Vec` keeps both, and the field then appears twice in the prompt — the adapter asks the model
/// to answer it once per copy and rejects a reply that answered it once. Nothing here is reachable
/// from a well-formed hand-written string; it is reachable the moment a signature is *generated*,
/// which is what a graph builder does.
fn collapsed(declared: Vec<Declared>) -> Vec<Declared> {
    let mut order: Vec<String> = Vec::new();
    let mut latest: std::collections::HashMap<String, Declared> = std::collections::HashMap::new();
    for field in declared {
        if !latest.contains_key(&field.name) {
            order.push(field.name.clone());
        }
        latest.insert(field.name.clone(), field);
    }
    order
        .into_iter()
        .filter_map(|name| latest.remove(&name))
        .collect()
}

/// dspy sorts the names it found on both sides, so the message does not depend on which side
/// was walked first.
fn refuse_duplicates(inputs: &[Declared], outputs: &[Declared]) -> Result<()> {
    let mut shared: Vec<&str> = inputs
        .iter()
        .filter(|input| outputs.iter().any(|output| output.name == input.name))
        .map(|input| input.name.as_str())
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
fn declarations(text: &str, side: &str) -> Result<Vec<Declared>> {
    let parts = split_top_level(text, ',');
    // An empty part is a stray comma — `q,, x` — which upstream's `ast.parse` refuses. Skipping it
    // silently made `q,, x -> a` a two-input signature that Python would not have built at all.
    let trailing = parts.len().saturating_sub(1);
    parts
        .into_iter()
        .enumerate()
        .filter(|(at, part)| !(*at == trailing && part.trim().is_empty()))
        .map(|(_, part)| declaration(part.trim(), side))
        .collect()
}

/// One field as the string form declares it.
struct Declared {
    name: String,
    kind: FieldKind,
    /// The members, when the annotation is a `Literal`. Nothing else closes a field's set.
    values: Option<Vec<LiteralValue>>,
    /// What the annotation names, as a schema — the note an output prints. A struct hands this to
    /// schemars; a string signature has no Rust type, so the annotation is resolved instead.
    schema: Option<Value>,
}

fn declaration(part: &str, side: &str) -> Result<Declared> {
    let (name, annotation) = match part.split_once(':') {
        // An unannotated field is a string, which upstream notes it would rather be explicit
        // about and cannot be without breaking programs that leave the type off.
        None => (part.trim(), None),
        Some((name, annotation)) => (name.trim(), Some(annotation.trim())),
    };
    super::identifier::refuse_unless_identifier(name, side)?;
    let (kind, values) = kind_of(annotation, name)?;
    // Only a `Json` field prints a schema; a scalar states its shape in the field line itself.
    let schema = match &kind {
        FieldKind::Json(_) => annotation.and_then(|text| {
            // One of dspy's own types prints its type's schema, which no annotation string can
            // build; everything else prints the schema its annotation stands for.
            super::annotation::custom_type(text)
                .and_then(super::annotation::custom_schema)
                .or_else(|| super::annotation::schema_for(text))
        }),
        _ => None,
    };
    Ok(Declared {
        name: name.to_owned(),
        kind,
        values,
        schema,
    })
}

/// The scalar kinds name themselves; everything else travels as the annotation dspy would print.
fn kind_of(annotation: Option<&str>, name: &str) -> Result<(FieldKind, Option<Vec<LiteralValue>>)> {
    Ok(match annotation {
        // A bare `None` is not a type a field can hold, and upstream renders the field as a
        // plain `str` — the same as leaving the annotation off.
        None | Some("str") | Some("None") => (FieldKind::Str, None),
        Some("int") => (FieldKind::Int, None),
        Some("float") => (FieldKind::Float, None),
        Some("bool") => (FieldKind::Bool, None),
        Some(other) => match literal_members(other) {
            Some(values) => (FieldKind::Str, Some(values)),
            // An annotation this cannot resolve is one upstream cannot resolve either. dspy
            // evaluates it against builtins and `typing` and nothing else — `Image` is
            // `ValueError: Unknown name: Image` — and a Rust program has no namespace to add to,
            // so the resolvable set is closed and an unknown name is a mistake rather than a type
            // this port has not learned.
            //
            // Carried as an opaque field until then, which put `1. `a` (unknown_type):` in a
            // prompt and asked the model for a type that does not exist.
            None => {
                super::annotation::refuse_unless_resolvable(other, name)?;
                match super::annotation::custom_type(other) {
                    // A custom type renders under its class name — `dspy.History` is the only
                    // spelling that resolves, and `get_annotation_name` prints `History` — and it
                    // carries whatever prose the type says about itself.
                    Some(custom) => super::annotation::custom_field(custom),
                    None => (
                        FieldKind::Json(JsonType {
                            annotation: canonical(other),
                            ..Default::default()
                        }),
                        None,
                    ),
                }
            }
        },
    })
}

/// The members of a `Literal[...]`, under either spelling upstream resolves — `Literal[…]` and
/// `typing.Literal[…]` both come back as `typing.Literal[…]` from dspy's own string form.
///
/// Falling through to [`kind_of`]'s opaque-annotation arm instead cost more than a type name: a
/// field with no `values` renders no allowed-values note, so the model was never told what it may
/// answer, and nothing rejected a reply outside the set.
fn literal_members(annotation: &str) -> Option<Vec<LiteralValue>> {
    let inner = annotation
        .trim()
        .strip_prefix("typing.")
        .unwrap_or(annotation.trim())
        .strip_prefix("Literal[")?
        .strip_suffix(']')?;
    split_top_level(inner, ',')
        .into_iter()
        .map(|member| literal_member(member.trim()))
        .collect::<Option<Vec<_>>>()
        .filter(|members| !members.is_empty())
}

/// One member, as Python spells it inside the annotation.
fn literal_member(member: &str) -> Option<LiteralValue> {
    if member.len() > 1
        && let Some(quote) = member.chars().next().filter(|c| *c == '\'' || *c == '"')
        && member.ends_with(quote)
    {
        return Some(LiteralValue::Str(member[1..member.len() - 1].to_owned()));
    }
    match member {
        "True" => Some(LiteralValue::Bool(true)),
        "False" => Some(LiteralValue::Bool(false)),
        "" => None,
        // An enum member reaches the annotation as `Colour.RED`, which is neither quoted nor a
        // number — upstream prints it bare and asks the model for it bare.
        other => Some(match other.parse::<i64>() {
            Ok(number) => LiteralValue::Int(number),
            Err(_) => LiteralValue::Bare(other.to_owned()),
        }),
    }
}

/// The spelling Python prints for an annotation, which is not always the one it was given.
///
/// Upstream resolves an annotation into a `typing` object and prints *that object* — so every
/// optional, however it was spelled, comes back as `Union[T, NoneType]`. `Optional[str]`,
/// `str | None` and `Union[str, None]` are three ways of writing one type and one way of printing
/// it. This claimed the reverse, that a union collapses *to* `Optional`, which is the spelling no
/// annotation ever reaches a prompt under. Measured over fifteen annotations, ten read differently
/// here than upstream; the derive had it right the whole time and only the string form did not.
///
/// A module-qualified name loses its module: `list[dspy.Tool]` reads `list[Tool]`, because what is
/// printed is the class, not the path that reached it.
///
/// All of it applies at every depth, because the members are resolved before the type around them.
fn canonical(annotation: &str) -> String {
    let annotation = annotation.trim();
    let members = split_top_level(annotation, '|');
    if members.len() > 1 {
        return canonical_union(&members);
    }
    let Some((head, rest)) = annotation.split_once('[') else {
        return leaf(annotation);
    };
    let head = head.trim();
    let arguments = split_top_level(rest.strip_suffix(']').unwrap_or(rest), ',');
    // `Optional[T]` *is* a union, and prints as one.
    if head == "Optional" || head == "typing.Optional" {
        return canonical_union(&[arguments.first().copied().unwrap_or(""), "None"]);
    }
    if head == "Union" || head == "typing.Union" {
        return canonical_union(&arguments);
    }
    // A `Literal`'s members are values, not types: they are printed back rather than resolved,
    // with `typing`'s own spacing.
    if head == "Literal" || head == "typing.Literal" {
        let members: Vec<&str> = arguments.iter().map(|member| member.trim()).collect();
        return format!("Literal[{}]", members.join(", "));
    }
    let arguments: Vec<String> = arguments
        .iter()
        .map(|argument| canonical(argument))
        .collect();
    format!("{head}[{}]", arguments.join(", "))
}

/// One name as Python prints it: `None` is the type `NoneType`, and one of dspy's own types is
/// printed without the module that reached it.
fn leaf(annotation: &str) -> String {
    match annotation {
        "None" | "NoneType" => "NoneType".to_owned(),
        other => super::annotation::custom_type(other)
            .unwrap_or(other)
            .to_owned(),
    }
}

/// A union as `typing` builds one: nested unions flattened into it, repeats dropped, and a union
/// of one collapsed to the member itself.
///
/// All three are `typing`'s doing rather than dspy's — `Union[str]` *is* `str`,
/// `Optional[Optional[str]]` *is* `Optional[str]` — and printing the object is what dspy does. A
/// union kept as written prints a type Python cannot construct.
fn canonical_union(members: &[&str]) -> String {
    let mut flattened: Vec<String> = Vec::new();
    for member in members {
        let spelled = canonical(member);
        match spelled
            .strip_prefix("Union[")
            .and_then(|rest| rest.strip_suffix(']'))
        {
            Some(nested) => flattened.extend(
                split_top_level(nested, ',')
                    .iter()
                    .map(|part| part.trim().to_owned()),
            ),
            None => flattened.push(spelled),
        }
    }
    flattened.dedup_by(|a, b| a == b);
    let mut seen: Vec<String> = Vec::new();
    for member in flattened {
        if !seen.contains(&member) {
            seen.push(member);
        }
    }
    match seen.as_slice() {
        [only] => only.clone(),
        _ => format!("Union[{}]", seen.join(", ")),
    }
}

/// Split on the separators outside every bracket: `dict[str, int]` is one field, not two, and
/// the `|` in `dict[str, int | None]` belongs to the dict's value rather than to the field.
pub(crate) fn split_top_level(text: &str, separator: char) -> Vec<&str> {
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

    /// A name declared twice on one side is one field: Python collects each side into a `dict`,
    /// so the repeat overwrites rather than adding. **First position, last value** — measured.
    ///
    /// A `Vec` kept both, and the field then appeared twice in the prompt: the adapter asked the
    /// model to answer it once per copy and rejected a reply that answered it once. Unreachable
    /// from a well-formed hand-written string, and reachable the moment a signature is generated —
    /// a graph builder deriving one output field per outgoing edge hits it on any fan-out.
    #[test]
    fn a_name_declared_twice_on_one_side_is_one_field() {
        let signature = parse("q, q -> answer, answer").expect("parses");
        let (inputs, outputs) = names(&signature);
        assert_eq!(inputs, ["q"]);
        assert_eq!(outputs, ["answer"]);
    }

    /// The survivor keeps the first declaration's *place* and the last one's *type*.
    #[test]
    fn the_repeat_keeps_its_place_and_takes_the_later_type() {
        let signature = parse("q: int, ctx: str, q: str -> a").expect("parses");
        let (inputs, _) = names(&signature);
        assert_eq!(inputs, ["q", "ctx"], "the first declaration's place");
        assert_eq!(
            signature.inputs[0].kind,
            FieldKind::Str,
            "the last declaration's type"
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
    use crate::make_signature;

    /// `Predict!("subject -> haiku")` builds the module, the way dspy's `Predict(spelling)` does.
    #[test]
    fn the_string_form_builds_a_module() {
        let mut haiku = crate::Predict!("subject -> haiku");
        let named: Vec<String> = crate::module::Module::named_predictors(&mut haiku)
            .into_iter()
            .map(|predictor| predictor.signature.outputs[0].name.clone())
            .collect();
        assert_eq!(named, ["haiku"]);
    }

    /// The spelling a caller writes, checked while this crate compiles.
    #[test]
    fn the_macro_reads_the_same_signature_the_parser_reads() {
        let built = make_signature!("subject -> haiku");
        assert_eq!(built.inputs[0].name, "subject");
        assert_eq!(built.outputs[0].name, "haiku");
        let parsed: crate::Signature = "subject -> haiku".parse().expect("parses");
        assert!(built == parsed, "the macro and the parser should agree");
    }
}

#[cfg(test)]
mod ergonomics {
    use std::sync::Arc;

    use crate::lm::global::install_for_test;
    use crate::{DummyLM, Predict, example};

    /// The shortest spelling end to end, against dspy's:
    ///
    /// ```python
    /// haiku_generator = dspy.Predict("subject -> haiku")
    /// result = haiku_generator(subject="computer science")
    /// ```
    #[allow(clippy::await_holding_lock)] // the installer's own note: `SERIAL` is a test token, taken by nothing under test
    #[tokio::test]
    async fn a_single_input_task_is_declared_and_called_in_two_lines() {
        let _configured = install_for_test(Arc::new(DummyLM::new([
            example! { haiku: "silicon dreaming" },
        ])));

        let haiku_generator = Predict!("subject -> haiku");
        let result = haiku_generator
            .call("computer science")
            .await
            .expect("asks");

        assert_eq!(result["haiku"], "silicon dreaming");
    }
}

#[cfg(test)]
mod many_inputs {
    use std::sync::Arc;

    use crate::lm::global::install_for_test;
    use crate::{DummyLM, Module, Predict, call, example, input};

    /// Both spellings, on a signature with more than one input.
    #[allow(clippy::await_holding_lock)] // the installer's own note: `SERIAL` is a test token, taken by nothing under test
    #[tokio::test]
    async fn several_inputs_are_named_in_either_spelling() {
        let _configured = install_for_test(Arc::new(DummyLM::keyed([(
            "computer science",
            example! { haiku: "silicon dreaming", mood: "wry" },
        )])));

        let haiku = Predict!("subject, tone -> haiku, mood");

        let first = haiku
            .forward(input! { subject: "computer science", tone: "wry" })
            .await
            .expect("asks");
        let second = call!(haiku, subject = "computer science", tone = "wry")
            .await
            .expect("asks");

        assert_eq!(first.get("haiku").unwrap(), "silicon dreaming");
        assert_eq!(first.get("mood").unwrap(), "wry");
        assert_eq!(first.example, second.example);
    }
}
