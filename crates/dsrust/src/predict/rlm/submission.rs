//! dspy `RLM._process_final_output`: what a `SUBMIT()` must be, and what it is told when it is not.
//!
//! Three stages, and each answers with prose rather than raising: a refused submission is the next
//! turn's output, so the model reads why and submits again. The whole point is that the run does
//! not end on a bad `SUBMIT()`.

use serde_json::{Map, Value};

use crate::adapter::python_json::python_type_name;
use crate::signature::{FieldKind, LiteralValue, OutField, Signature, coerce_value};

/// The submitted mapping, or the message the model is shown instead.
pub(crate) fn process(signature: &Signature, value: &Value) -> Result<Map<String, Value>, String> {
    let names: Vec<&str> = signature
        .outputs
        .iter()
        .map(|field| field.name.as_str())
        .collect();

    let Some(fields) = value.as_object() else {
        return Err(format!(
            "[Error] FINAL returned {}, expected dict with fields: {names:?}",
            python_type_name(value)
        ));
    };

    let mut missing: Vec<&str> = names
        .iter()
        .copied()
        .filter(|name| !fields.contains_key(*name))
        .collect();
    if !missing.is_empty() {
        missing.sort_unstable();
        return Err(format!(
            "[Error] Missing output fields: {missing:?}. Use SUBMIT({})",
            names.join(", ")
        ));
    }

    typed(signature, fields)
}

/// dspy's third stage: every value read as its field's annotation, and every field that could not
/// be reported at once rather than the first one stopping the rest.
fn typed(signature: &Signature, fields: &Map<String, Value>) -> Result<Map<String, Value>, String> {
    let mut outputs = Map::new();
    let mut errors = Vec::new();
    for field in &signature.outputs {
        let submitted = &fields[&field.name];
        match read(field, submitted.clone()) {
            Ok(value) => {
                outputs.insert(field.name.clone(), value);
            }
            Err(why) => errors.push(format!(
                "{}: expected {}, got {}: {why}",
                field.name,
                annotation_name(field),
                python_type_name(submitted)
            )),
        }
    }
    match errors.is_empty() {
        true => Ok(outputs),
        false => Err(format!("[Type Error] {}", errors.join("; "))),
    }
}

/// One value as its field's annotation, or why it is not one.
fn read(field: &OutField, mut value: Value) -> Result<Value, String> {
    if let Some(members) = &field.values {
        return member_of(members, value);
    }
    coerce_value(&field.kind, &field.name, &mut value)
        .map(|()| value)
        .map_err(|error| error.to_string())
}

/// dspy's `parse_value` on a `Literal`: the value as written, else the value with a surrounding
/// quote or `Literal[...]` wrapper taken off, else the refusal — which is a Python `repr` pair and
/// reaches a prompt, so it is spelled Python's way.
fn member_of(members: &[LiteralValue], value: Value) -> Result<Value, String> {
    let allowed: Vec<Value> = members.iter().map(LiteralValue::to_json).collect();
    if allowed.contains(&value) {
        return Ok(value);
    }
    if let Some(text) = value.as_str() {
        let unwrapped = Value::String(unwrapped(text).to_owned());
        if allowed.contains(&unwrapped) {
            return Ok(unwrapped);
        }
    }
    Err(format!(
        "{} is not one of {}",
        crate::python::repr(&value),
        crate::python::tuple(&allowed)
    ))
}

/// dspy strips a `Literal[…]`/`str[…]` wrapper and then one matched quote pair, in that order.
fn unwrapped(text: &str) -> &str {
    let text = text.trim();
    let inner =
        match text.ends_with(']') && (text.starts_with("Literal[") || text.starts_with("str[")) {
            true => &text[text.find('[').expect("a bracket") + 1..text.len() - 1],
            false => text,
        };
    let bytes = inner.as_bytes();
    match inner.len() > 1
        && bytes[0] == bytes[inner.len() - 1]
        && (bytes[0] == b'"' || bytes[0] == b'\'')
    {
        true => &inner[1..inner.len() - 1],
        false => inner,
    }
}

/// dspy's `annotation.__name__`, which is the *base* of the type rather than its full spelling:
/// `Literal['yes', 'no']` reports `Literal`, and `list[str]` reports `list`.
fn annotation_name(field: &OutField) -> String {
    if field.values.is_some() {
        return "Literal".to_owned();
    }
    match &field.kind {
        FieldKind::Str | FieldKind::Reasoning => "str".to_owned(),
        FieldKind::Bool => "bool".to_owned(),
        FieldKind::Int => "int".to_owned(),
        FieldKind::Float => "float".to_owned(),
        FieldKind::Enum(name) => name.clone(),
        FieldKind::Json(json) => {
            let annotation = &json.annotation;
            annotation[..annotation.find('[').unwrap_or(annotation.len())].to_owned()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn signature(spelling: &str) -> Signature {
        spelling.parse().expect("parses")
    }

    /// A signature whose named output carries a closed set. The string spelling does not build one
    /// — `values` is what a derive, a saved program or the bridge sets — so it is set here the way
    /// a caller with a `Literal` field has it.
    fn closed(spelling: &str, field: &str, members: &[LiteralValue]) -> Signature {
        let mut signature = signature(spelling);
        for output in &mut signature.outputs {
            if output.name == field {
                output.kind = FieldKind::Str;
                output.values = Some(members.to_vec());
            }
        }
        signature
    }

    fn words(members: &[&str]) -> Vec<LiteralValue> {
        members
            .iter()
            .map(|m| LiteralValue::Str((*m).to_owned()))
            .collect()
    }

    #[test]
    fn a_submission_that_is_not_a_mapping_names_what_it_was() {
        let refused = process(&signature("q -> answer"), &json!([1, 2])).expect_err("refuses");
        assert_eq!(
            refused,
            "[Error] FINAL returned list, expected dict with fields: [\"answer\"]"
        );
    }

    #[test]
    fn a_missing_field_is_named_and_the_call_is_spelled_out() {
        let refused = process(
            &signature("q -> answer, count: int"),
            &json!({ "answer": "a" }),
        )
        .expect_err("refuses");
        assert_eq!(
            refused,
            "[Error] Missing output fields: [\"count\"]. Use SUBMIT(answer, count)"
        );
    }

    /// The case that motivated the stage: a value outside its closed set, in dspy's own wording.
    #[test]
    fn a_value_outside_its_closed_set_is_refused_in_pythons_spelling() {
        let signature = closed("q -> answer", "answer", &words(&["yes", "no"]));
        let refused = process(&signature, &json!({ "answer": "maybe" })).expect_err("refuses");
        assert_eq!(
            refused,
            "[Type Error] answer: expected Literal, got str: 'maybe' is not one of ('yes', 'no')"
        );
        // And the same set accepts a member.
        let accepted = process(&signature, &json!({ "answer": "yes" })).expect("accepts");
        assert_eq!(accepted["answer"], json!("yes"));
    }

    /// dspy takes a `Literal[…]` wrapper and a quote pair off before giving up.
    #[test]
    fn a_wrapped_member_is_unwrapped_before_being_refused() {
        let signature = closed("q -> answer", "answer", &words(&["yes", "no"]));
        for written in ["'yes'", "\"yes\"", "Literal['yes']", "Literal[yes]"] {
            let accepted = process(&signature, &json!({ "answer": written }))
                .unwrap_or_else(|why| panic!("{written} should be read as a member: {why}"));
            assert_eq!(accepted["answer"], json!("yes"), "for {written}");
        }
    }

    /// A one-member set keeps Python's trailing comma, and a non-string member its own spelling.
    #[test]
    fn a_refusal_spells_the_allowed_tuple_the_way_python_does() {
        let one = process(
            &closed("q -> answer", "answer", &words(&["yes"])),
            &json!({ "answer": "no" }),
        )
        .expect_err("refuses");
        assert!(one.ends_with("'no' is not one of ('yes',)"), "got: {one}");
        let numbers = process(
            &closed("q -> n", "n", &[LiteralValue::Int(1), LiteralValue::Int(2)]),
            &json!({ "n": 3 }),
        )
        .expect_err("refuses");
        assert!(
            numbers.ends_with("3 is not one of (1, 2)"),
            "got: {numbers}"
        );
    }

    /// Every failing field is reported, not just the first, joined the way dspy joins them.
    #[test]
    fn every_bad_field_is_reported_at_once() {
        let mut signature = closed("q -> a, b", "a", &words(&["x"]));
        signature.outputs[1].values = Some(words(&["y"]));
        let refused = process(&signature, &json!({ "a": "1", "b": "2" })).expect_err("refuses");
        assert_eq!(
            refused,
            "[Type Error] a: expected Literal, got str: '1' is not one of ('x',); \
             b: expected Literal, got str: '2' is not one of ('y',)"
        );
    }

    /// A scalar that cannot be read as its kind is refused too, and names the kind it wanted.
    #[test]
    fn a_scalar_that_does_not_parse_names_its_annotation() {
        let refused = process(&signature("q -> count: int"), &json!({ "count": "many" }))
            .expect_err("refuses");
        assert!(
            refused.starts_with("[Type Error] count: expected int, got str: "),
            "got: {refused}"
        );
        // A string that does parse is coerced, as dspy's `parse_value` coerces it.
        let accepted =
            process(&signature("q -> count: int"), &json!({ "count": "42" })).expect("accepts");
        assert_eq!(accepted["count"], json!(42));
    }

    /// Only the signature's own outputs are kept, in its order.
    #[test]
    fn extra_keys_are_dropped_and_the_order_is_the_signatures() {
        let accepted = process(
            &signature("q -> answer, count: int"),
            &json!({ "count": 1, "answer": "a", "extra": true }),
        )
        .expect("accepts");
        assert_eq!(accepted.keys().collect::<Vec<_>>(), ["answer", "count"]);
    }
}
