//! Making a reply's value fit the type its field declares — dspy's `parse_value`.
//!
//! Split from the type description beside it because the two answer different questions: what a
//! field's type *is*, and what a model's answer has to become to be one. Every function here can
//! refuse, and the sentence it refuses with reaches the model on a retry.

use anyhow::{Result, anyhow};
use serde_json::Value;

use super::field_type::{FieldKind, LiteralValue};

/// One value, cast to the kind its field declares.
pub(crate) fn coerce_value(kind: &FieldKind, name: &str, value: &mut Value) -> Result<()> {
    match kind {
        // `Reasoning` carries its content as text, exactly as a `Str` does.
        //
        // A value that is not already text becomes text the way Python's `str()` does, which is
        // what `parse_value(v, str)` calls: a JSON adapter can hand a `str` field a whole object,
        // and dspy renders it `{'why': 'Because.'}` — single quotes, `None`, `True` — not as JSON.
        FieldKind::Str | FieldKind::Reasoning => {
            if !value.is_string() {
                *value = Value::String(crate::python::repr(value));
            }
            Ok(())
        }
        FieldKind::Bool => coerce_bool(name, value),
        FieldKind::Int => coerce_int(name, value),
        FieldKind::Float => coerce_float(name, value),
        FieldKind::Json(json) => coerce_json(name, &json.annotation, value),
        // A member reaches the marker path as the text of its value, which is what the model
        // was asked for; naming the member it belongs to is the declared type's job.
        FieldKind::Enum(_) => Ok(()),
    }
}

/// dspy `parse_value`'s `Literal` branch: the member as it stands, or unwrapped from what a model
/// wrapped it in.
///
/// Runs *before* the kind's own coercion, as upstream's does — a `Literal` annotation never
/// reaches the `str` branch there, so a member that is not a string stays one here.
///
/// The three unwrappings are upstream's, in its order: trim, then a `Literal[…]` or `str[…]`
/// spelling of the annotation itself, then one matched pair of surrounding quotes. A value still
/// outside the set is refused in dspy's words, which reach the model as retry feedback.
pub(crate) fn coerce_literal(values: &[LiteralValue], value: &mut Value) -> Result<()> {
    if values.iter().any(|member| &member.to_json() == value) {
        return Ok(());
    }
    if let Some(text) = value.as_str() {
        let bare = unwrapped(text);
        if values.iter().any(|member| member.wire_form() == bare) {
            *value = Value::String(bare.to_owned());
            return Ok(());
        }
    }
    let allowed: Vec<Value> = values.iter().map(LiteralValue::to_json).collect();
    Err(anyhow!(
        "{} is not one of {}",
        crate::python::repr(value),
        crate::python::tuple(&allowed)
    ))
}

/// What upstream strips before looking a member up again.
fn unwrapped(text: &str) -> &str {
    let trimmed = text.trim();
    let inner = match trimmed.strip_suffix(']') {
        Some(head) if head.starts_with("Literal[") || head.starts_with("str[") => {
            head.split_once('[').expect("a prefix ending in `[`").1
        }
        _ => trimmed,
    };
    let mut characters = inner.chars();
    match (characters.next(), characters.next_back()) {
        (Some(open), Some(close))
            if open == close && (open == '\'' || open == '"') && inner.len() > 1 =>
        {
            &inner[open.len_utf8()..inner.len() - close.len_utf8()]
        }
        _ => inner,
    }
}

/// Either case of the two keywords, because the crate asks the model for a bool in Python's
/// spelling and renders one the same way; a reply that echoes what it was shown has to parse.
fn coerce_bool(name: &str, value: &mut Value) -> Result<()> {
    if value.is_boolean() {
        return Ok(());
    }
    let parsed = value
        .as_str()
        .and_then(|text| match text.trim().to_ascii_lowercase().as_str() {
            "true" => Some(true),
            "false" => Some(false),
            _ => None,
        });
    match parsed {
        Some(parsed) => {
            *value = Value::Bool(parsed);
            Ok(())
        }
        None => Err(anyhow!("{name} must be true or false, got {value}")),
    }
}

fn coerce_int(name: &str, value: &mut Value) -> Result<()> {
    if value.as_i64().is_some() {
        return Ok(());
    }
    if let Some(text) = value.as_str()
        && let Ok(parsed) = text.trim().parse::<i64>()
    {
        *value = Value::from(parsed);
        return Ok(());
    }
    Err(anyhow!("{name} must be an integer, got {value}"))
}

/// A native value of any non-string shape passes through; a string parses as JSON, with a
/// surrounding code fence tolerated because marker-path models like to wrap JSON in one.
fn coerce_json(name: &str, annotation: &str, value: &mut Value) -> Result<()> {
    let Some(text) = value.as_str() else {
        return Ok(());
    };
    match serde_json::from_str(strip_code_fence(text)) {
        Ok(parsed) => {
            *value = parsed;
            Ok(())
        }
        // A type whose own string form is not JSON — a `datetime`, a `date` — is what dspy hands to
        // that type rather than rejecting, so the bare string is left for the caller's typing (or,
        // across the bridge, `parse_value`) to read. A container or model still needs valid JSON.
        Err(_) if accepts_string_form(annotation) => Ok(()),
        Err(error) => Err(anyhow!("{name} must be valid JSON: {error}")),
    }
}

/// Whether this annotation's Python type can hold a bare, non-JSON string at all.
///
/// After a JSON parse fails, that is the only question worth asking: a value the type could never
/// hold is malformed, and one it could is a value in its own string form for the caller's typing
/// (or, across the bridge, `parse_value`) to read. dspy asks it by handing the string to
/// `TypeAdapter().validate_python` and seeing what happens.
///
/// The set was a list of the cases somebody had hit — `datetime`, `date`, `time`, `timedelta`,
/// `Code` — and it was five of eleven. Measured against pydantic instead: a union with a string
/// arm holds one (`Optional[str]` and `Union[str, int]` both do), `Any` and `bytes` hold one, and
/// so does every scalar whose serialization *is* a string. A container never does, whatever it
/// holds: `list[Optional[str]]` rejects a bare string even though its element would take it.
///
/// Found by comparing the `parse` direction, which nothing had compared: `JsonAdapter` and
/// `BamlAdapter` both refused `{"note": "seen"}` for an `Optional[str]` output that dspy reads.
fn accepts_string_form(annotation: &str) -> bool {
    let annotation = annotation.trim();
    if matches!(
        annotation,
        "str" | "Any" | "bytes" | "datetime" | "date" | "time" | "timedelta" | "UUID"
    ) || annotation == "Code"
        || annotation.starts_with("Code_")
        || annotation.starts_with("Literal[")
    {
        return true;
    }
    // A union holds a string when any arm does. A container never does, whatever it holds.
    union_arms(annotation).is_some_and(|arms| arms.iter().any(|arm| accepts_string_form(arm)))
}

/// The arms of `Union[...]`, and nothing for any other container.
fn union_arms(annotation: &str) -> Option<Vec<&str>> {
    let inside = annotation
        .strip_prefix("Union[")
        .or_else(|| annotation.strip_prefix("Optional["))?
        .strip_suffix(']')?;
    Some(crate::signature::split_top_level(inside, ','))
}

/// The content of a ```json ... ``` (or bare ```) block, when the whole text is one fence.
fn strip_code_fence(text: &str) -> &str {
    let trimmed = text.trim();
    let Some(fenced) = trimmed
        .strip_prefix("```")
        .and_then(|rest| rest.strip_suffix("```"))
    else {
        return trimmed;
    };
    fenced.strip_prefix("json").unwrap_or(fenced).trim()
}

/// Any native JSON number counts as a float; JSON cannot spell a non-finite one, so only
/// the string form needs the finiteness check.
fn coerce_float(name: &str, value: &mut Value) -> Result<()> {
    if value.is_number() {
        return Ok(());
    }
    if let Some(text) = value.as_str()
        && let Ok(parsed) = text.trim().parse::<f64>()
        && parsed.is_finite()
    {
        *value = Value::from(parsed);
        return Ok(());
    }
    Err(anyhow!("{name} must be a number, got {value}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A bare string is kept exactly where pydantic can hold one.
    ///
    /// The set was a list of the cases somebody had hit — `datetime`, `date`, `time`, `timedelta`,
    /// `Code` — five of eleven, and it refused `{"note": "seen"}` for an `Optional[str]` output
    /// that dspy reads. Measured against pydantic instead, the rule is: a union with a string arm
    /// holds one, `Any` and `bytes` hold one, and so does every scalar whose serialization *is* a
    /// string. A container never does, whatever its element would take.
    ///
    /// Found by comparing the `parse` direction, which nothing had ever compared.
    #[test]
    fn a_bare_string_is_kept_where_a_string_can_live() {
        for holds in [
            "str",
            "Any",
            "bytes",
            "datetime",
            "date",
            "time",
            "timedelta",
            "UUID",
            "Code",
            "Union[str, NoneType]",
            "Union[str, int]",
            "Optional[str]",
            "Literal['low', 'high']",
        ] {
            assert!(accepts_string_form(holds), "{holds} can hold a string");
        }
        // Measured: `list[Optional[str]]` rejects a bare string even though its element takes one.
        for refuses in [
            "int",
            "float",
            "bool",
            "list[str]",
            "dict[str, int]",
            "tuple[str, int]",
            "set[str]",
            "Union[int, NoneType]",
            "list[Union[str, NoneType]]",
        ] {
            assert!(!accepts_string_form(refuses), "{refuses} cannot");
        }
    }

    /// The whole point at the call site: an optional string output reads a bare value.
    #[test]
    fn an_optional_string_output_reads_a_bare_value() {
        let mut value = Value::from("seen");
        coerce_json("note", "Union[str, NoneType]", &mut value).expect("kept");
        assert_eq!(value, Value::from("seen"));

        let mut malformed = Value::from("seen");
        coerce_json("tags", "list[str]", &mut malformed)
            .expect_err("a list cannot hold a bare string");
    }
}
