//! The keywords that apply other schemas: `anyOf`, `oneOf`, `if`/`then`/`else`.

use super::super::{Error, Walk};
use crate::python::repr;
use serde_json::Value;

pub(super) fn any_of(walk: &Walk, stated: &Value, instance: &Value, schema: &Value) -> Vec<Error> {
    let Some(subschemas) = stated.as_array() else {
        return Vec::new();
    };
    let mut all_errors = Vec::new();
    for sub in subschemas {
        let errors = walk.descend(instance, sub, None);
        if errors.is_empty() {
            return Vec::new();
        }
        all_errors.extend(errors);
    }
    vec![
        Error::new(
            format!(
                "{} is not valid under any of the given schemas",
                repr(instance)
            ),
            "anyOf",
            instance,
            schema,
        )
        .with_context(all_errors),
    ]
}

pub(super) fn one_of(walk: &Walk, stated: &Value, instance: &Value, schema: &Value) -> Vec<Error> {
    let Some(subschemas) = stated.as_array() else {
        return Vec::new();
    };
    let mut all_errors = Vec::new();
    let mut first_valid = None;
    let mut rest = subschemas.iter();
    for sub in rest.by_ref() {
        let errors = walk.descend(instance, sub, None);
        if errors.is_empty() {
            first_valid = Some(sub);
            break;
        }
        all_errors.extend(errors);
    }
    let Some(first_valid) = first_valid else {
        return vec![
            Error::new(
                format!(
                    "{} is not valid under any of the given schemas",
                    repr(instance)
                ),
                "oneOf",
                instance,
                schema,
            )
            .with_context(all_errors),
        ];
    };
    let mut more_valid: Vec<&Value> = rest.filter(|each| walk.is_valid(instance, each)).collect();
    if more_valid.is_empty() {
        return Vec::new();
    }
    more_valid.push(first_valid);
    let reprs: Vec<String> = more_valid.iter().map(|each| repr(each)).collect();
    vec![Error::new(
        format!(
            "{} is valid under each of {}",
            repr(instance),
            reprs.join(", ")
        ),
        "oneOf",
        instance,
        schema,
    )]
}

pub(super) fn if_then_else(
    walk: &Walk,
    stated: &Value,
    instance: &Value,
    schema: &Value,
) -> Vec<Error> {
    let branch = match walk.is_valid(instance, stated) {
        true => schema.get("then"),
        false => schema.get("else"),
    };
    branch
        .map(|sub| walk.descend(instance, sub, None))
        .unwrap_or_default()
}
