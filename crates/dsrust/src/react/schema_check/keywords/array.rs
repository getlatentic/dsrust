//! The keywords that walk an array: `items`, `prefixItems`, `contains`.

use super::super::{Error, Step, Walk};
use crate::python::repr;
use serde_json::Value;

pub(super) fn items(walk: &Walk, stated: &Value, instance: &Value, schema: &Value) -> Vec<Error> {
    let Some(elements) = instance.as_array() else {
        return Vec::new();
    };
    let prefix = schema
        .get("prefixItems")
        .and_then(Value::as_array)
        .map_or(0, Vec::len);
    if elements.len() <= prefix {
        return Vec::new();
    }
    if stated == &Value::Bool(false) {
        let extra = elements.len() - prefix;
        let rest = match extra == 1 {
            true => elements[prefix].clone(),
            false => Value::Array(elements[prefix..].to_vec()),
        };
        let item = match prefix == 1 {
            true => "item",
            false => "items",
        };
        return vec![Error::new(
            format!(
                "Expected at most {prefix} {item} but found {extra} extra: {}",
                repr(&rest)
            ),
            "items",
            instance,
            schema,
        )];
    }
    elements
        .iter()
        .enumerate()
        .skip(prefix)
        .flat_map(|(index, element)| walk.descend(element, stated, Some(Step::Index(index))))
        .collect()
}

pub(super) fn prefix_items(
    walk: &Walk,
    stated: &Value,
    instance: &Value,
    _schema: &Value,
) -> Vec<Error> {
    let (Some(elements), Some(prefixes)) = (instance.as_array(), stated.as_array()) else {
        return Vec::new();
    };
    elements
        .iter()
        .zip(prefixes)
        .enumerate()
        .flat_map(|(index, (element, sub))| walk.descend(element, sub, Some(Step::Index(index))))
        .collect()
}

pub(super) fn contains(
    walk: &Walk,
    stated: &Value,
    instance: &Value,
    schema: &Value,
) -> Vec<Error> {
    let Some(elements) = instance.as_array() else {
        return Vec::new();
    };
    let min_contains = schema
        .get("minContains")
        .and_then(Value::as_u64)
        .unwrap_or(1) as usize;
    let max_contains = schema
        .get("maxContains")
        .and_then(Value::as_u64)
        .map_or(elements.len(), |n| n as usize);
    let mut matches = 0;
    for element in elements {
        if walk.is_valid(element, stated) {
            matches += 1;
            if matches > max_contains {
                return vec![Error::new(
                    format!(
                        "Too many items match the given schema (expected at most {max_contains})"
                    ),
                    "maxContains",
                    instance,
                    schema,
                )];
            }
        }
    }
    if matches >= min_contains {
        return Vec::new();
    }
    match matches {
        0 => vec![Error::new(
            format!(
                "{} does not contain items matching the given schema",
                repr(instance)
            ),
            "contains",
            instance,
            schema,
        )],
        _ => vec![Error::new(
            format!(
                "Too few items match the given schema (expected at least {min_contains} but only {matches} matched)"
            ),
            "minContains",
            instance,
            schema,
        )],
    }
}
