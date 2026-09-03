//! The keywords that walk an object: its properties, what it must carry, what it may not.

use super::super::{Error, Step, Walk};
use crate::python::repr;
use serde_json::{Map, Value};

pub(super) fn additional_properties(
    walk: &Walk,
    stated: &Value,
    instance: &Value,
    schema: &Value,
) -> Vec<Error> {
    let Some(object) = instance.as_object() else {
        return Vec::new();
    };
    let extras = find_additional_properties(object, schema);
    if stated.is_object() {
        return extras
            .iter()
            .flat_map(|extra| walk.descend(&object[extra], stated, Some(Step::Key(extra.clone()))))
            .collect();
    }
    if stated != &Value::Bool(false) || extras.is_empty() {
        return Vec::new();
    }
    let mut sorted = extras;
    sorted.sort();
    let message = match schema.get("patternProperties").and_then(Value::as_object) {
        Some(patterns) => {
            let verb = match sorted.len() {
                1 => "does",
                _ => "do",
            };
            let mut names: Vec<&String> = patterns.keys().collect();
            names.sort();
            format!(
                "{} {verb} not match any of the regexes: {}",
                sorted
                    .iter()
                    .map(|each| repr(&Value::String(each.clone())))
                    .collect::<Vec<_>>()
                    .join(", "),
                names
                    .iter()
                    .map(|each| repr(&Value::String((*each).clone())))
                    .collect::<Vec<_>>()
                    .join(", "),
            )
        }
        None => {
            let verb = match sorted.len() {
                1 => "was",
                _ => "were",
            };
            format!(
                "Additional properties are not allowed ({} {verb} unexpected)",
                sorted
                    .iter()
                    .map(|each| repr(&Value::String(each.clone())))
                    .collect::<Vec<_>>()
                    .join(", "),
            )
        }
    };
    vec![Error::new(
        message,
        "additionalProperties",
        instance,
        schema,
    )]
}

/// `_utils.find_additional_properties`: the instance's keys that `properties` does not name and no
/// `patternProperties` pattern matches.
fn find_additional_properties(object: &Map<String, Value>, schema: &Value) -> Vec<String> {
    let named = schema.get("properties").and_then(Value::as_object);
    let patterns: Vec<regex::Regex> = schema
        .get("patternProperties")
        .and_then(Value::as_object)
        .map(|each| {
            each.keys()
                .filter_map(|pattern| regex::Regex::new(pattern).ok())
                .collect()
        })
        .unwrap_or_default();
    object
        .keys()
        .filter(|key| !named.is_some_and(|named| named.contains_key(*key)))
        .filter(|key| !patterns.iter().any(|pattern| pattern.is_match(key)))
        .cloned()
        .collect()
}

pub(super) fn properties(walk: &Walk, stated: &Value, instance: &Value) -> Vec<Error> {
    let (Some(object), Some(declared)) = (instance.as_object(), stated.as_object()) else {
        return Vec::new();
    };
    declared
        .iter()
        .filter_map(|(property, sub)| object.get(property).map(|value| (property, value, sub)))
        .flat_map(|(property, value, sub)| {
            walk.descend(value, sub, Some(Step::Key(property.clone())))
        })
        .collect()
}

pub(super) fn pattern_properties(walk: &Walk, stated: &Value, instance: &Value) -> Vec<Error> {
    let (Some(object), Some(patterns)) = (instance.as_object(), stated.as_object()) else {
        return Vec::new();
    };
    patterns
        .iter()
        .filter_map(|(pattern, sub)| regex::Regex::new(pattern).ok().map(|re| (re, sub)))
        .flat_map(|(re, sub)| {
            object
                .iter()
                .filter(move |(key, _)| re.is_match(key))
                .flat_map(move |(key, value)| {
                    walk.descend(value, sub, Some(Step::Key(key.clone())))
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

pub(super) fn required(stated: &Value, instance: &Value, schema: &Value) -> Vec<Error> {
    let (Some(object), Some(names)) = (instance.as_object(), stated.as_array()) else {
        return Vec::new();
    };
    names
        .iter()
        .filter_map(Value::as_str)
        .filter(|name| !object.contains_key(*name))
        .map(|name| {
            Error::new(
                format!(
                    "{} is a required property",
                    repr(&Value::String(name.to_owned()))
                ),
                "required",
                instance,
                schema,
            )
        })
        .collect()
}

pub(super) fn dependent_required(stated: &Value, instance: &Value, schema: &Value) -> Vec<Error> {
    let (Some(object), Some(dependencies)) = (instance.as_object(), stated.as_object()) else {
        return Vec::new();
    };
    dependencies
        .iter()
        .filter(|(property, _)| object.contains_key(*property))
        .flat_map(|(property, needed)| {
            needed
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .filter(|each| !object.contains_key(*each))
                .map(|each| {
                    Error::new(
                        format!(
                            "{} is a dependency of {}",
                            repr(&Value::String(each.to_owned())),
                            repr(&Value::String(property.clone()))
                        ),
                        "dependentRequired",
                        instance,
                        schema,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

pub(super) fn dependent_schemas(walk: &Walk, stated: &Value, instance: &Value) -> Vec<Error> {
    let (Some(object), Some(dependencies)) = (instance.as_object(), stated.as_object()) else {
        return Vec::new();
    };
    dependencies
        .iter()
        .filter(|(property, _)| object.contains_key(*property))
        .flat_map(|(_, dependency)| walk.descend(instance, dependency, None))
        .collect()
}
