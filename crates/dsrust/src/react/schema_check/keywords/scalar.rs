//! The keywords that judge one scalar: its type, its bounds, its length, its pattern.

use super::super::Error;
use super::super::values::{as_f64, char_count, is_type};
use super::{Limit, one_if};
use crate::python::repr;
use serde_json::Value;

pub(super) fn type_of(stated: &Value, instance: &Value, schema: &Value) -> Vec<Error> {
    let kinds: Vec<&str> = match stated {
        Value::String(one) => vec![one],
        Value::Array(many) => many.iter().filter_map(Value::as_str).collect(),
        _ => return Vec::new(),
    };
    let stated_as: Vec<String> = kinds
        .iter()
        .map(|kind| repr(&Value::String((*kind).to_owned())))
        .collect();
    one_if(
        !kinds.iter().any(|kind| is_type(instance, kind)),
        || format!("{} is not of type {}", repr(instance), stated_as.join(", ")),
        "type",
        instance,
        schema,
    )
}

pub(super) fn bound(
    instance: &Value,
    stated: &Value,
    schema: &Value,
    validator: &'static str,
    breached: fn(f64, f64) -> bool,
    phrase: &str,
) -> Vec<Error> {
    let (Some(x), Some(limit)) = (as_f64(instance), as_f64(stated)) else {
        return Vec::new();
    };
    one_if(
        !instance.is_boolean() && breached(x, limit),
        || format!("{} {phrase} {}", repr(instance), repr(stated)),
        validator,
        instance,
        schema,
    )
}

pub(super) fn multiple_of(stated: &Value, instance: &Value, schema: &Value) -> Vec<Error> {
    let (Some(x), Some(divisor)) = (as_f64(instance), as_f64(stated)) else {
        return Vec::new();
    };
    if instance.is_boolean() {
        return Vec::new();
    }
    let failed = match stated.is_f64() {
        true => {
            let quotient = x / divisor;
            !quotient.is_finite() || quotient.trunc() != quotient
        }
        false => x % divisor != 0.0,
    };
    one_if(
        failed,
        || format!("{} is not a multiple of {}", repr(instance), repr(stated)),
        "multipleOf",
        instance,
        schema,
    )
}

pub(super) fn length(
    instance: &Value,
    stated: &Value,
    schema: &Value,
    breached: fn(usize, usize) -> bool,
    limit: Limit,
) -> Vec<Error> {
    let (Some(text), Some(stated_limit)) = (instance.as_str(), stated.as_u64()) else {
        return Vec::new();
    };
    let stated_limit = stated_limit as usize;
    one_if(
        breached(char_count(text), stated_limit),
        || format!("{} {}", repr(instance), limit.phrase(stated_limit)),
        limit.validator,
        instance,
        schema,
    )
}

pub(super) fn pattern(stated: &Value, instance: &Value, schema: &Value) -> Vec<Error> {
    let (Some(text), Some(pattern)) = (instance.as_str(), stated.as_str()) else {
        return Vec::new();
    };
    let found = regex::Regex::new(pattern).is_ok_and(|re| re.is_match(text));
    one_if(
        !found,
        || format!("{} does not match {}", repr(instance), repr(stated)),
        "pattern",
        instance,
        schema,
    )
}
