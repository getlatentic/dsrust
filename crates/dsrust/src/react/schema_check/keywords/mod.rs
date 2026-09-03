//! `jsonschema._keywords`, one function per keyword, each yielding the errors and messages upstream
//! yields. `apply` dispatches a keyword to its function; the functions live with the kind of
//! instance they judge. `walk` carries the root schema for `$ref`; `schema` is the subschema the
//! keyword sits in.

mod applicator;
mod array;
mod object;
mod scalar;

use super::values::{equal, uniq};
use super::{Error, Walk};
use crate::python::repr;
use applicator::{any_of, if_then_else, one_of};
use array::{contains, items, prefix_items};
use object::{
    additional_properties, dependent_required, dependent_schemas, pattern_properties, properties,
    required,
};
use scalar::{bound, length, multiple_of, pattern, type_of};
use serde_json::{Map, Value};

pub(super) fn apply(
    walk: &Walk,
    keyword: &str,
    stated: &Value,
    instance: &Value,
    schema: &Value,
) -> Vec<Error> {
    match keyword {
        "$ref" => stated
            .as_str()
            .map(|reference| walk.reference(reference, instance))
            .unwrap_or_default(),
        "additionalProperties" => additional_properties(walk, stated, instance, schema),
        "items" => items(walk, stated, instance, schema),
        "prefixItems" => prefix_items(walk, stated, instance, schema),
        "const" => one_if(
            !equal(instance, stated),
            || format!("{} was expected", repr(stated)),
            "const",
            instance,
            schema,
        ),
        "contains" => contains(walk, stated, instance, schema),
        "exclusiveMinimum" => bound(
            instance,
            stated,
            schema,
            "exclusiveMinimum",
            |x, m| x <= m,
            "is less than or equal to the minimum of",
        ),
        "exclusiveMaximum" => bound(
            instance,
            stated,
            schema,
            "exclusiveMaximum",
            |x, m| x >= m,
            "is greater than or equal to the maximum of",
        ),
        "minimum" => bound(
            instance,
            stated,
            schema,
            "minimum",
            |x, m| x < m,
            "is less than the minimum of",
        ),
        "maximum" => bound(
            instance,
            stated,
            schema,
            "maximum",
            |x, m| x > m,
            "is greater than the maximum of",
        ),
        "multipleOf" => multiple_of(stated, instance, schema),
        "minItems" => sized(
            instance,
            stated,
            schema,
            Value::as_array,
            |n, m| n < m,
            Limit::least("minItems", "is too short"),
        ),
        "maxItems" => sized(
            instance,
            stated,
            schema,
            Value::as_array,
            |n, m| n > m,
            Limit::most("maxItems", "is too long"),
        ),
        "uniqueItems" => one_if(
            stated.as_bool() == Some(true) && instance.as_array().is_some_and(|items| !uniq(items)),
            || format!("{} has non-unique elements", repr(instance)),
            "uniqueItems",
            instance,
            schema,
        ),
        "pattern" => pattern(stated, instance, schema),
        "minLength" => length(
            instance,
            stated,
            schema,
            |n, m| n < m,
            Limit::least("minLength", "is too short"),
        ),
        "maxLength" => length(
            instance,
            stated,
            schema,
            |n, m| n > m,
            Limit::most("maxLength", "is too long"),
        ),
        "dependentRequired" => dependent_required(stated, instance, schema),
        "dependentSchemas" => dependent_schemas(walk, stated, instance),
        "enum" => one_if(
            stated
                .as_array()
                .is_some_and(|enums| enums.iter().all(|each| !equal(each, instance))),
            || format!("{} is not one of {}", repr(instance), repr(stated)),
            "enum",
            instance,
            schema,
        ),
        "type" => type_of(stated, instance, schema),
        "properties" => properties(walk, stated, instance),
        "required" => required(stated, instance, schema),
        "minProperties" => sized(
            instance,
            stated,
            schema,
            Value::as_object,
            |n, m| n < m,
            Limit::least("minProperties", "does not have enough properties"),
        ),
        "maxProperties" => sized(
            instance,
            stated,
            schema,
            Value::as_object,
            |n, m| n > m,
            Limit::most("maxProperties", "has too many properties"),
        ),
        "allOf" => stated
            .as_array()
            .map(|all| {
                all.iter()
                    .flat_map(|sub| walk.descend(instance, sub, None))
                    .collect()
            })
            .unwrap_or_default(),
        "anyOf" => any_of(walk, stated, instance, schema),
        "oneOf" => one_of(walk, stated, instance, schema),
        "not" => one_if(
            walk.is_valid(instance, stated),
            || {
                format!(
                    "{} should not be valid under {}",
                    repr(instance),
                    repr(stated)
                )
            },
            "not",
            instance,
            schema,
        ),
        "if" => if_then_else(walk, stated, instance, schema),
        "patternProperties" => pattern_properties(walk, stated, instance),
        "propertyNames" => instance
            .as_object()
            .map(|object| {
                object
                    .keys()
                    .flat_map(|key| walk.descend(&Value::String(key.clone()), stated, None))
                    .collect()
            })
            .unwrap_or_default(),
        _ => Vec::new(),
    }
}

pub(super) fn one_if(
    condition: bool,
    message: impl FnOnce() -> String,
    validator: &'static str,
    instance: &Value,
    schema: &Value,
) -> Vec<Error> {
    match condition {
        true => vec![Error::new(message(), validator, instance, schema)],
        false => Vec::new(),
    }
}

/// A size keyword's wording: what it says at the one limit upstream words specially, and otherwise.
pub(super) struct Limit {
    validator: &'static str,
    at_special: &'static str,
    otherwise: &'static str,
    special: usize,
}

impl Limit {
    pub(super) fn least(validator: &'static str, otherwise: &'static str) -> Self {
        Limit {
            validator,
            at_special: "should be non-empty",
            otherwise,
            special: 1,
        }
    }

    pub(super) fn most(validator: &'static str, otherwise: &'static str) -> Self {
        Limit {
            validator,
            at_special: "is expected to be empty",
            otherwise,
            special: 0,
        }
    }

    pub(super) fn phrase(&self, limit: usize) -> &'static str {
        match limit == self.special {
            true => self.at_special,
            false => self.otherwise,
        }
    }
}

pub(super) fn sized<T: ?Sized + Len>(
    instance: &Value,
    stated: &Value,
    schema: &Value,
    as_sized: fn(&Value) -> Option<&T>,
    breached: fn(usize, usize) -> bool,
    limit: Limit,
) -> Vec<Error> {
    let (Some(sized), Some(stated_limit)) = (as_sized(instance), stated.as_u64()) else {
        return Vec::new();
    };
    let stated_limit = stated_limit as usize;
    one_if(
        breached(sized.len(), stated_limit),
        || format!("{} {}", repr(instance), limit.phrase(stated_limit)),
        limit.validator,
        instance,
        schema,
    )
}

pub(super) trait Len {
    fn len(&self) -> usize;
}

impl Len for Vec<Value> {
    fn len(&self) -> usize {
        Vec::len(self)
    }
}

impl Len for Map<String, Value> {
    fn len(&self) -> usize {
        Map::len(self)
    }
}
