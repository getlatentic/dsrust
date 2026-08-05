//! `serde` and `serde_json` interoperation, behind the `serde` feature.
//!
//! [`Value`] is its own type because it has to be: Python distinguishes `7` from `7.0` and holds an
//! integer of any width, and `serde_json::Value` does neither. That is worth carrying inside the
//! parser and is a nuisance at the edges, so this makes the edges easy — a value converts to and
//! from `serde_json::Value`, serializes, deserializes, and [`Repair::loads_as`] goes straight to a
//! type of your own.
//!
//! What this does *not* do is replace [`crate::repair_json`]. `serde_json` writes JSON its own way;
//! Python's `json.dumps` puts a space after every comma and colon, escapes everything outside
//! `\x20`-`\x7e`, and renders a float with `float.__repr__` — `1e+16`, not `10000000000000000`.
//! Reproducing those bytes is the point of this crate, so the writer stays its own.

use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::value::{Object, Value};
use crate::{Error, Repair, Result};

impl Repair {
    /// Repairs the text and deserializes it into `T`.
    ///
    /// ```
    /// # #[cfg(feature = "serde")] {
    /// use serde::Deserialize;
    ///
    /// #[derive(Deserialize, PartialEq, Debug)]
    /// struct Answer {
    ///     answer: String,
    ///     score: i32,
    /// }
    ///
    /// let parsed: Answer = json_repair::Repair::new().loads_as("{answer: 'Paris', score: 7,}")?;
    /// assert_eq!(parsed, Answer { answer: "Paris".into(), score: 7 });
    /// # }
    /// # Ok::<(), json_repair::Error>(())
    /// ```
    ///
    /// The repair runs first and deserialization second, so a value the schema of `T` will not take
    /// is an error from `serde`, reported through [`Error`].
    pub fn loads_as<T: DeserializeOwned>(&self, text: &str) -> Result<T> {
        let value: serde_json::Value = self.loads(text)?.into();
        serde_json::from_value(value).map_err(|error| Error::new(&error.to_string()))
    }
}

/// Repairs the text and deserializes it into `T`, with the default arguments.
///
/// The shorthand for [`Repair::loads_as`], as [`crate::loads`] is for [`Repair::loads`].
pub fn loads_as<T: DeserializeOwned>(text: &str) -> Result<T> {
    Repair::new().loads_as(text)
}

impl From<Value> for serde_json::Value {
    /// Converts to `serde_json`'s value.
    ///
    /// Two of Python's shapes have no `serde_json` spelling and are named here rather than left to
    /// chance. An integer past `u64::MAX` becomes a float, which is what `serde_json` already does
    /// reading one out of ordinary JSON. A non-finite float becomes `null`, because JSON has no
    /// literal for one — [`crate::repair_json`] still writes Python's `NaN` and `Infinity`, so
    /// nothing is lost by going through this crate's own writer instead.
    fn from(value: Value) -> Self {
        match value {
            Value::Null => serde_json::Value::Null,
            Value::Bool(flag) => serde_json::Value::Bool(flag),
            Value::Int(number) => serde_json::Value::from(number),
            Value::BigInt(digits) => wide_integer(&digits),
            Value::Float(number) => serde_json::Number::from_f64(number)
                .map_or(serde_json::Value::Null, serde_json::Value::Number),
            Value::Str(text) => serde_json::Value::String(text),
            Value::Array(items) => {
                serde_json::Value::Array(items.into_iter().map(Into::into).collect())
            }
            Value::Object(fields) => serde_json::Value::Object(
                fields
                    .into_iter()
                    .map(|(key, value)| (key, value.into()))
                    .collect(),
            ),
        }
    }
}

/// An integer wider than a machine word, as near as `serde_json` can hold it.
fn wide_integer(digits: &str) -> serde_json::Value {
    if let Ok(number) = digits.parse::<i128>()
        && let Some(number) = serde_json::Number::from_i128(number)
    {
        return serde_json::Value::Number(number);
    }
    digits
        .parse::<f64>()
        .ok()
        .and_then(serde_json::Number::from_f64)
        .map_or(serde_json::Value::Null, serde_json::Value::Number)
}

impl From<serde_json::Value> for Value {
    /// Converts from `serde_json`'s value.
    ///
    /// A `serde_json` number that is neither an `i64` nor a `u64` becomes a [`Value::Float`], since
    /// that is the only shape it could have arrived in.
    fn from(value: serde_json::Value) -> Self {
        match value {
            serde_json::Value::Null => Value::Null,
            serde_json::Value::Bool(flag) => Value::Bool(flag),
            serde_json::Value::Number(number) => match number.as_i64() {
                Some(whole) => Value::Int(whole),
                None => match number.as_u64() {
                    Some(whole) => Value::BigInt(whole.to_string()),
                    None => Value::Float(number.as_f64().unwrap_or(f64::NAN)),
                },
            },
            serde_json::Value::String(text) => Value::Str(text),
            serde_json::Value::Array(items) => {
                Value::Array(items.into_iter().map(Into::into).collect())
            }
            serde_json::Value::Object(fields) => Value::Object(
                fields
                    .into_iter()
                    .map(|(key, value)| (key, value.into()))
                    .collect::<Object>(),
            ),
        }
    }
}

impl Serialize for Value {
    /// Serializes as the value it is, through `serde_json`'s shapes — so the *bytes* are
    /// `serde_json`'s, not Python's. Use [`crate::repair_json`] when the bytes matter.
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        serde_json::Value::from(self.clone()).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Value {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        serde_json::Value::deserialize(deserializer).map(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_value_round_trips_through_serde_json() {
        let value = crate::loads("{a: 1, b: [1.5, 'x'], c: null, d: true}").expect("repaired");
        let json: serde_json::Value = value.clone().into();
        assert_eq!(
            json.to_string(),
            r#"{"a":1,"b":[1.5,"x"],"c":null,"d":true}"#
        );
        assert_eq!(Value::from(json), value);
    }

    #[test]
    fn the_shapes_serde_json_has_no_room_for_are_named_rather_than_guessed() {
        let wide = Value::BigInt("123456789012345678901234567890".into());
        assert_eq!(
            serde_json::Value::from(wide).to_string(),
            "1.2345678901234568e+29"
        );
        assert_eq!(
            serde_json::Value::from(Value::Float(f64::INFINITY)),
            serde_json::Value::Null
        );
        // The crate's own writer still spells both the way Python does.
        assert_eq!(Value::Float(f64::INFINITY).to_string(), "Infinity");
        assert_eq!(
            Value::BigInt("123456789012345678901234567890".into()).to_string(),
            "123456789012345678901234567890"
        );
    }

    #[test]
    fn an_integer_that_fits_a_wide_word_keeps_every_digit() {
        // `serde_json` holds up to a `u64` without the `arbitrary_precision` feature, so the band
        // above `i64` and below `u64::MAX` is exact and everything past it is a float.
        let fits = Value::BigInt("18446744073709551615".into());
        assert_eq!(
            serde_json::Value::from(fits).to_string(),
            "18446744073709551615"
        );
        let does_not = Value::BigInt("99999999999999999999".into());
        assert_eq!(serde_json::Value::from(does_not).to_string(), "1e+20");
    }

    #[test]
    fn loads_as_deserializes_into_a_callers_own_type() {
        #[derive(Deserialize, PartialEq, Debug)]
        struct Answer {
            answer: String,
            tags: Vec<String>,
        }
        let parsed: Answer = loads_as("{answer: 'Paris', tags: [a, b],}").expect("repaired");
        assert_eq!(
            parsed,
            Answer {
                answer: "Paris".into(),
                tags: vec!["a".into(), "b".into()]
            }
        );
    }

    #[test]
    fn a_value_that_will_not_fit_the_type_is_an_error_and_not_a_panic() {
        #[derive(Deserialize, Debug)]
        struct Answer {
            #[allow(dead_code)]
            score: i32,
        }
        let refused = loads_as::<Answer>("{score: 'not a number'}");
        assert!(refused.is_err(), "got {refused:?}");
    }
}
