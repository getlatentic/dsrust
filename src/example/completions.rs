//! dspy `primitives/prediction.py`: the `Completions` container.

use anyhow::{Result, bail};
use serde_json::Value;

use super::{Example, Prediction};

/// dspy's `Completions`: several candidate answers to one request, held by field rather than by
/// candidate — `{"answer": ["red", "blue"]}` for two candidates, not two mappings.
///
/// That is the shape `Predict` produces when asked for `n` completions, and the shape
/// [`at`](Self::at) reads a single [`Prediction`] back out of. Fields keep the order they were
/// first seen in, as upstream's dict does.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Completions {
    fields: Vec<(String, Vec<Value>)>,
}

impl Completions {
    /// The by-field form directly. Every field must carry one value per candidate, which is the
    /// pair of assertions upstream makes on construction.
    pub fn new(fields: impl IntoIterator<Item = (impl Into<String>, Vec<Value>)>) -> Result<Self> {
        let fields: Vec<(String, Vec<Value>)> =
            fields.into_iter().map(|(name, values)| (name.into(), values)).collect();
        if let Some((_, first)) = fields.first()
            && let Some((name, values)) = fields.iter().find(|(_, values)| values.len() != first.len())
        {
            bail!(
                "all fields must hold one value per candidate; `{name}` holds {} where the first \
                 holds {}",
                values.len(),
                first.len()
            );
        }
        Ok(Self { fields })
    }

    /// dspy's list form: one mapping per candidate, transposed into one list per field. A field
    /// missing from a candidate would leave the lists uneven, which [`new`](Self::new) refuses.
    pub fn from_candidates<'a>(candidates: impl IntoIterator<Item = &'a Example>) -> Result<Self> {
        let mut fields: Vec<(String, Vec<Value>)> = Vec::new();
        for candidate in candidates {
            for (name, value) in candidate.fields() {
                match fields.iter_mut().find(|(field, _)| field == name) {
                    Some((_, values)) => values.push(value.clone()),
                    None => fields.push((name.to_owned(), vec![value.clone()])),
                }
            }
        }
        Self::new(fields)
    }

    /// How many candidates this holds — the length of any one field's values, since they agree.
    pub fn len(&self) -> usize {
        self.fields.first().map_or(0, |(_, values)| values.len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Every candidate's value for one field. dspy reaches the same list by attribute or by key.
    pub fn get(&self, name: &str) -> Option<&[Value]> {
        self.fields.iter().find(|(field, _)| field == name).map(|(_, values)| values.as_slice())
    }

    pub fn contains(&self, name: &str) -> bool {
        self.fields.iter().any(|(field, _)| field == name)
    }

    /// One candidate as a prediction — dspy's `completions[i]`, which rebuilds a `Prediction` from
    /// each field's i-th value. The raw reply is not held per field, so it comes back empty.
    pub fn at(&self, index: usize) -> Option<Prediction> {
        if index >= self.len() {
            return None;
        }
        let example = Example::new(
            self.fields.iter().map(|(name, values)| (name.clone(), values[index].clone())),
        );
        Some(Prediction::new(example, ""))
    }

    /// Each field with every candidate's value for it, in order.
    pub fn items(&self) -> impl Iterator<Item = (&str, &[Value])> {
        self.fields.iter().map(|(name, values)| (name.as_str(), values.as_slice()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::example;
    use serde_json::json;

    fn two_candidates() -> Completions {
        Completions::new([("answer", vec![json!("red"), json!("blue")])]).expect("even fields")
    }

    #[test]
    fn it_holds_each_candidates_value_by_field() {
        let completions = two_candidates();
        assert_eq!(completions.len(), 2);
        assert_eq!(completions.get("answer"), Some([json!("red"), json!("blue")].as_slice()));
        assert!(completions.contains("answer"));
        assert!(completions.get("missing").is_none());
    }

    /// dspy transposes a list of candidate mappings into one list per field, first-seen order kept.
    #[test]
    fn it_transposes_candidates_into_fields() {
        let candidates = [
            example! { answer: "red", why: "warm" },
            example! { answer: "blue", why: "cool" },
        ];
        let completions = Completions::from_candidates(&candidates).expect("even fields");
        let names: Vec<&str> = completions.items().map(|(name, _)| name).collect();
        assert_eq!(names, ["answer", "why"]);
        assert_eq!(completions.get("why"), Some([json!("warm"), json!("cool")].as_slice()));
    }

    /// `completions[i]` is the i-th value of every field, as one prediction.
    #[test]
    fn it_reads_one_candidate_back_as_a_prediction() {
        let completions = two_candidates();
        assert_eq!(completions.at(1).expect("a candidate").get("answer"), Some(&json!("blue")));
        assert!(completions.at(2).is_none());
    }

    /// dspy asserts every field holds the same number of values; an uneven set is refused rather
    /// than silently truncated.
    #[test]
    fn it_refuses_fields_of_different_lengths() {
        let uneven = Completions::new([
            ("answer", vec![json!("red"), json!("blue")]),
            ("why", vec![json!("warm")]),
        ]);
        assert!(uneven.is_err());
        assert!(Completions::new(Vec::<(String, Vec<Value>)>::new()).is_ok());
    }
}
