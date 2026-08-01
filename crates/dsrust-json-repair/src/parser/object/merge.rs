//! A `[` where an object key should be, when the member before it was already a list.
//!
//! This is the shape a model produces when it writes a table one row per line and forgets that the
//! rows belong to one array. The continuation is read as more of the previous member, and the loose
//! values that were never wrapped are regrouped into rows of the width the existing rows agree on.

use crate::Result;
use crate::parser::Parser;
use crate::value::{Object, Value};

impl Parser {
    /// Answers whether the `[` was consumed as a continuation of the previous member.
    pub(crate) fn merge_object_array_continuation(&mut self, obj: &mut Object) -> Result<bool> {
        // An empty key is falsy in Python, so an object whose last key is `""` does not merge.
        let Some(previous_key) = obj
            .last_key()
            .filter(|key| !key.is_empty())
            .map(str::to_owned)
        else {
            return Ok(false);
        };
        if !matches!(obj.get(&previous_key), Some(Value::Array(_))) || self.strict {
            return Ok(false);
        }

        self.index += 1;
        let new_array = self.parse_array_items(None, "$", ']')?;
        if let Some(Value::Array(previous)) = obj.get_mut(&previous_key) {
            merge_rows(previous, new_array, self);
        }

        self.skip_whitespaces();
        if self.char_here() == Some(',') {
            self.index += 1;
        }
        self.skip_whitespaces();
        Ok(true)
    }
}

/// The row width the existing rows agree on, when they agree and it is not zero.
///
/// Python tests the width with `if expected_len:`, so rows of width zero fall through to the flat
/// append rather than being grouped into empty rows forever.
fn agreed_row_width(previous: &[Value]) -> Option<usize> {
    let mut widths = previous.iter().filter_map(|item| match item {
        Value::Array(row) => Some(row.len()),
        _ => None,
    });
    let first = widths.next()?;
    if !widths.all(|width| width == first) || first == 0 {
        return None;
    }
    Some(first)
}

fn merge_rows(previous: &mut Vec<Value>, new_array: Vec<Value>, parser: &Parser) {
    let Some(width) = agreed_row_width(previous) else {
        // No rows to match: a single wrapped row is unwrapped, anything else is appended flat.
        let flat = match new_array.len() == 1 && matches!(new_array.first(), Some(Value::Array(_)))
        {
            true => match new_array.into_iter().next() {
                Some(Value::Array(inner)) => inner,
                other => other.into_iter().collect(),
            },
            false => new_array,
        };
        previous.extend(flat);
        return;
    };

    let mut tail = Vec::new();
    while previous
        .last()
        .is_some_and(|item| !matches!(item, Value::Array(_)))
    {
        tail.push(previous.pop().expect("just checked there is one"));
    }
    if !tail.is_empty() {
        tail.reverse();
        if tail.len().is_multiple_of(width) {
            parser.log("While parsing an object we found row values without an inner array, grouping them into rows");
            for row in tail.chunks(width) {
                previous.push(Value::Array(row.to_vec()));
            }
        } else {
            previous.extend(tail);
        }
    }
    if new_array.is_empty() {
        return;
    }
    if new_array.iter().all(|item| matches!(item, Value::Array(_))) {
        parser.log(
            "While parsing an object we found additional rows, appending them without flattening",
        );
        previous.extend(new_array);
    } else {
        previous.push(Value::Array(new_array));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(values: &[i64]) -> Value {
        Value::Array(values.iter().map(|number| Value::Int(*number)).collect())
    }

    #[test]
    fn loose_values_are_regrouped_to_the_width_the_rows_agree_on() {
        let parser = Parser::new("", &crate::Repair::new(), None);
        let mut previous = vec![row(&[1, 2]), Value::Int(3), Value::Int(4)];
        merge_rows(&mut previous, vec![row(&[5, 6])], &parser);
        assert_eq!(previous, vec![row(&[1, 2]), row(&[3, 4]), row(&[5, 6])]);
    }

    #[test]
    fn a_tail_that_does_not_divide_evenly_stays_flat() {
        let parser = Parser::new("", &crate::Repair::new(), None);
        let mut previous = vec![row(&[1, 2]), Value::Int(3)];
        merge_rows(&mut previous, vec![], &parser);
        assert_eq!(previous, vec![row(&[1, 2]), Value::Int(3)]);
    }

    #[test]
    fn rows_of_width_zero_are_not_a_width() {
        assert_eq!(agreed_row_width(&[row(&[]), row(&[])]), None);
        assert_eq!(agreed_row_width(&[row(&[1]), row(&[1, 2])]), None);
        assert_eq!(agreed_row_width(&[row(&[1, 2]), Value::Int(9)]), Some(2));
    }
}
