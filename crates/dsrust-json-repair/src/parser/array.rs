//! `parse_array`: values until the closing bracket, and several ways of not finding one.
//!
//! A `}` ends the array as surely as a `]` does, since a model that loses a bracket usually loses
//! it in the direction of the enclosing object. A quoted item followed by a colon is a member of
//! an object nobody opened, and is read as one. A run of separators between items is skipped
//! wholesale rather than counted, so `[1,,2]` has two items and `[1, ,2]` does too.

use crate::parser::Parser;
use crate::parser::context::ContextValue;
use crate::schema::ArraySchemaConfig;
use crate::value::Value;
use crate::{Result, STRING_DELIMITERS, Schema, pychar};

impl Parser {
    pub(crate) fn parse_array(
        &mut self,
        schema: Option<Schema>,
        path: &str,
        closing_delimiter: char,
    ) -> Result<Value> {
        Ok(Value::Array(self.parse_array_items(
            schema,
            path,
            closing_delimiter,
        )?))
    }

    pub(crate) fn parse_array_items(
        &mut self,
        schema: Option<Schema>,
        path: &str,
        closing_delimiter: char,
    ) -> Result<Vec<Value>> {
        let (guided, config) = crate::schema::resolve_parser_array_schema(
            self.schema_repairer.as_deref(),
            schema.as_ref(),
        )?;
        self.context.set(ContextValue::Array);
        let parsed = self.array_body(guided, config.as_ref(), path, closing_delimiter);
        self.context.reset();
        parsed
    }

    fn array_body(
        &mut self,
        guided: bool,
        config: Option<&ArraySchemaConfig>,
        path: &str,
        closing_delimiter: char,
    ) -> Result<Vec<Value>> {
        let salvage = guided && self.repairer_salvages();
        let mut items = Vec::new();
        self.skip_whitespaces();
        let mut idx = 0_usize;
        while let Some(char) = self.char_here() {
            if char == closing_delimiter || char == '}' {
                break;
            }
            let (item_schema, drop_item) = crate::schema::resolve_array_item_schema(config, idx)?;
            let item_path = format!("{path}[{idx}]");
            let active = guided && !drop_item && !salvage;
            let value = self.array_item(char, active, item_schema, &item_path)?;

            if value.is_strictly_empty()
                && !matches!(self.char_here(), Some(c) if c == closing_delimiter || c == ',')
            {
                self.index += 1;
            } else if value == Value::Str("...".into()) && self.get_char_at(-1) == Some('.') {
                self.log("While parsing an array, found a stray '...'; ignoring it");
            } else if !drop_item {
                items.push(value);
            } else if guided {
                self.repairer_log("Dropped extra array item not covered by schema", &item_path);
            }

            idx += 1;
            while let Some(char) = self.char_here() {
                if char == closing_delimiter || !(pychar::is_space(char) || char == ',') {
                    break;
                }
                self.index += 1;
            }
        }
        if self.char_here() != Some(closing_delimiter) {
            self.log(&format!(
                "While parsing an array we missed the closing {closing_delimiter}, ignoring it"
            ));
        }
        self.index += 1;
        Ok(items)
    }

    /// One item. A quoted run followed by a colon is a member of an object that was never opened.
    fn array_item(
        &mut self,
        char: char,
        active: bool,
        item_schema: Option<Schema>,
        item_path: &str,
    ) -> Result<Value> {
        if !STRING_DELIMITERS.contains(&char) {
            return if active {
                self.parse_json(item_schema, item_path)
            } else {
                self.parse_json(None, "$")
            };
        }
        let closing = self.skip_to_character(&[char], 1);
        let after = self.scroll_whitespaces(closing + 1);
        if self.get_char_at(after) == Some(':') {
            let value = if active {
                let value = self.parse_object(item_schema.clone(), item_path)?;
                self.repair_value(value, item_schema, item_path)?
            } else {
                self.parse_object(None, "$")?
            };
            return Ok(value);
        }
        let value = self.parse_string()?;
        if active {
            return self.repair_value(value, item_schema, item_path);
        }
        Ok(value)
    }
}
