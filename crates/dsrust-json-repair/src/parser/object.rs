//! `parse_object`: members until the closing brace, and the several ways a model loses one.
//!
//! Almost every line here is a recovery: a `:` before the key, a key with no `:` after it, a
//! duplicate key that means a *new* object rather than an overwrite, a `]` that says this object
//! belongs to an array nobody closed, a comma after the closing brace that carries more members.
//! What comes out is a `dict` whenever one can be salvaged and, when one cannot, the input is read
//! again as something else — see [`empty`].

pub(crate) mod empty;
pub(crate) mod merge;

use crate::parser::Parser;
use crate::parser::context::ContextValue;
use crate::schema::ObjectSchemaConfig;
use crate::value::{Object, Value};
use crate::{Error, Result, STRING_DELIMITERS, Schema};

impl Parser {
    pub(crate) fn parse_object(&mut self, schema: Option<Schema>, path: &str) -> Result<Value> {
        let mut obj = Object::new();
        let start_index = self.index;
        let parsing_object_value = self.context.is(ContextValue::ObjectValue);
        let (guided, schema, config) = crate::schema::resolve_parser_object_schema(
            self.schema_repairer.as_deref(),
            schema.as_ref(),
        )?;

        while let Some(char) = self.char_here() {
            if char == '}' {
                break;
            }
            if !self.object_member(
                &mut obj,
                guided,
                config.as_ref(),
                path,
                parsing_object_value,
            )? {
                break;
            }
        }
        self.index += 1;

        if let Some(repaired) =
            self.repair_empty_object_result(&obj, start_index, schema.as_ref(), path)?
        {
            return Ok(repaired);
        }
        self.complete_object_parse(obj, schema, config, path)
    }

    /// One `key: value`. Answers whether the object is still open afterwards.
    fn object_member(
        &mut self,
        obj: &mut Object,
        guided: bool,
        config: Option<&ObjectSchemaConfig>,
        path: &str,
        parsing_object_value: bool,
    ) -> Result<bool> {
        self.skip_whitespaces();
        if self.char_here() == Some(':') {
            self.log("While parsing an object we found a : before a key, ignoring");
            self.index += 1;
        }

        let (key, rollback_index) = self.parse_object_key(obj)?;
        if self.context.contains(ContextValue::Array)
            && obj.contains_key(&key)
            && !self.duplicate_key_keeps_the_object_open(rollback_index, parsing_object_value)?
        {
            return Ok(false);
        }

        self.skip_whitespaces();
        if matches!(self.char_here(), None | Some('}')) {
            return Ok(true);
        }
        self.skip_whitespaces();
        if self.char_here() != Some(':') {
            if self.strict {
                self.log(
                    "Missing ':' after key in strict mode while parsing object, raising an error",
                );
                return Err(Error::new(
                    "Missing ':' after key in strict mode while parsing object.",
                ));
            }
            self.log("While parsing an object we missed a : after a key");
        }
        self.index += 1;

        let (prop_schema, extra_schemas, drop_property) =
            self.resolve_object_property_schema(guided, config, &key)?;
        let key_path = format!("{path}.{key}");
        let mut value = self.parse_object_value(guided, prop_schema, &key_path)?;
        if guided {
            for extra in extra_schemas {
                value = self.repair_value(value, extra, &key_path)?;
            }
        }

        if !guided
            && value.is_empty_string()
            && self.strict
            && !self
                .get_char_at(-1)
                .is_some_and(|char| STRING_DELIMITERS.contains(&char))
        {
            self.log("Parsed value is empty in strict mode while parsing object, raising an error");
            return Err(Error::new(
                "Parsed value is empty in strict mode while parsing object.",
            ));
        }

        if !guided || !drop_property {
            obj.insert(key, value);
        } else {
            self.repairer_log("Dropped extra property not covered by schema", &key_path);
        }

        if matches!(self.char_here(), Some(',' | '\'' | '"')) {
            self.index += 1;
        }
        if self.char_here() == Some(']') && self.context.contains(ContextValue::Array) {
            self.log(
                "While parsing an object we found a closing array bracket, closing the object here and rolling back the index",
            );
            self.index -= 1;
            return Ok(false);
        }
        self.skip_whitespaces();
        Ok(true)
    }

    /// The key, and where it started — which a duplicate needs in order to close the object there.
    fn parse_object_key(&mut self, obj: &mut Object) -> Result<(String, usize)> {
        let mut key = String::new();
        let mut rollback_index = self.index;
        self.context.set(ContextValue::ObjectKey);
        let parsed = self.object_key_body(obj, &mut key, &mut rollback_index);
        self.context.reset();
        parsed?;
        Ok((key, rollback_index))
    }

    fn object_key_body(
        &mut self,
        obj: &mut Object,
        key: &mut String,
        rollback_index: &mut usize,
    ) -> Result<()> {
        while self.char_here().is_some() {
            *rollback_index = self.index;
            if self.char_here() == Some('[')
                && key.is_empty()
                && self.merge_object_array_continuation(obj)?
            {
                continue;
            }
            let raw_key = self.parse_string()?;
            let Value::Str(raw_key) = raw_key else {
                // Python asserts the key is a string here, and a fenced block inside a key is the
                // one way past it. An assertion is not a repair, so it ends the parse as it does
                // upstream rather than being papered over with a cast.
                return Err(Error::new("Object key did not parse as a string."));
            };
            *key = raw_key;
            if key.is_empty() {
                self.skip_whitespaces();
            }
            if !key.is_empty() || matches!(self.char_here(), Some(':' | '}')) {
                if key.is_empty() && self.strict {
                    self.log(
                        "Empty key found in strict mode while parsing object, raising an error",
                    );
                    return Err(Error::new(
                        "Empty key found in strict mode while parsing object.",
                    ));
                }
                break;
            }
        }
        Ok(())
    }

    /// A key already present. In strict mode that is an error; otherwise it usually means a second
    /// object began without anyone opening it, and the input is rewritten to say so.
    fn duplicate_key_keeps_the_object_open(
        &mut self,
        rollback_index: usize,
        parsing_object_value: bool,
    ) -> Result<bool> {
        if self.strict {
            self.log("Duplicate key found in strict mode while parsing object, raising an error");
            return Err(Error::new(
                "Duplicate key found in strict mode while parsing object.",
            ));
        }
        if parsing_object_value {
            return Ok(true);
        }
        if !self.should_split_duplicate_object(rollback_index) {
            self.log(
                "While parsing an object we found a duplicate key with a normal comma separator, keeping duplicate-key overwrite behavior",
            );
            return Ok(true);
        }
        self.log(
            "While parsing an object we found a duplicate key, closing the object here and rolling back the index",
        );
        // `rollback_index` is where a key started, which is always past the brace this object
        // opened on.
        self.index = rollback_index
            .checked_sub(1)
            .expect("an object key starts past the opening brace");
        self.json_str.insert(self.index + 1, '{');
        Ok(false)
    }

    /// A quoted key, a comma before it and a colon after it is an ordinary duplicate — the last
    /// value wins. Anything else is two objects run together.
    fn should_split_duplicate_object(&self, rollback_index: usize) -> bool {
        let mut lookback = rollback_index as isize - self.index as isize - 1;
        while self
            .get_char_at(lookback)
            .is_some_and(crate::pychar::is_space)
        {
            lookback -= 1;
        }
        let previous = self.get_char_at(lookback);
        let key_start_char = self.get_char_at(rollback_index as isize - self.index as isize);
        let next = self.get_char_at(self.scroll_whitespaces(0));
        !(key_start_char.is_some_and(|char| STRING_DELIMITERS.contains(&char))
            && previous == Some(',')
            && next == Some(':'))
    }

    fn parse_object_value(
        &mut self,
        guided: bool,
        prop_schema: Option<Schema>,
        key_path: &str,
    ) -> Result<Value> {
        self.context.set(ContextValue::ObjectValue);
        let parsed = self.object_value_body(guided, prop_schema, key_path);
        self.context.reset();
        parsed
    }

    fn object_value_body(
        &mut self,
        guided: bool,
        prop_schema: Option<Schema>,
        key_path: &str,
    ) -> Result<Value> {
        self.skip_whitespaces();
        if let Some(char @ (',' | '}')) = self.char_here() {
            self.log(&format!(
                "While parsing an object value we found a stray {char}, ignoring it"
            ));
            if guided {
                return self.repair_missing_value(prop_schema, key_path);
            }
            return Ok(Value::Str(String::new()));
        }
        if guided {
            return self.parse_json(prop_schema, key_path);
        }
        self.parse_json(None, "$")
    }

    /// After the closing brace: a stray one belonging to an enclosing container is stepped over,
    /// and a comma at the top level can carry members that were never inside the braces at all.
    fn complete_object_parse(
        &mut self,
        mut obj: Object,
        schema: Option<Schema>,
        config: Option<ObjectSchemaConfig>,
        path: &str,
    ) -> Result<Value> {
        if !self.context.empty {
            if self.char_here() == Some('}')
                && !self.context.is(ContextValue::ObjectKey)
                && !self.context.is(ContextValue::ObjectValue)
            {
                self.log("Found an extra closing brace that shouldn't be there, skipping it");
                self.index += 1;
            }
            return Ok(Value::Object(obj));
        }

        self.skip_whitespaces();
        if self.char_here() == Some(',') {
            self.index += 1;
            self.skip_whitespaces();
            if self
                .char_here()
                .is_some_and(|char| STRING_DELIMITERS.contains(&char))
                && !self.strict
            {
                self.log(
                    "Found a comma and string delimiter after object closing brace, checking for additional key-value pairs",
                );
                if let Value::Object(additional) = self.parse_object(schema.clone(), path)? {
                    obj.update(additional);
                }
            }
        }
        self.finalize_object(obj, config, path)
    }
}
