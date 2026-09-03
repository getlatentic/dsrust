"""Record the message `jsonschema.validate` raises, keyword by keyword, by running python-jsonschema.

dspy checks a tool argument with `validate(instance=v, schema=self.args[k])` and raises
`ValueError(f"Arg {k} is invalid: {e.message}")`. `e` is `best_match` of every error the draft
2020-12 keywords yield, so the message depends on the keyword templates, the type checker, the
order errors are yielded in, and the relevance heuristic that picks among them.

Every case with a `rust_type` carries the schema schemars emits for that type, and the port checks
the two agree before it checks the message. Cases without one exercise a keyword a parameter type
does not produce on its own.

    .venv/bin/python scripts/generate_schema_message_fixture.py
"""

from __future__ import annotations

import json
import pathlib
import sys
from importlib.metadata import version

import jsonschema

OUT = pathlib.Path(__file__).parent.parent / "crates" / "dsrust" / "tests" / "conformance" / "react"

U32 = {"minimum": 0, "type": "integer"}
U8 = {"maximum": 255, "minimum": 0, "type": "integer"}
INNER = {
    "properties": {
        "name": {"title": "Name", "type": "string"},
        "count": {"maximum": 255, "minimum": 0, "title": "Count", "type": "integer"},
    },
    "required": ["name", "count"],
    "title": "Inner",
    "type": "object",
}
INNER_ITEM = {k: v for k, v in INNER.items() if k != "title"}
SHAPE = {
    "oneOf": [
        {
            "additionalProperties": False,
            "properties": {
                "Circle": {
                    "properties": {"radius": {"title": "Radius", "type": "number"}},
                    "required": ["radius"],
                    "title": "Circle",
                    "type": "object",
                }
            },
            "required": ["Circle"],
            "type": "object",
        },
        {
            "additionalProperties": False,
            "properties": {"Square": {"minimum": 0, "title": "Square", "type": "integer"}},
            "required": ["Square"],
            "type": "object",
        },
    ]
}
TYPED = {
    "u32": U32,
    "u8": U8,
    "f64": {"type": "number"},
    "bool": {"type": "boolean"},
    "String": {"type": "string"},
    "Option<u32>": {"anyOf": [U32, {"type": "null"}]},
    "Vec<String>": {"items": {"type": "string"}, "type": "array"},
    "Vec<u32>": {"items": U32, "type": "array"},
    "Inner": INNER,
    "Vec<Inner>": {"items": INNER_ITEM, "type": "array"},
    "Option<Inner>": {"anyOf": [INNER_ITEM, {"type": "null"}]},
    "Colour": {"enum": ["Red", "Green"], "title": "Colour", "type": "string"},
    "Shape": SHAPE,
    "(u32, String)": {"maxItems": 2, "minItems": 2, "prefixItems": [U32, {"type": "string"}], "type": "array"},
    "HashMap<String, u32>": {"additionalProperties": U32, "type": "object"},
}

# (label, rust type or None, schema override or None, instance)
CASES = [
    ("string_given_int", "String", None, 3),
    ("bool_given_string", "bool", None, "true"),
    ("number_given_string", "f64", None, "1.5"),
    ("integer_given_string", "u32", None, "three"),
    ("integer_given_null", "u32", None, None),
    ("integer_given_bool", "u32", None, True),
    ("integer_given_fraction", "u32", None, 3.5),
    ("integer_given_integral_float", "u32", None, 3.0),
    ("integer_below_minimum", "u32", None, -1),
    ("integer_above_maximum", "u8", None, 300),
    ("integer_valid", "u8", None, 7),
    ("option_given_string", "Option<u32>", None, "x"),
    ("option_given_negative", "Option<u32>", None, -1),
    ("option_given_null", "Option<u32>", None, None),
    ("array_given_object", "Vec<String>", None, {"a": 1}),
    ("array_item_wrong", "Vec<String>", None, ["a", 2, "c"]),
    ("array_two_items_wrong", "Vec<u32>", None, [1, "a", -1]),
    ("object_missing_required", "Inner", None, {"count": 1}),
    ("object_missing_and_wrong", "Inner", None, {"count": 300}),
    ("object_field_wrong", "Inner", None, {"name": "a", "count": "x"}),
    ("object_two_fields_wrong", "Inner", None, {"name": 1, "count": "x"}),
    ("object_valid", "Inner", None, {"name": "a", "count": 1}),
    ("nested_item_missing_field", "Vec<Inner>", None, [{"name": "a", "count": 1}, {"name": "b"}]),
    ("option_object_wrong_field", "Option<Inner>", None, {"name": "a", "count": "x"}),
    ("option_object_missing_field", "Option<Inner>", None, {"count": 1}),
    ("enum_not_a_member", "Colour", None, "Blue"),
    ("enum_wrong_type", "Colour", None, 1),
    ("one_of_neither", "Shape", None, {"Triangle": 1}),
    ("one_of_inner_missing", "Shape", None, {"Circle": {}}),
    ("one_of_valid", "Shape", None, {"Square": 2}),
    ("tuple_too_short", "(u32, String)", None, [1]),
    ("tuple_too_long", "(u32, String)", None, [1, "a", 2]),
    ("tuple_both_wrong", "(u32, String)", None, ["a", 1]),
    ("map_value_wrong", "HashMap<String, u32>", None, {"a": "x"}),
    ("map_two_values_wrong", "HashMap<String, u32>", None, {"b": "x", "a": -1}),
    ("const_wrong", None, {"const": "fixed"}, "other"),
    ("const_bool_vs_int", None, {"const": 1}, True),
    ("enum_bool_vs_int", None, {"enum": [0, 1]}, False),
    ("enum_int_vs_float", None, {"enum": [1, 2]}, 2.0),
    ("multiple_of_int", None, {"multipleOf": 3}, 7),
    ("multiple_of_float", None, {"multipleOf": 0.5}, 0.75),
    ("exclusive_minimum", None, {"exclusiveMinimum": 0}, 0),
    ("exclusive_maximum", None, {"exclusiveMaximum": 10}, 10.5),
    ("min_length_one", None, {"minLength": 1}, ""),
    ("min_length_more", None, {"minLength": 3}, "ab"),
    ("max_length_zero", None, {"maxLength": 0}, "a"),
    ("max_length_more", None, {"maxLength": 2}, "abc"),
    ("min_items_one", None, {"minItems": 1}, []),
    ("max_items_zero", None, {"maxItems": 0}, [1]),
    ("unique_items", None, {"uniqueItems": True}, [1, 2, 1]),
    ("unique_items_bool_vs_int", None, {"uniqueItems": True}, [1, True]),
    ("pattern", None, {"pattern": "^[a-z]+$"}, "ABC"),
    ("items_false", None, {"prefixItems": [{"type": "integer"}], "items": False}, [1, 2, 3]),
    ("items_false_one_extra", None, {"prefixItems": [{"type": "integer"}], "items": False}, [1, "x"]),
    ("additional_false", None, {"properties": {"a": {}}, "additionalProperties": False}, {"a": 1, "c": 2, "b": 3}),
    ("additional_false_one", None, {"properties": {"a": {}}, "additionalProperties": False}, {"a": 1, "z": 2}),
    ("additional_with_patterns", None, {"patternProperties": {"^x": {}}, "additionalProperties": False}, {"xa": 1, "b": 2}),
    ("pattern_property_wrong", None, {"patternProperties": {"^x": {"type": "integer"}}}, {"xa": "s"}),
    ("property_names", None, {"propertyNames": {"maxLength": 1}}, {"ab": 1}),
    ("min_properties", None, {"minProperties": 1}, {}),
    ("min_properties_more", None, {"minProperties": 2}, {"a": 1}),
    ("max_properties_zero", None, {"maxProperties": 0}, {"a": 1}),
    ("max_properties_more", None, {"maxProperties": 1}, {"a": 1, "b": 2}),
    ("dependent_required", None, {"dependentRequired": {"a": ["b", "c"]}}, {"a": 1}),
    ("dependent_schemas", None, {"dependentSchemas": {"a": {"required": ["b"]}}}, {"a": 1}),
    ("all_of", None, {"allOf": [{"type": "integer"}, {"minimum": 5}]}, 2),
    ("not", None, {"not": {"type": "string"}}, "s"),
    ("if_then", None, {"if": {"type": "integer"}, "then": {"minimum": 5}}, 2),
    ("if_else", None, {"if": {"type": "integer"}, "else": {"type": "null"}}, "s"),
    ("contains_none", None, {"contains": {"type": "string"}}, [1, 2]),
    ("contains_too_few", None, {"contains": {"type": "string"}, "minContains": 2}, ["a", 1]),
    ("contains_too_many", None, {"contains": {"type": "string"}, "maxContains": 1}, ["a", "b"]),
    ("ref_to_defs", None, {"$defs": {"n": {"type": "integer"}}, "$ref": "#/$defs/n"}, "s"),
    ("false_schema", None, False, 1),
    ("type_list", None, {"type": ["integer", "null"]}, "s"),
    ("float_repr", None, {"maximum": 0.5}, 1e16),
    ("nested_repr", None, {"enum": [{"a": [1, None, True]}]}, {"a": [1.5, "x"]}),
]


def main() -> None:
    recorded = []
    for label, rust_type, override, instance in CASES:
        schema = TYPED[rust_type] if rust_type else override
        try:
            jsonschema.validate(instance=instance, schema=schema)
            message = None
        except jsonschema.ValidationError as error:
            message = error.message
        entry = {"label": label, "schema": schema, "instance": instance, "message": message}
        if rust_type:
            entry["rust_type"] = rust_type
        recorded.append(entry)
        print(f"    {label}: {message!r}", file=sys.stderr)
    fixture = {
        "source": f"generated from jsonschema=={version('jsonschema')} via scripts/generate_schema_message_fixture.py",
        "jsonschema_version": version("jsonschema"),
        "cases": recorded,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "schema_messages.json"
    path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
    print(f"  wrote {path.name}: {len(recorded)} cases", file=sys.stderr)


if __name__ == "__main__":
    main()
