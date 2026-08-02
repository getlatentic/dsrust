"""The schema-guided cases `generate_json_repair_schema_fixture.py` runs.

One group per decision in `schema_repair.py` and `parser_schema.py`, named for the decision rather
than for the shape of the input. The generator refuses to write a fixture whose cases between them
skip a keyword named there, which is what keeps this list from drifting into whatever still passes.
"""

from __future__ import annotations


def case(name: str, why: str, schema: object, text: str, mode: str = "standard") -> dict:
    return {"name": name, "why": why, "schema": schema, "input": text, "mode": mode}


COERCION = [
    case("number_to_string", "a number where a string was declared", {"type": "string"}, "7"),
    case("string_to_integer", "and the reverse", {"type": "integer"}, '"7"'),
    case("float_to_integer", "a whole float narrows", {"type": "integer"}, "7.0"),
    case("float_that_will_not_narrow", "and a fractional one does not", {"type": "integer"}, "7.5"),
    case("string_to_number", "a decimal in quotes", {"type": "number"}, '"1.5"'),
    case("yes_is_true", "one of the ten spellings a boolean takes", {"type": "boolean"}, '"yes"'),
    case("bool_is_not_an_integer", "even though Python says True == 1", {"type": "integer"}, "true"),
    case("null_type", "the only value a null schema takes", {"type": "null"}, "null"),
    case("unsupported_type", "a type the coercions have no rule for", {"type": "date"}, '"today"'),
]

OBJECTS = [
    case("required_present", "the ordinary path",
         {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}, '{a: "1"}'),
    case("required_missing", "a required property with nothing to fill it",
         {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}, "{}"),
    # There are two default-insertion paths and `{}` reaches neither: it is valid JSON *and* valid
    # against the schema, so the fast path hands it back untouched. Each case below is shaped to
    # reach one of them — the parser's `_finalize_object`, and the repairer's `_repair_object`.
    case("default_inserted_while_parsing", "the parser fills an absent optional",
         {"type": "object", "properties": {"a": {"type": "integer", "default": 3}}}, "{b: 1"),
    case("default_inserted_while_repairing", "and so does the repair pass over a valid-JSON value "
         "the schema rejected",
         {"type": "object", "properties": {"a": {"type": "integer", "default": 3},
                                           "b": {"type": "integer"}}, "required": ["b"]},
         '{"b": "1"}'),
    case("value_missing_takes_default", "a member with no value at all takes it too",
         {"type": "object", "properties": {"a": {"type": "integer", "default": 3}}}, '{"a": }'),
    case("additional_properties_false", "a key the schema forbids is dropped",
         {"type": "object", "properties": {"a": {"type": "integer"}}, "additionalProperties": False},
         '{a: 1, b: 2}'),
    case("additional_properties_schema", "and one it constrains is coerced",
         {"type": "object", "properties": {}, "additionalProperties": {"type": "string"}}, "{a: 1}"),
    case("pattern_properties", "a key matched by an anchored literal",
         {"type": "object", "patternProperties": {"^id": {"type": "string"}}}, "{id_x: 7}"),
    case("pattern_properties_unsupported", "a real regex, which is skipped rather than matched",
         {"type": "object", "patternProperties": {"^a.*z$": {"type": "string"}}}, "{abcz: 7}"),
    case("min_properties", "an object that does not carry enough",
         {"type": "object", "minProperties": 2, "properties": {"a": {"type": "integer"}}}, "{a: 1}"),
    case("property_schema_is_null", "`properties.get(key)` cannot tell an absent property from "
         "one stored as null, and the repair pass reaches `.get` on it",
         {"type": "object", "properties": {"a": None}}, "{a: 1"),
    case("property_schema_is_null_and_required", "the same, where the salvage fill skips it and "
         "the required check still refuses",
         {"type": "object", "properties": {"a": None}, "required": ["a"]}, "{b: 1"),
    case("object_from_json_string", "a value that is an object inside quotes",
         {"type": "object", "properties": {"a": {"type": "integer"}}}, '"{\\"a\\": 1}"'),
]

ARRAYS = [
    case("items_schema", "one schema for every item",
         {"type": "array", "items": {"type": "integer"}}, '["1", "2"]'),
    case("items_tuple", "a schema per position",
         {"type": "array", "items": [{"type": "string"}, {"type": "integer"}]}, "[1, '2']"),
    case("additional_items_false", "a tuple that forbids the rest",
         {"type": "array", "items": [{"type": "string"}], "additionalItems": False}, "['a', 'b']"),
    case("additional_items_schema", "and one that constrains them",
         {"type": "array", "items": [{"type": "string"}], "additionalItems": {"type": "integer"}},
         "['a', '2']"),
    case("min_items", "an array that is too short", {"type": "array", "minItems": 3}, "[1, 2"),
    case("wrapped_in_array", "a scalar where an array was declared",
         {"type": "array", "items": {"type": "integer"}}, "7"),
    case("array_from_json_string", "an array inside quotes",
         {"type": "array", "items": {"type": "integer"}}, '"[1, 2]"'),
]

UNIONS = [
    case("one_of_second_branch", "the first branch fails and the second takes it",
         {"oneOf": [{"type": "integer"}, {"type": "string"}]}, '"abc"'),
    case("any_of_no_branch", "neither branch takes it",
         {"anyOf": [{"type": "integer"}, {"type": "boolean"}]}, '"abc"'),
    case("all_of", "every subschema applied in turn",
         {"allOf": [{"type": "string"}, {"enum": ["7"]}]}, "7"),
    case("type_union", "a `type` that lists several", {"type": ["integer", "string"]}, '"abc"'),
    case("type_union_first_wins", "and the first that fits wins",
         {"type": ["integer", "string"]}, '"12"'),
    # `_repair_union` catches plain `ValueError`, and `SchemaDefinitionError` is one — so a broken
    # subschema is *not* re-raised here the way it is inside an array or a list mapping. The port
    # had it the other way round until this case existed.
    # The input has to be *malformed*, or the whole-input fast path asks the validator about the
    # union before the repairer ever gets to the branches — which is what the first attempt at this
    # case did, and it proved nothing.
    case("one_of_with_a_broken_branch", "a subschema the repairer cannot read is a branch that "
         "failed, not an error",
         {"type": "object", "properties": {"a": {"oneOf": [{"type": "date"}, {"type": "string"}]}}},
         "{a: 7"),
]

ENUMS_AND_REFS = [
    case("enum_match", "a value that is one of them", {"enum": ["a", "b"]}, "'a'"),
    case("enum_miss", "and one that is not", {"enum": ["a", "b"]}, "'c'"),
    case("const_match", "a fixed value", {"const": 5}, "5"),
    case("const_miss", "and the wrong one", {"const": 5}, "6"),
    case("enum_fills_a_missing_value", "the first enum value stands in for nothing at all",
         {"type": "object", "properties": {"a": {"enum": ["x", "y"]}}}, '{"a": }'),
    # Malformed schemas, which say more about the translation than well-formed ones do: each of
    # these reaches a branch where upstream's `if expected_type is None` or `if not enum_values`
    # decides, and a `_` arm in their place answers differently.
    case("type_present_but_not_a_string", "blocks the inference from the schema's shape, so "
         "nothing is filled — a `_` arm would read `properties` and fill `{}`",
         {"type": "object", "properties": {"a": {"type": 7, "properties": {}}}}, '{"a": }'),
    case("enum_present_but_empty", "no first value to take",
         {"type": "object", "properties": {"a": {"enum": []}}}, '{"a": }'),
    case("enum_present_but_a_string", "upstream subscripts whatever `enum` holds, so a string "
         "yields its first character", {"type": "object", "properties": {"a": {"enum": "abc"}}},
         '{"a": }'),
    case("ref_resolved", "a `$ref` into the root schema",
         {"type": "object", "properties": {"a": {"$ref": "#/$defs/small"}},
          "$defs": {"small": {"type": "integer"}}}, '{a: "4"}'),
    case("ref_unresolvable", "a pointer to nothing",
         {"type": "object", "properties": {"a": {"$ref": "#/$defs/missing"}}}, "{a: 1}"),
    case("ref_circular", "a pointer that comes back to itself",
         {"type": "object", "properties": {"a": {"$ref": "#/$defs/loop"}},
          "$defs": {"loop": {"$ref": "#/$defs/loop"}}}, "{a: 1"),
    case("ref_circular_pair", "and a two-step cycle, which identity catches and a ref-name set "
         "catches for a different reason",
         {"type": "object", "properties": {"a": {"$ref": "#/$defs/x"}},
          "$defs": {"x": {"$ref": "#/$defs/y"}, "y": {"$ref": "#/$defs/x"}}}, "{a: 1"),
    case("ref_not_a_string", "`$ref` present but a number",
         {"type": "object", "properties": {"a": {"$ref": 7}}}, "{a: 1"),
    case("ref_to_a_boolean_schema", "a pointer at `true`, which resolves to the schema that "
         "accepts anything", {"type": "object", "properties": {"a": {"$ref": "#/$defs/any"}},
                              "$defs": {"any": True}}, "{a: 1"),
    case("ref_chain", "three hops to a real schema",
         {"type": "object", "properties": {"a": {"$ref": "#/$defs/one"}},
          "$defs": {"one": {"$ref": "#/$defs/two"}, "two": {"$ref": "#/$defs/three"},
                    "three": {"type": "string"}}}, "{a: 1"),
    case("ref_escaped_pointer", "a pointer whose segment holds `~1` and `~0`",
         {"type": "object", "properties": {"a": {"$ref": "#/$defs/a~1b~0c"}},
          "$defs": {"a/b~c": {"type": "string"}}}, "{a: 1"),
    case("ref_not_local", "a pointer out of the document",
         {"type": "object", "properties": {"a": {"$ref": "http://x/y"}}}, "{a: 1}"),
    case("boolean_schema_true", "the schema that accepts anything", True, "{a: 1}"),
    case("boolean_schema_false", "and the one that accepts nothing", False, "{a: 1}"),
]

SALVAGE = [
    case("salvage_drops_a_bad_item", "an item that will not fit is dropped rather than fatal",
         {"type": "array", "items": {"type": "integer"}}, '[1, "x", 3]', mode="salvage"),
    case("salvage_does_not_drop_a_broken_schema", "an item schema the repairer cannot *read* is "
         "re-raised instead, which is the one place `SchemaDefinitionError` is told apart from an "
         "ordinary refusal", {"type": "array", "items": {"type": "date"}}, "[1, 2", mode="salvage"),
    case("salvage_fills_required", "a required property filled from its default",
         {"type": "object", "properties": {"a": {"type": "integer", "default": 9}}, "required": ["a"]},
         "{}", mode="salvage"),
    case("salvage_type_not_a_string", "the same `is None` guard on the salvage path: nothing is "
         "filled, so the required property is still missing",
         {"type": "object", "properties": {"a": {"type": 7, "items": {}}}, "required": ["a"]},
         "{b: 1", mode="salvage"),
    case("salvage_fills_required_while_parsing", "the same fill, reached through the parser rather "
         "than the repair pass — `_finalize_object` skips a *required* key, leaving it to salvage",
         {"type": "object", "properties": {"a": {"type": "integer", "default": 9},
                                           "b": {"type": "integer"}}, "required": ["a"]},
         "{b: 1", mode="salvage"),
    case("salvage_maps_a_list", "a list mapped onto the properties in order",
         {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "string"}}},
         "[1, 'x']", mode="salvage"),
    case("salvage_unwraps_the_root", "a single-item root array unwrapped",
         {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}},
         '[{"a": 1, "b": 2}]', mode="salvage"),
    case("salvage_repairs_a_json_string", "a malformed object inside quotes, repaired first",
         {"type": "object", "properties": {"a": {"type": "integer"}}}, "\"{a: 1\"", mode="salvage"),
    case("salvage_set_like_object", "set-like members read as keys with null values",
         {"type": "object", "properties": {"a": {}, "b": {}}}, '{"a", "b"}', mode="salvage"),
]

#: Every keyword `json_repair` reads out of a schema, each given a value of the wrong type.
#:
#: This is a bug *class*, not a list of cases. Upstream reaches for a keyword with `schema.get(key)`
#: and then asks `is None`, `isinstance(..., list)` or `if not ...` — three different questions, and
#: a Rust `match` arm that answers "absent or odd" as one thing gets all three wrong in the same
#: direction. Three such arms were found by reading; these are what would have found them without.
WRONG_TYPES = [
    case(f"wrong_type_{keyword}_{label.replace(chr(32), chr(95))}",
         f"`{keyword}` present but {label}, which upstream tells apart from absent",
         {"type": "object", "properties": {"a": {keyword: bad, "type": "integer"}}},
         '{"a": }')
    for keyword in ("enum", "const", "default", "items", "additionalItems", "properties",
                    "patternProperties", "additionalProperties", "required", "minItems",
                    "minProperties", "oneOf", "anyOf", "allOf")
    for label, bad in (("a number", 7), ("a string", "x"), ("null", None), ("an empty list", []))
] + [
    case(f"wrong_type_root_type_{label.replace(chr(32), chr(95))}", f"the root `type` is {label}",
         {"type": bad, "properties": {"a": {"type": "integer"}}}, '{"a": 1}')
    for label, bad in (("a number", 7), ("null", None), ("an empty list", []), ("a nested list", [[]]))
]

CASES: list[dict] = [*WRONG_TYPES, *COERCION, *OBJECTS, *ARRAYS, *UNIONS, *ENUMS_AND_REFS, *SALVAGE]
