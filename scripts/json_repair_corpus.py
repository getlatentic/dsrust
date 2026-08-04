"""The inputs `generate_json_repair_fixture.py` runs through `json_repair`.

Chosen by reading the library's branches rather than by imagining what a model emits: every group
below names a decision in the source, and the coverage floor in the generator is what stops the
list drifting away from them. The shapes a well-behaved model produces are here too, but they are
the least useful cases — they take the `json.loads` fast path and never reach a repair at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Case:
    name: str
    why: str
    text: str
    options: dict[str, object] = field(default_factory=dict)
    #: Set when Rust cannot follow Python here, naming which limit — the test asserts the
    #: divergence rather than the agreement, so closing it turns the test red.
    diverges: str | None = None
    #: Read through `load(fd)` rather than `loads(text)`, which is a different parse.
    from_file: bool = False


def case(name: str, why: str, text: str, diverges: str | None = None, **options: object) -> Case:
    return Case(name=name, why=why, text=text, options=options, diverges=diverges)


def file_case(name: str, why: str, text: str) -> Case:
    """A case read through `load(fd)` rather than `loads(text)`.

    The two are not the same call: upstream turns the suffix fast path off for file input, so a
    valid JSON value after a prefix goes through the repair parser instead of CPython's scanner.
    Measured over twenty thousand generated inputs, they disagree on 37.
    """
    return Case(name=name, why=why, text=text, options={}, from_file=True)


VALID = [
    case("valid_object", "the fast path: json.loads answers and no repair runs", '{"a": 1}'),
    case("valid_nested", "containers through the fast path", '{"a": [1, {"b": null}], "c": true}'),
    case("valid_float_vs_int", "7 and 7.0 are different answers", '{"i": 7, "f": 7.0}'),
    case("valid_big_int", "wider than a machine word, and exact", '{"n": 123456789012345678901234567890}'),
    case("valid_escapes", "the escapes CPython's scanner resolves", r'{"s": "a\tbé🙂"}'),
    case("valid_empty_containers", "an empty object is not a missing one", '{"a": {}, "b": []}'),
    case("valid_unicode_keys", "a reply in Chinese takes the isalnum branch", '{"答案": "北京"}'),
]

# `repair_json` opens with `json.loads`, so what CPython's scanner accepts decides whether any
# repair happens at all. This group is that boundary, and it matters more than it looks: the
# scanner is the one part of the port written without its source to hand — `json/decoder.py`
# describes the *pure-Python* fallback, and the C accelerator is what actually runs.
CPYTHON_SCANNER = [
    case("nan", "json.loads accepts NaN and serde does not", '{"n": NaN}'),
    case("infinity", "and Infinity, which JSON has no spelling for", '{"n": Infinity, "m": -Infinity}'),
    case("leading_zero", "01 is two tokens, so the fast path fails and the repair runs", "{[01]}"),
    case("bare_leading_zero", "and at the top level it is `Extra data`", "01"),
    case("plus_sign", "a number may not open with one", "+1"),
    case("bare_leading_dot", "nor with a dot", ".5"),
    case("trailing_dot", "nor end on one", "5."),
    case("float_overflows_to_infinity", "which json.loads reads without complaint", "1e999"),
    case("negative_overflow", "and in the other direction", "-1e999"),
    case("duplicate_key_last_wins", "a dict assignment, so the position is the first one's",
         '{"a": 1, "a": 2}'),
    case("empty_key", "an empty key is a key", '{"": 0}'),
    case("escaped_solidus", "the one escape JSON has that nothing needs", r'{"a": "\/"}'),
    case("underscore_in_unicode_escape", "the C scanner refuses it; the pure-Python "
         "`_decode_uXXXX` would take it, because it ends in `int(esc, 16)`", r'{"a": "\u1_23"}'),
    case("vertical_tab_is_not_json_whitespace", "the scanner takes four characters and this is "
         "not one of them, though `str.isspace()` says otherwise", "\x0b1"),
    case("extra_data", "two values at the top level", "1 2"),
    case("space_around_the_colon", "which is fine", '{"a" : 1}'),
    case("two_arrays", "and this is not", "[1][2]"),
]

ORDINARY_MALFORMATIONS = [
    case("trailing_comma", "the commonest one", '{"a": 1,}'),
    case("single_quotes", "Python's spelling of a string", "{'a': 'b'}"),
    case("unquoted_key", "a bare key, which reaches parse_string with missing_quotes", "{a: 1}"),
    case("missing_closing_brace", "the reply was cut off", '{"a": 1'),
    case("missing_closing_bracket", "the same, in an array", '{"a": [1, 2'),
    case("python_literals", "True/False/None reach parse_boolean_or_null", "{'a': True, 'b': False, 'c': None}"),
    case("missing_colon", "a key with no separator after it", '{"a" 1}'),
    case("colon_before_key", "a stray separator before the key", '{: "a": 1}'),
    case("stray_value_comma", "a member with no value at all", '{"a": , "b": 2}'),
    case("double_comma", "a run of separators inside an array", "[1,,2]"),
    case("prose_around", "the model explained itself first", 'Sure! Here you go: {"a": 1} — done'),
    case("code_fence", "a fenced block, which parse_json_llm_block unwraps", '```json\n{"a": 1}\n```'),
    case("code_fence_not_json", "fences that do not enclose JSON, so the fences stay text", '"```notjson```"'),
    case("fenced_value", "a fenced block as a quoted value, the only way into parse_json_llm_block",
         '{"a": "```json {"b": 1}```"}'),
    case("fenced_value_wrong_language", "fences the block reader declines, so the string goes on",
         '{"a": "```js {"b": 1}```"}'),
    # `_post_fence_container_starts_next_member` was the one ported function in `parse_string.py`
    # that no corpus ever entered — found by tracing the pin rather than by reading it. Reaching it
    # takes an unterminated object *value* that hits `}` with a fence right after, and a container
    # right after that: the question it answers is whether that container is the next member of the
    # object or part of the string. Both answers appear below, and where the value comes out the
    # same the repair log does not.
    case("fence_then_array_at_end", "the container ends the input, so it starts the next member",
         '{"a": text}```[1,2]'),
    case("fence_then_array_closed", "a closing fence after it, so it does not", '{"a": text}```[1,2]```'),
    case("fence_then_array_inside_string", "the string closes after the fence, keeping all of it",
         '{"a": text}```[1,2]```"}'),
    # The same lookahead scrolls *past comments* to find where the next member starts, and both of
    # its skip loops were cold: a comment has to sit between the fence and the container for either
    # to run. `lookahead.rs` carried 61 surviving mutants, sixteen of them here.
    case("fence_then_line_comment_then_array", "a `#` comment scrolled to its newline",
         '{"a": text}```# note\n[1,2]'),
    case("fence_then_block_comment_then_array", "and a `/* */` one scrolled to its terminator",
         '{"a": text}```/* note */[1,2]'),
    case("fence_then_unterminated_block_comment", "a block comment with no `*/`, which ends the "
         "scan at the end of the input rather than at the terminator",
         '{"a": text}```/* unterminated [1,2]'),
    case("fence_then_paren_at_end", "the parenthesised form of the same decision", '{"a": text}```(1)'),
    case("fence_then_paren_closed", "and its closed form, which answers the other way",
         '{"a": text}```(1)```'),
    # `_scroll_comment_prefixed_member_start` skips comments on the way to the next member, and the
    # corpus only ever reached it with `//`. Each kind is its own arm — deleting the `#` arm or the
    # `/* */` arm survived every test — and the quote-follows path is the caller where the answer
    # decides whether a mid-string quote closes the string.
    case("comment_hash_before_member", "a hash comment between the comma and the member, seen from "
         "inside an unterminated string", '{"a": "x" mid", # note\n "b": 1}'),
    case("comment_block_before_member", "the block form, whose close the scan must find",
         '{"a": "x" mid", /* note */ "b": 1}'),
    case("comment_line_at_line_start", "a newline immediately before the //, which pins where the "
         "line scan starts", '{"a": "x" mid",\n// note\n"b": 1}'),
    case("comment_block_unterminated", "a block comment that never closes, so the scan ends the "
         "input", '{"a": "x" mid", /* never closed "b": 1}'),
    case("comment_hash_at_end", "a hash comment as the last thing there is",
         '{"a": "x" mid", #'),
    case("fence_comma_comment_member", "the post-fence path: container, comma, comment, member",
         '{"a": text}```{"x": 1}, # note\n"b": 2}'),
    # `_starts_nested_inline_container` classifies a bracket inside the container after a fence:
    # nested, or prose that lets the container close early. In `[k,[X],m:1]` the classification
    # decides whether `m:1` is inside the container or a member of the outer object — one answer
    # keeps the reply a single object, the other splits it in two. X walks the next-char set the
    # rule reads: each literal initial, a digit, a quote, another opener, and prose as the contrast.
    case("fenced_inner_bracket_true", "t admits a nested container", '{"a":"x}``` [k,[true],m:1]\n","b":"y"}'),
    case("fenced_inner_bracket_false", "as does f", '{"a":"x}``` [k,[false],m:1]\n","b":"y"}'),
    case("fenced_inner_bracket_null", "and n", '{"a":"x}``` [k,[null],m:1]\n","b":"y"}'),
    case("fenced_inner_bracket_minus", "and a minus", '{"a":"x}``` [k,[-1],m:1]\n","b":"y"}'),
    case("fenced_inner_bracket_digit", "and a digit", '{"a":"x}``` [k,[9],m:1]\n","b":"y"}'),
    case("fenced_inner_bracket_quote", "and a quote", '{"a":"x}``` [k,["s"],m:1]\n","b":"y"}'),
    case("fenced_inner_bracket_opener", "and another opener", '{"a":"x}``` [k,[[1]],m:1]\n","b":"y"}'),
    case("fenced_inner_bracket_closer", "an immediate close counts too", '{"a":"x}``` [k,[],m:1]\n","b":"y"}'),
    case("fenced_inner_bracket_prose", "prose does not, and the container closes early",
         '{"a":"x}``` [k,[w],m:1]\n","b":"y"}'),
    case("fenced_inner_bracket_after_prose", "a bracket whose *previous* character is prose is not "
         "nested regardless of what follows", '{"a":"x}``` [a [1]],m:1]\n","b":"y"}'),
    case("fenced_inner_paren", "the parenthesis arm of the same set", '{"a":"x}``` [k,(1),m:1]\n","b":"y"}'),
    case("fenced_inner_brace_after_comma", "a brace after a comma is not nested even holding a "
         "bare key, since only a colon admits one", '{"a":"x}``` {k:1,{j:2}}\n","b":"y"}'),
    case("fenced_inner_brace_after_colon", "after a colon it is", '{"a":"x}``` {k:{j:2},m:3}\n","b":"y"}'),
    case("fenced_inner_brace_quote", "a brace holding a quoted key is nested either way",
         '{"a":"x}``` {k:{"q":2}}\n","b":"y"}'),
    case("fenced_inner_brace_empty", "as is an empty one", '{"a":"x}``` {k:{}}\n","b":"y"}'),
    case("bare_word_value", "an unquoted value", "{a: hello}"),
    case("array_of_objects", "the shape dspy sees when a model wraps its answer", '[{"a": 1}]'),
]

ASYMMETRIC_QUOTES = [
    # Every disagreement the differential fuzzer found before this port came from this group.
    case("fuzz_shape", "the exact shape the fuzzer refused, from backlog `json-repair-port`",
         '{answer: "{}", "unknown: "7", reasoning": "[]"'),
    case("key_missing_open_quote", "a key whose opening quote is gone", '{answer": "Paris"}'),
    case("key_missing_close_quote", "and one whose closing quote is", '{"answer: "Paris"}'),
    case("value_missing_close_quote", "a value that never closes, then another member", '{"a": "one, "b": "two"}'),
    case("quote_inside_value", "a quote in the middle of a value that is not the end of it",
         '{"a": "he said "hi" to me", "b": 2}'),
    case("quote_then_colon", "the quote opens the next key rather than closing this value",
         '{"a": "one" "b": "two"}'),
    case("array_even_delimiters", "an even count of quotes before the bracket keeps them inside",
         '["a "b" c"]'),
    case("array_odd_delimiters", "an odd count does not", '["a "b c"]'),
    case("unquoted_value_with_comma", "a comma inside an unquoted value, which is prose not a member",
         '{"a": one, two, "b": 3}'),
    case("comma_then_bare_key", "a comma followed by something that really is the next member",
         '{"a": one, b: 2}'),
    case("comma_then_bare_key_no_value", "a bare key with nothing recoverable after it stays in the string",
         '{"a": one, floof: explanation'),
    case("brace_inside_value", "a balanced object inside an unterminated value belongs to it",
         '{"a": text {"k": 1} more"}'),
    case("brace_closes_object", "and an unbalanced one closes the object", '{"a": text}'),
    case("regex_character_class", "a bare quote inside [...] is a character class, not a delimiter",
         '{"pattern": "[\\"\']+", "b": 1}'),
    case("doubled_opening_quote", "two quotes where one was meant", '{""a"": 1}'),
    case("doubled_quote_empty", "two quotes that really are an empty value", '{"a": "", "b": 1}'),
    case("smart_quotes", "the curly pair, which opens and closes differently", "{“a”: “b”}"),
    case("low_smart_quote", "„ … ” is a quote pair no other rule knows about", '{"a": "say „hi" there”"}'),
    # Both found by scripts/fuzz_json_repair.py, seed 0. The misplaced quote steps the cursor
    # *back* onto the character before it, and where the string is then judged to have ended
    # decides whether the next key starts on that character or one past it — so the key comes out
    # as `s` rather than `Paris`.
    case("unquoted_value_ends_before_the_next_key", "the cursor steps back onto the character "
         "before a quote that opened the next key", '{"a":Paris", \'b\' :1, ": {}}'),
    case("unquoted_value_ends_before_a_backslash_run", "the same step-back, with the run of "
         "backslashes that made it visible", '["answer", {"北京""a", a": he said "hi"\\\\"}, "Paris"'),
]

ESCAPES = [
    case("stray_escape", "a backslash before a character that is not an escape", r'{"a": "b\qc"}'),
    case("hex_escape", r"\x41, which JSON has no escape for", r'{"a": "\x41"}'),
    case("unicode_escape", "a \\u escape resolved by the repair parser rather than the scanner",
         '{"a": "\\u00e9" '),
    case("lone_surrogate", "a Rust char cannot hold one; the crate substitutes U+FFFD",
         r'{"a": "\ud800" ', diverges="lone-surrogate"),
    case("backslash_run", "an even run halves", r'{"a": "b\\\\c"'),
    case("escaped_delimiter", "a delimiter escaped that had no business being", r"""{"a": "b\'c" """),
    case("escaped_object_keys", "keys whose quotes arrived escaped, reparsed as an object",
         r'{\"a\": 1, \"b\": 2}'),
    case("trailing_backslash", "the reply stopped on a backslash", r'{"a": "b\\'),
    case("trailing_backslash_streaming", "and the same under stream_stable", r'{"a": "b\\', stream_stable=True),
]

COMMENTS = [
    case("hash_comment", "a line comment between members", '{"a": 1, # note\n "b": 2}'),
    case("slash_comment", "the other line comment", '{"a": 1, // note\n "b": 2}'),
    case("block_comment", "a block comment", '{"a": /* note */ 1}'),
    case("unclosed_block_comment", "one that never closes", '{"a": /* note 1}'),
    case("comment_holding_a_colon", "a colon inside a comment is not a member separator",
         "{# note: here\n}"),
    case("top_level_comments", "a run of them before the value", "# one\n// two\n{\"a\": 1}"),
    case("comment_terminated_by_structure", "a comment that runs into the closing bracket", "[1, # note]"),
]

CONTAINERS_AND_SETS = [
    case("set_like_object", "an object with no separators is read as an array", '{"a", "b"}'),
    case("empty_object_with_junk", "an object that consumed characters and produced nothing", "{ x }"),
    case("duplicate_key_splits", "a duplicate key means a second object, not an overwrite",
         '[{"a": 1 "a": 2}]'),
    case("duplicate_key_overwrites", "a duplicate key with an ordinary comma keeps the last value",
         '[{"a": 1, "a": 2}]'),
    case("object_closed_by_bracket", "a `]` says the object belongs to an array", '[{"a": 1]'),
    case("object_closed_by_bracket_mid_array", "and the cursor rolls back onto it, so what follows "
         "is read as the next item", '[{"a": 1], {"b": 2}]'),
    case("object_closed_by_bracket_then_value", "the same rollback with a bare value after it",
         '[{"a": 1], 2]'),
    case("extra_closing_brace", "one brace too many", '{"a": 1}}'),
    case("members_after_brace", "a comma after the closing brace carrying more members",
         '{"a": 1}, "b": 2'),
    case("array_row_continuation", "rows written one array per line, merged into the first member",
         '{"rows": [[1, 2]\n[3, 4]]}'),
    case("array_row_regrouping", "loose values regrouped to the width the rows agree on",
         '{"rows": [[1, 2], 3, 4\n[5, 6]]}'),
    case("string_then_colon_in_array", "a quoted item followed by a colon is an object nobody opened",
         '["a": 1]'),
    case("stray_ellipsis", "a `...` the model left in an array", "[1, ..., 2]"),
    case("multiple_top_level", "two values, gathered into a list", '{"a": 1} {"b": 2}'),
    case("repeated_same_shape", "the same shape twice is an update, and the newest wins",
         '{"a": 1} {"a": 2}'),
    case("comma_separated_top_level", "and with a comma between them both survive", '{"a": 1}, {"a": 2}'),
]

TUPLES = [
    case("tuple", "a Python tuple literal", "(1, 2)"),
    case("grouped_value", "a single grouped value is not a tuple", "(1)"),
    case("empty_tuple", "empty brackets are", "()"),
    case("tuple_in_object", "one inside an object, where the test is permissive", '{"a": (1, 2)}'),
    # Both parenthesis tests walk the text themselves, tracking quotes, backslashes and every
    # bracket depth, and the corpus reached neither loop: every parenthesised case here held bare
    # numbers. `parenthesized.rs` carried 52 surviving mutants against that, twenty-one in
    # `Nesting::open_or_close` alone, where whole match arms delete unnoticed.
    case("tuple_of_quoted", "a quote inside the brackets, which is the whole quoting loop",
         "('a', 'b')"),
    case("tuple_with_an_escaped_quote", "and a backslash before one, where the parity decides "
         "whether the string ends", '("a\\", b", 2)'),
    case("tuple_nested", "a tuple inside a tuple, so the parenthesis depth is not just 0 or 1",
         "((1, 2), 3)"),
    case("tuple_holding_an_array", "square brackets tracked separately from parentheses",
         "(1, [2, 3])"),
    case("tuple_holding_an_object", "and braces from both", '({"a": 1}, 2)'),
    case("tuple_of_containers", "each depth counter moving at once", "([1], {2: 3})"),
    # The lookahead refuses a `(` that starts prose, and reads four characters for `true`/`null`
    # and five for `false` to tell a grouped literal from one.
    case("grouped_true", "a grouped literal, which the four-character read admits", "(true)"),
    case("grouped_false", "the five-character one", "(false)"),
    case("parenthesis_starts_prose", "words after the bracket, which is not a value", "(a for a in b)"),
    # A comma at the top level of the brackets *is* what makes it a tuple, so each of these hangs on
    # one depth counter being tracked: forget the opener and the inner comma reads as top level, and
    # a grouped container becomes a tuple holding it. Nothing else in the corpus can tell those
    # apart, which is why every arm of `Nesting::open_or_close` could be deleted unnoticed.
    case("grouped_array", "one list in brackets is the list, not a tuple holding it", "([1, 2])"),
    case("grouped_object", "and one object is the object", '({"a": 1, "b": 2})'),
    case("grouped_tuple", "a tuple inside brackets, where the counter is parentheses", "((1, 2))"),
    case("array_then_comma", "a top-level comma after a bracketed one, which does make a tuple",
         "([1, 2], 3)"),
    case("comma_inside_a_string", "quoted, so it is text rather than a separator", '("a, b")'),
    case("comma_inside_a_single_quoted_string", "the same through the other delimiter", "('a, b')"),
    # A closer with no opener, which is what the `> 0` guards on each closing arm are for.
    case("stray_close_bracket", "a `]` with nothing open, which must not take the count below zero",
         "(] 1, 2)"),
    case("stray_close_brace", "and a `}`", "(} 1, 2)"),
    case("unclosed_array_in_parens", "a bracket that never closes before the parenthesis does",
         "([1, 2)"),
    case("prose_parenthesis", "an aside, which must not swallow the JSON after it",
         'the answer (see below): {"a": 1}'),
    case("parenthesis_own_line", "brackets alone on their line do open a value", "\n(1, 2)\n"),
]

NUMBERS = [
    case("number_with_underscore", "digit groups, and the rollback that ignores them", "{a: 1_000}"),
    case("number_with_comma", "a thousands separator makes it a string", "{a: 1,234}"),
    case("fraction", "a slash makes it a string too", "{a: 3/4}"),
    case("trailing_minus", "a number ending on a character a number cannot end on", "{a: 1-}"),
    case("number_then_letters", "a run that turns out to be a word", "{a: 12abc}"),
    case("leading_dot", "a value starting with a dot", "{a: .5}"),
    case("exponent", "an exponent, which makes it a float", "{a: 1e3}"),
    case("negative_zero", "the sign survives", "{a: -0.0}"),
    case("huge_exponent", "a float that overflows to infinity", "{a: 1e400}"),
]

STRICT = [
    case("strict_valid", "strict mode leaves valid JSON alone", '{"a": 1}', strict=True),
    case("strict_duplicate_key", "and refuses a duplicate", '[{"a": 1, "a": 2}]', strict=True),
    case("strict_missing_colon", "and a missing separator", '{"a" 1}', strict=True),
    case("strict_empty_key", "and an empty key", '{"": 1}', strict=True),
    case("strict_empty_value", "and an empty value", '{"a": }', strict=True),
    case("strict_multiple_top_level", "and a second top-level element", '{"a": 1} {"b": 2}', strict=True),
    case("strict_doubled_quotes", "and doubled quotes", '{""a"": 1}', strict=True),
]

OPTIONS = [
    case("skip_json_loads_valid", "the whole-input check skipped, so the parser reads valid JSON",
         '{"a": 1}', skip_json_loads=True),
    case("skip_json_loads_suffix", "and the suffix fast path still applies after a prefix",
         'text {"a": 1}', skip_json_loads=True),
    case("stream_stable_partial", "a value still arriving keeps its trailing whitespace",
         '{"a": "val ', stream_stable=True),
    case("stream_stable_newline", "and its trailing newline", '{"a": "val\\n', stream_stable=True),
]

EDGES = [
    case("empty_input", "nothing at all", ""),
    case("whitespace_only", "only whitespace", "   \n\t"),
    case("prose_only", "a reply with no JSON in it", "I could not answer that."),
    case("brace_only", "an opening brace and nothing else", "{"),
    case("bracket_only", "an opening bracket and nothing else", "["),
    case("cpython_space", "a file separator, which CPython calls a space and Rust does not",
         "{a:\x1c1}"),
    case("nbsp_between", "a non-breaking space, which both call one", "{\xa0\"a\"\xa0: 1}"),
    case("control_char_in_string", "a raw control character, which the fast path refuses",
         '{"a": "b\x01c"}'),
]

#: The three shapes a differential run found `loads(text)` and `load(fd)` disagreeing on, plus the
#: ordinary case that proves the file path works at all.
FILE_INPUT = [
    file_case("file_plain", "an ordinary file, read the way `load` reads one", '{"a": 1}'),
    file_case("file_prefix_then_value", "a prefix and then valid JSON, which `loads` decodes with "
              "CPython's scanner and `load` repairs instead", '```json\n["a\\n"] done'),
    file_case("file_nan_after_comment", "the same split, where the repair parser reads NaN as text",
              '# note\n[NaN, "[a-z"]+", "_id"]'),
    file_case("file_trailing_brace", "and where it keeps a trailing brace inside the string",
              '# note\n["true story\\\\"]}'),
]

CASES: list[Case] = [
    *VALID,
    *CPYTHON_SCANNER,
    *ORDINARY_MALFORMATIONS,
    *ASYMMETRIC_QUOTES,
    *ESCAPES,
    *COMMENTS,
    *CONTAINERS_AND_SETS,
    *TUPLES,
    *NUMBERS,
    *STRICT,
    *OPTIONS,
    *EDGES,
    *FILE_INPUT,
]
