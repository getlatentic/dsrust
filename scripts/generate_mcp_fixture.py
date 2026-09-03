"""Record what dspy 3.3 makes of an MCP tool's input schema.

dspy supports MCP tools via `dspy.Tool.from_mcp_tool`, whose core is
`convert_input_schema_to_tool_args` (`dspy/adapters/types/tool.py`): it turns an MCP tool's input
JSON schema into the `Tool.args` map — each property under its name, `$ref`s resolved inline. This
pins that conversion so the Rust `mcp_tool_args` can assert byte equality in `tests/mcp_conformance.rs`,
the way the LM wire is pinned to dspy elsewhere.

    .dspy-venv-3.3/bin/python scripts/generate_mcp_fixture.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import dspy

from pins import require
from dspy.adapters.types.tool import convert_input_schema_to_tool_args
from dspy.utils.mcp import _convert_mcp_tool_result

# Read from the pin rather than written here: a generator that names its own
# version cannot follow a bump, and six of them refused to run at 3.3.0 for
# exactly that reason while claiming the pin had drifted.
PINNED = require("dspy")
OUT = pathlib.Path(__file__).parent.parent / "crates" / "dsrust" / "tests" / "conformance" / "react" / "mcp_tool_args.json"

CASES = {
    "flat": {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "the city"}, "days": {"type": "integer"}},
        "required": ["city"],
    },
    "no_properties": {"type": "object"},
    "empty_schema": {},
    "nested_ref": {
        "type": "object",
        "properties": {"location": {"$ref": "#/$defs/Location"}, "note": {"type": "string"}},
        "$defs": {"Location": {"type": "object", "properties": {"lat": {"type": "number"}, "lon": {"type": "number"}}}},
        "required": ["location"],
    },
    "ref_inside_array": {
        "type": "object",
        "properties": {"stops": {"type": "array", "items": {"$ref": "#/$defs/Stop"}}},
        "$defs": {"Stop": {"type": "object", "properties": {"name": {"type": "string"}}}},
    },
}


class Result:
    """A `CallToolResult` as far as `_convert_mcp_tool_result` reads one: content blocks, the error
    flag, and the structured content.

    The blocks are the SDK's own types, because upstream tells text from non-text with
    `isinstance(block, TextContent)` — a look-alike falls into the non-text arm and the recording
    would say a lone text block comes back as a list. The result fields around them are read by
    name (`_get_field`/`_field_is_set`), so a plain object carrying them is what dspy reads, and
    each SDK version's spelling can be recorded without two installs.
    """

    def __init__(self, fields: dict):
        from mcp.types import ImageContent, TextContent

        self.__dict__.update(fields)
        self.model_fields_set = set(fields)
        self.content = [
            TextContent(**block) if block.get("type") == "text" else ImageContent(**block)
            for block in fields.get("content", [])
        ]


# dspy 3.3.1's `result_mode`: `"text"` is the conversion it always did, `"structured"` hands back
# structured content whenever the server *set* it — null included — and falls back to the text.
# The v2 SDK renamed the fields, and upstream reads either name, so both spellings are recorded.
RESULT_CASES = {
    "one_text_block": {"content": [{"type": "text", "text": "Paris"}]},
    "several_text_blocks": {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
    # Spelled the way the v2 SDK dumps it, so the recorded input and the recorded output are the
    # same block: dspy hands non-text content straight back, and a spelling the model renames would
    # record the SDK's alias rather than anything dspy decides.
    "no_text_blocks": {"content": [{"type": "image", "data": "…", "mime_type": "image/png"}]},
    "error_camel": {"content": [{"type": "text", "text": "no such place"}], "isError": True},
    "error_snake": {"content": [{"type": "text", "text": "no such place"}], "is_error": True},
    "structured_camel": {"content": [{"type": "text", "text": '{"temp": 22}'}], "structuredContent": {"temp": 22}},
    "structured_snake": {"content": [{"type": "text", "text": '{"temp": 22}'}], "structured_content": {"temp": 22}},
    "structured_null": {"content": [{"type": "text", "text": "t"}], "structuredContent": None},
    "structured_unset": {"content": [{"type": "text", "text": "t"}]},
    "error_with_structured": {
        "content": [{"type": "text", "text": "boom"}],
        "is_error": True,
        "structured_content": {"x": 1},
    },
}


def plain(value):
    """A returned value as JSON. `"text"` mode hands back the *content objects* when no block is
    text, so the recording keeps them as the mapping the SDK dumps rather than as a repr."""
    if isinstance(value, list):
        return [plain(item) for item in value]
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    return value


def read(fields: dict, mode: str) -> dict:
    try:
        return {"ok": True, "value": plain(_convert_mcp_tool_result(Result(fields), result_mode=mode))}
    except Exception as error:
        return {"ok": False, "error": str(error)}


def main() -> None:
    if dspy.__version__ != PINNED:
        raise SystemExit(f"expected dspy {PINNED}, found {dspy.__version__}")
    # convert_input_schema_to_tool_args returns (args, arg_types, arg_desc); the Rust side builds
    # `Tool.args`, which is the first of those.
    cases = [{"name": name, "input_schema": schema, "args": convert_input_schema_to_tool_args(schema)[0]} for name, schema in CASES.items()]
    results = [
        {"name": name, "result": fields, "text": read(fields, "text"), "structured": read(fields, "structured")}
        for name, fields in RESULT_CASES.items()
    ]
    fixture = {
        "source": f"dspy=={PINNED} adapters/types/tool.convert_input_schema_to_tool_args and "
        "utils/mcp._convert_mcp_tool_result",
        "dspy_version": PINNED,
        "cases": cases,
        "results": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
    print(f"  wrote {OUT.name}: {len(cases)} schema cases, {len(results)} result cases", file=sys.stderr)


if __name__ == "__main__":
    main()
