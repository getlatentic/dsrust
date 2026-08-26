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


def main() -> None:
    if dspy.__version__ != PINNED:
        raise SystemExit(f"expected dspy {PINNED}, found {dspy.__version__}")
    # convert_input_schema_to_tool_args returns (args, arg_types, arg_desc); the Rust side builds
    # `Tool.args`, which is the first of those.
    cases = [{"name": name, "input_schema": schema, "args": convert_input_schema_to_tool_args(schema)[0]} for name, schema in CASES.items()]
    fixture = {"source": f"dspy=={PINNED} adapters/types/tool.convert_input_schema_to_tool_args", "dspy_version": PINNED, "cases": cases}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
    print(f"  wrote {OUT.name}: {len(cases)} cases", file=sys.stderr)


if __name__ == "__main__":
    main()
