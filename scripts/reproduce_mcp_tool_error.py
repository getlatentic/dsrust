#!/usr/bin/env python3
"""`upstream_test_mcp.py::test_convert_mcp_tool` fails against this port, and against dspy itself.

The test asks an MCP server for a tool that raises `ValueError("error!")` and expects the client
to see that message:

    with pytest.raises(RuntimeError, match="error!"):
        await error_tool.acall()

`mcp` 2.1.1 does not send it. `mcp/server/mcpserver/tools/base.py` catches whatever a tool raised
and re-raises `UnexpectedToolError(f"Error executing tool {self.name}")` from it, so the text that
reaches the client names the tool and drops the cause. dspy then reports what it was given:
`Failed to call a MCP tool: Error executing tool wrong_tool`, which the regex does not match.

Nothing in this crate is on that path. The failure is between dspy's test and the SDK version, and
dspy's own CI does not meet it: `third_party/dspy/uv.lock` pins `mcp 1.29.0`, where the server
forwards the cause, and the test is marked `extra`, which dspy's conftest deselects by default. This
harness installs `mcp` 2.x deliberately, because `test_convert_mcp_tool_with_v2_client` skips itself
below SDK 2 and the newer client path would otherwise go untested.

So this script runs upstream's own test against unmodified dspy and prints the verdict. Run it
whenever the entry in the conftest's `FAILS_ON_UPSTREAM_TOO` is in doubt: while it prints FAILED,
the entry is a statement about the installed SDK; if it ever prints PASSED, either the SDK forwards
the cause again or the pinned dspy updated the test, and the entry is stale.
"""

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DSPY = ROOT / "third_party" / "dspy"
TEST = "tests/utils/test_mcp.py::test_convert_mcp_tool"


def main() -> int:
    if not (DSPY / "tests").is_dir():
        print(f"the dspy submodule is not checked out at {DSPY}", file=sys.stderr)
        return 2
    finished = subprocess.run(
        [sys.executable, "-m", "pytest", TEST, "-q", "--extra", "-p", "no:cacheprovider"],
        cwd=DSPY,
        capture_output=True,
        text=True,
    )
    tail = finished.stdout.strip().splitlines()[-1:] or ["(no output)"]
    verdict = "PASSED" if finished.returncode == 0 else "FAILED"
    print(f"unmodified dspy, {TEST}: {verdict}")
    print(f"  {tail[0]}")
    if verdict == "PASSED":
        print("\nThe SDK now sends the tool's own message. Drop the FAILS_ON_UPSTREAM_TOO entry.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
