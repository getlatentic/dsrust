#!/usr/bin/env bash
# Run Python DSPy's own adapter tests with this crate's renderer underneath.
#
# Nothing here rewrites a test: upstream's file is downloaded at the pinned tag and executed
# as-is, with conftest.py swapping dspy.ChatAdapter for the Rust-backed subclass.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="$(cat "$ROOT/scripts/DSPY_VERSION")"
WORK="$ROOT/target/upstream-tests"
VENV="$ROOT/.dspy-venv"

[ -x "$VENV/bin/python" ] || { echo "run: uv venv .dspy-venv --python 3.12 && uv pip install --python $VENV/bin/python dspy==$VERSION pytest pytest-asyncio maturin" >&2; exit 1; }

echo "==> Building and installing the Rust bridge"
# maturin knows how to link a CPython extension module on each platform; hand-rolled linker
# flags produced a dylib that dlopen rejected.
( cd "$ROOT/bridge" && VIRTUAL_ENV="$VENV" "$VENV/bin/maturin" develop --release 2>&1 | tail -1 )

mkdir -p "$WORK"
cp "$ROOT/bridge/python/rust_adapter.py" "$ROOT/bridge/python/conftest.py" "$WORK/"

echo "==> Fetching upstream tests at dspy $VERSION (unmodified)"
for file in tests/adapters/test_chat_adapter.py tests/adapters/conftest.py; do
  out="$WORK/upstream_$(basename "$file")"
  curl -sSf --max-time 30 \
    "https://raw.githubusercontent.com/stanfordnlp/dspy/$VERSION/$file" -o "$out" || true
done
# Upstream's own conftest must not shadow ours; ours imports what it needs.
rm -f "$WORK/upstream_conftest.py"

echo "==> Running upstream's suite against Rust"
cd "$WORK"
PYTHONPATH="$WORK" "$VENV/bin/python" -m pytest upstream_test_chat_adapter.py \
  "$@"
