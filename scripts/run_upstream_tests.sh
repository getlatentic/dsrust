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
# maturin supplies the platform's extension-module link arguments; build.rs keeps them scoped
# to this crate. Install the wheel rather than develop-mode, so a stale .so cannot linger.
( cd "$ROOT/bridge" && PYO3_PYTHON="$VENV/bin/python" "$VENV/bin/maturin" build --release 2>&1 | tail -1 )
WHEEL=$(ls -t "$ROOT"/target/wheels/dsrs_bridge-*.whl | head -1)
"$VENV/bin/python" -m pip install --force-reinstall --quiet --no-deps "$WHEEL" 2>/dev/null \
  || uv pip install --python "$VENV/bin/python" --force-reinstall -q --no-deps "$WHEEL"

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
