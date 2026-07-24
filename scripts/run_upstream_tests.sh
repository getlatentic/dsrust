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

[ -x "$VENV/bin/python" ] || { echo "run: uv venv .dspy-venv --python 3.12 && uv pip install --python $VENV/bin/python dspy==$VERSION pytest pytest-asyncio maturin pillow" >&2; exit 1; }

echo "==> Building and installing the Rust bridge"
# maturin supplies the platform's extension-module link arguments; build.rs keeps them scoped
# to this crate. Install the wheel rather than develop-mode, so a stale .so cannot linger.
( cd "$ROOT/bridge" && PYO3_PYTHON="$VENV/bin/python" "$VENV/bin/maturin" build --release 2>&1 | tail -1 )
WHEEL=$(ls -t "$ROOT"/target/wheels/dsrs_bridge-*.whl | head -1)
"$VENV/bin/python" -m pip install --force-reinstall --quiet --no-deps "$WHEEL" 2>/dev/null \
  || uv pip install --python "$VENV/bin/python" --force-reinstall -q --no-deps "$WHEEL"

mkdir -p "$WORK"
cp "$ROOT"/bridge/python/{rust_adapter,rust_signature,crossings,reflect,conftest}.py "$WORK/"

# The upstream files this crate is held to. Adding one here is how coverage grows: it will
# arrive with failures, and each becomes a named entry in conftest.py's to-do list or a fix.
SUITES=(
  adapters/test_chat_adapter.py adapters/test_json_adapter.py
  adapters/test_adapter_utils.py adapters/test_base_type.py adapters/test_code.py adapters/test_citation.py
  adapters/test_document.py adapters/test_audio.py adapters/test_reasoning.py adapters/test_tool.py
  adapters/test_xml_adapter.py adapters/test_baml_adapter.py adapters/test_two_step_adapter.py
  predict/test_predict.py predict/test_chain_of_thought.py predict/test_react.py
  teleprompt/test_bootstrap.py
  signatures/test_signature.py signatures/test_custom_types.py
  signatures/test_adapter_file.py signatures/test_adapter_image.py
  primitives/test_example.py
  predict/test_multi_chain_comparison.py predict/test_aggregation.py
)

# SUITES is an allowlist, so a green run only speaks for the files in it. Reporting that against
# everything upstream ships stops green from reading as done, and catches a name that no longer
# exists after a version bump.
MANIFEST="$ROOT/scripts/upstream_tests.txt"
echo "==> Upstream coverage"
# The backlog says which suites a sprint shipped; this array says which ones actually run. A
# claim without evidence is the failure mode a plan has, so the two are checked against each
# other before anything else.
python3 "$ROOT/scripts/check_plan.py"
TOTAL=$(grep -vc '^#' "$MANIFEST")
echo "  ${#SUITES[@]} of $TOTAL upstream test files ($(( ${#SUITES[@]} * 100 / TOTAL ))%)"
grep -v '^#' "$MANIFEST" | sed 's|tests/||;s|/.*||' | sort -u | while read -r area; do
  have=$(printf '%s\n' "${SUITES[@]}" | grep -c "^$area/" || true)
  want=$(grep -c "^tests/$area/" "$MANIFEST" || true)
  [ "$have" -lt "$want" ] && printf '    %-14s %s of %s\n' "$area" "$have" "$want"
done || true

# The whole tests/ tree is needed, not just the run files: every shared helper a suite imports
# (tests.adapters.conftest's format_messages_and_lm_kwargs, tests.test_utils, …) has to be an
# importable package, which is how dspy 3.3's new shared conftest first surfaced.
#
# The pinned tree lives in the `third_party/dspy` submodule (checked out at tag $VERSION), so the
# exact source the crate is held to is captured in this repo and needs no network. The tarball
# download is the fallback for a checkout where the submodule was not initialised.
SUBMODULE="$ROOT/third_party/dspy/tests"
SRC="$WORK/upstream_src"
rm -rf "$SRC"; mkdir -p "$SRC/tests"
if [ -d "$SUBMODULE" ]; then
  echo "==> Upstream tests from the dspy submodule (pinned at $VERSION)"
  PINNED_AT="$(git -C "$ROOT/third_party/dspy" describe --tags 2>/dev/null || echo unknown)"
  [ "$PINNED_AT" = "$VERSION" ] || echo "  warning: submodule is at $PINNED_AT, not $VERSION"
  cp -R "$SUBMODULE/." "$SRC/tests/"
else
  echo "==> Fetching upstream tests at dspy $VERSION (submodule not initialised; downloading)"
  echo "     run \`git submodule update --init third_party/dspy\` to use the pinned tree instead"
  TARBALL="$WORK/dspy-$VERSION.tar.gz"
  [ -f "$TARBALL" ] || curl -sSfL --max-time 120 \
    "https://github.com/stanfordnlp/dspy/archive/refs/tags/$VERSION.tar.gz" -o "$TARBALL"
  tar xzf "$TARBALL" -C "$SRC" --strip-components=1 "dspy-$VERSION/tests"
fi

# Run flattened, prefixed copies so pytest loads OUR conftest (not upstream's tree conftests), while
# `PYTHONPATH` still resolves `tests.*` helper imports against the full tree below.
for file in "${SUITES[@]}"; do
  cp "$SRC/tests/$file" "$WORK/upstream_$(basename "$file")"
done
# Upstream's top-level conftest pulls in a litellm test server this harness does not run; neutralise
# it so importing anything under tests/ can never drag the server in. Sub-package helpers stay intact.
: > "$SRC/tests/conftest.py"

echo "==> Running upstream's suite against Rust"
cd "$WORK"
PYTHONPATH="$WORK:$SRC" "$VENV/bin/python" -m pytest $(for f in "${SUITES[@]}"; do echo "upstream_$(basename "$f")"; done) \
  "$@"
