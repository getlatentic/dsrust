#!/usr/bin/env bash
# The gates that need no Python: cargo's four, plus the file-size rule.
#
# `scripts/run_upstream_tests.sh` is the fifth, and it records what it observed into
# `backlog.toml`'s `[status]`. This one exists so `rust_tests` is recorded the same way. It was the
# only number in that block no run wrote, which is why it was the only one that drifted — 48 behind
# by the time anyone counted. Proving a change is now what updates it.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p target

echo "==> cargo test --workspace"
# Teed rather than piped, so this script exits with the run's own status and not the tee's. The
# recorder reads the copy, and writes nothing when the run was not green.
set +e
cargo test --workspace 2>&1 | tee target/last-cargo-test.txt
STATUS=${PIPESTATUS[0]}
set -e
python3 scripts/record_rust_tests.py < target/last-cargo-test.txt
[ "$STATUS" -eq 0 ] || exit "$STATUS"

echo "==> cargo build --all-targets"
cargo build --all-targets

echo "==> cargo fmt --check"
cargo fmt --check

echo "==> cargo doc --no-deps"
cargo doc --no-deps

echo "==> file sizes"
./scripts/file_sizes.py

# Last, because it builds against the crate the gates above just proved. It is the only check here
# that runs from outside the workspace, which is the only place a leaked dependency is visible.
./scripts/check_external_consumer.sh
