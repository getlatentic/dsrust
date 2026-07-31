#!/usr/bin/env bash
# The gates that need no Python: cargo's four, plus the file-size rule.
#
# `scripts/run_upstream_tests.sh` is the fifth, and it records what it observed into
# `backlog.toml`'s `[status]`. This one exists so `rust_tests` is recorded the same way. It was the
# only number in that block no run wrote, which is why it was the only one that drifted — 48 behind
# by the time anyone counted. Proving a change is now what updates it.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# How long any one step may take before it is killed and the gate fails.
#
# A gate that waits forever is not a gate. A wedged `cargo test` once sat for fifteen hours on a
# test whose loopback stub never answered — it held a slot, said nothing, and the only reason
# anyone noticed was an unrelated look at the process table. A hang has to read as a failure.
CAP=${DSRS_GATE_TIMEOUT:-1800}

#: Run a command under `CAP` seconds, killing its whole process group on expiry.
#:
#: The group matters: `cargo test` spawns the test binary as a child, and TERM to cargo alone
#: leaves the binary running — which is exactly the shape the fifteen-hour process had.
bounded() {
  set -m
  "$@" &
  local job=$!
  ( sleep "$CAP"
    if kill -0 "$job" 2>/dev/null; then
      echo "  TIMED OUT after ${CAP}s — killing; raise DSRS_GATE_TIMEOUT if this is legitimate" >&2
      kill -TERM -"$job" 2>/dev/null || kill -TERM "$job" 2>/dev/null
      sleep 10
      kill -KILL -"$job" 2>/dev/null || kill -KILL "$job" 2>/dev/null
    fi ) &
  local watchdog=$!
  local status=0
  wait "$job" || status=$?
  kill "$watchdog" 2>/dev/null || true
  wait "$watchdog" 2>/dev/null || true
  set +m
  return "$status"
}
cd "$ROOT"
mkdir -p target

echo "==> cargo test --workspace"
# Teed rather than piped, so this script exits with the run's own status and not the tee's. The
# recorder reads the copy, and writes nothing when the run was not green.
set +e
# Bounded, and written to the file rather than teed: `bounded` backgrounds its command, so a
# pipeline would put the watchdog on the wrong end of it. The transcript is echoed afterwards.
bounded cargo test --workspace > target/last-cargo-test.txt 2>&1
STATUS=$?
set -e
cat target/last-cargo-test.txt
python3 scripts/record_rust_tests.py < target/last-cargo-test.txt
[ "$STATUS" -eq 0 ] || exit "$STATUS"

echo "==> cargo build --all-targets"
cargo build --all-targets

echo "==> cargo fmt --check"
cargo fmt --check

echo "==> cargo doc --no-deps"
# Fresh, because cargo caches rendered docs: seven unresolved intra-doc links sat at HEAD for as
# long as nobody touched the files holding them, and this gate reported clean the whole time.
rm -rf target/doc
cargo doc --no-deps

echo "==> file sizes"
./scripts/file_sizes.py

# Last, because both build against the crate the gates above just proved, and both build from
# *outside* the workspace — the only place a leaked dependency is visible.
./scripts/check_external_consumer.sh
./scripts/check_docs.py
