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
# `-D warnings` because a broken intra-doc link is a warning, and a gate that prints one and passes
# is not enforcing it. Three appeared in one session — a link left pointing at a renamed method, and
# two names that became ambiguous once a macro joined a function of the same name — each noticed only
# by grepping output nobody was required to read.
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps

echo "==> file sizes"
./scripts/file_sizes.py

# Last, because both build against the crate the gates above just proved, and both build from
# *outside* the workspace — the only place a leaked dependency is visible.
./scripts/check_external_consumer.sh
./scripts/check_docs.py

# The one ignored test the gate can actually satisfy: dspy's own reader, on a file this crate
# wrote. It is `#[ignore]` because it needs `.dspy-venv`, which not every checkout has — but this
# script needs that venv anyway for the suite below, so here the reason to skip does not apply.
#
# It had never run. The paths inside it were written as though `CARGO_MANIFEST_DIR` were the
# workspace root, so it could only ever have failed, and the interop claim on the README — that
# `dspy.load` opens what this saves — rested on a test nothing executed.
if [ "${DSRS_SKIP_UPSTREAM:-0}" = "1" ]; then
  echo "==> dspy reads a saved program (SKIPPED: DSRS_SKIP_UPSTREAM=1)"
else
  echo "==> dspy reads a saved program"
  cargo test -p dsrust --test saved_program -- --ignored
fi

# The upstream pytest suite, through the bridge. Last because it is the slowest — about four
# minutes — and because everything above must hold before Python's opinion is worth having.
#
# Not in this script until `dspy.Code` was found red here: two adapter tests had been failing at
# HEAD for as long as nobody remembered to run the suite by hand, and every gate anyone did run
# passed the whole time. A gate is what a script runs. Set DSRS_SKIP_UPSTREAM=1 for a fast local
# loop; CI must not.
if [ "${DSRS_SKIP_UPSTREAM:-0}" = "1" ]; then
  echo "==> upstream pytest suite (SKIPPED: DSRS_SKIP_UPSTREAM=1)"
else
  echo "==> upstream pytest suite"
  ./scripts/run_upstream_tests.sh
fi
