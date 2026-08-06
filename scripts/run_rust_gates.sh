#!/usr/bin/env bash
# The gates that run in a worktree: cargo's four, the file-size rule, an outside caller, the guides,
# and the ports held to the libraries they reproduce.
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

# A build directory belonging to *this* checkout.
#
# `~/.cargo/config.toml` sets one `build-dir` for the machine, and cargo's metadata hash is over the
# package name, version and features — not the path. Two checkouts of this repo therefore write the
# same `deps/gepa-434c7ff1220d564a`, and the second run to build clobbers the first's binary while
# `cargo test` happily runs whatever is sitting there.
#
# That is not hypothetical: this gate reported 1153 tests from a worktree whose source has 1145,
# because a sibling checkout's uncommitted `dsrust-gepa` had eight more and had built last. A test
# count is one of the few numbers this repo publishes about itself, and it was reading another
# working tree's.
#
# Per-checkout rather than `mktemp -d`, which would be unique and would also discard every
# incremental artifact on each run — the same reasoning `run_mutants.sh` records, in the other
# direction: it needs isolation per *run*, this needs isolation per *checkout*.
export CARGO_BUILD_BUILD_DIR="$ROOT/target/build-dir"

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

# The optional features, which the run above does not reach. `mp3` is off by default because LAME
# is LGPL-3.0 and this crate is MIT/Apache — so its encoder and the test that decodes what the
# encoder wrote are invisible to `--workspace`, which is how a test comes to never run at all.
# Counted separately rather than added to the ratchet: the total above is the default build's.
echo "==> cargo test -p dsrust --features mp3 (optional, not in the count)"
bounded cargo test -p dsrust --features mp3 --lib adapter::types::audio

echo "==> cargo build --all-targets"
cargo build --all-targets

echo "==> cargo fmt --check"
cargo fmt --check

echo "==> cursor-loop lint (scripts/lints/)"
# A while-loop advancing by hand-maintained arithmetic is one mutated operator from a spin, and
# the shape hung the suite four times in one day before this existed. The rule file carries the
# history; a legitimate cursor loop carries an in-source suppression naming why a stall still
# terminates loudly.
if ! ast-grep --version > /dev/null 2>&1; then
  echo "ast-grep is not installed: brew install ast-grep" >&2
  exit 127
fi
bounded ast-grep scan --rule scripts/lints/cursor_arithmetic_loop.yml crates/

echo "==> cargo doc --no-deps"
# Fresh, because cargo caches rendered docs: seven unresolved intra-doc links sat at HEAD for as
# long as nobody touched the files holding them, and this gate reported clean the whole time.
rm -rf target/doc
# `-D warnings` because a broken intra-doc link is a warning, and a gate that prints one and passes
# is not enforcing it. Three appeared in one session — a link left pointing at a renamed method, and
# two names that became ambiguous once a macro joined a function of the same name — each noticed only
# by grepping output nobody was required to read.
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps

echo "==> throughput floor (release, so the debug counters compile out)"
# The one regression the correctness suites cannot see: an optimisation produces identical output
# by construction, so a dead fast path fails nothing above — measured by turning the lookahead
# cache off, which passes all 57 crate tests and multiplies parse time by nine. The floor is a
# ratio against an in-process calibration loop, so a slower machine moves both sides together.
cargo test --release -p dsrust-json-repair --test throughput

echo "==> file sizes"
./scripts/file_sizes.py

# Last, because both build against the crate the gates above just proved, and both build from
# *outside* the workspace — the only place a leaked dependency is visible.
./scripts/check_external_consumer.sh
./scripts/check_docs.py

# The parity ledgers, which need the pinned Python packages and so need the environment.
#
# This script used to be "the gates that need no Python", and `check_json_repair_parity.py` was
# written, passed, and then run by nobody — a command someone had to remember, which is the shape
# `gates-were-a-checklist` is about. `run_upstream_tests.sh` was the only other home and it runs
# only in the main checkout, which is precisely where the agent who drops a function is not.
#
# So this script now wants `.venv`, the same way that one does, and says so rather than skipping.
# A skipped check is a checklist item with extra steps.
echo "==> library parity"
VENV="$ROOT/.venv"
[ -x "$VENV/bin/python" ] || {
  echo "  no .venv — run: uv sync   (a worktree can build its own, or symlink the main checkout's)" >&2
  exit 1
}
"$VENV/bin/python" "$ROOT/scripts/check_json_repair_parity.py"

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
