#!/usr/bin/env bash
# Ask the inverse of what the test gate asks: change the Rust, and does anything go red?
#
# A passing suite says nothing disagreed. It cannot tell a check that holds from a check with
# nothing behind it, and every conformance miss in this project has been the second kind — a
# comparison that filtered the field, a generator that invented it, a test that supplied the input
# it claimed to test, a fixture that stopped short of the regime that diverges. A surviving mutant
# is exactly that: a line no test constrains.
#
# Not part of `run_rust_gates.sh`. One crate takes about four minutes because every mutant rebuilds
# and reruns the suite, so this is run deliberately rather than on every change.
#
# The baselines below are a *ratchet*, not a target of zero. Some survivors are equivalent mutants —
# a change the compiler accepts and no behaviour can distinguish — and chasing those is wasted work.
# What matters is that the number never rises: a new survivor is a new line nothing checks.
#
#     ./scripts/run_mutants.sh            # the adapter slice, then every scoped crate
#     ./scripts/run_mutants.sh dsrust-tpe # one of them
#     ./scripts/run_mutants.sh adapter    # just the adapter slice (~55 minutes)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# package:survivors. Each number is a measured floor with its survivors accounted for below.
#
#   dsrust-tpe 1 — `n < 25` against `n <= 25` in `default_weights`. At n=25 the ramp is empty either
#                  way and both arms return twenty-five ones, so nothing can tell them apart.
#   pyrng      4 — one equivalent and three that hang rather than fail. `hi = len - 1` in `choices`
#                  against `hi = len`: the bound is unreachable because `random()` is strictly below
#                  one, so the target never reaches the top cumulative weight, and CPython refuses
#                  the all-zero weights that would be the only other way there. The three timeouts
#                  are `bisect_right`'s comparison and `below`'s rejection loop, where the mutant
#                  spins instead of answering — detected, but as a hang rather than a failure.
#   dsrust-gepa 28 — 25 survivors and 3 non-terminating, and still the largest gap measured. Was
#                    46, and re-measured at exactly 46 under the fixed methodology before any of it
#                    was closed — so unlike the adapter slice, this number was never an artifact.
#                    What closed the eleven: `state.rs::best_program`, whose tie clause decides
#                    which program GEPA hands back and which no case with a mean tie or a
#                    lower-mean-wider-coverage program could reach; and `candidate.rs`, where the
#                    only equality test asserted two candidates were *equal*, so `eq` could return
#                    `true` unconditionally — with equality being what makes a proposal a duplicate.
#                    What is left clusters in `engine.rs` (the optimize and propose loops),
#                    `merge.rs`, and `instruction_proposal.rs`'s fence parsing; the non-terminating
#                    ones are in `engine.rs::propose`. `pyset.rs` was ten of these and is three:
#                    both its loops terminated only on invariants they could not see — the probe
#                    on `add` having resized, the size search on the shift growing the value — so
#                    eight mutations hung the suite rather than failing it. Bounded, they fail; the
#                    run also went from 29 minutes to 14, since each hang cost two. The corpus
#                    gained a collision *near the top of the table*, which is the only way the
#                    perturb step runs at all — the nine-slot linear window absorbs every collision
#                    where `slot + 9 <= mask`, so shifting `perturb` the wrong way had changed no
#                    recorded order. The three left are equivalent and are marked as such in the
#                    source. A floor to work down, not a finished state.
#   dsrust    — not run whole: 3619 mutants at roughly half a minute each is some five hours, so it
#               is scoped by file. The byte-critical adapter slice (chat, prompt, exchange, demos,
#               history, parse) is a ratchet of its own below, at 1.
#
#               Its history is the case for doing this at all. It first measured 43 survivors of 143
#               viable, 35 in `parse.rs`, because nineteen fixtures pinned the prompt this crate
#               *sends* and none pinned what it *reads* — `parse-side-goldens`. Two later runs are
#               void and their numbers should not be quoted: one shared a build directory with
#               another session (see below), and every run before `fuzz_sweep.json` was committed
#               had `parse_fuzz` skipping, because its corpus lived in `target/` and a copied tree
#               has no `target/`. The parser's strongest oracle was absent from all of them.
#
#               Clean, on 2026-08-01: 150 mutants, 132 caught, 17 unviable, 0 timeouts, 1 missed.
BASELINES=(
  "dsrust-tpe:1"
  "pyrng:4"
  "dsrust-gepa:28"
)

# The one `dsrust` slice with a floor. Scoped by file because the whole crate is a five-hour run,
# and a gate nobody runs is not a gate.
#
#   1 — `parse_json`'s `start < end` against `start <= end`. `find('{')` and `rfind('}')` cannot
#       return the same index, because a byte is not both braces, so nothing can tell the two apart.
#       Equivalent, and recorded in the source so a later run does not offer it again as work.
ADAPTER_SLICE=(
  crates/dsrust/src/adapter/parse.rs
  crates/dsrust/src/adapter/chat.rs
  crates/dsrust/src/adapter/prompt.rs
  crates/dsrust/src/adapter/exchange.rs
  crates/dsrust/src/adapter/demos.rs
  crates/dsrust/src/adapter/types/history.rs
)
ADAPTER_BASELINE=1

# This machine shares `build.build-dir` across every project, to keep agent worktrees from each
# holding their own copy of the same dependency object code. That is the right default and it is why
# it cannot be used here: cargo-mutants isolates by copying the *source* tree, so every copy then
# compiles into the same build dir as the real one, and a mutated rlib is left where an ordinary
# `cargo test` will link it. Not hypothetical — it put `xyzzy`, the mutation marker, inside a BAML
# prompt assertion, with forty-one tests failing and `git status` clean, and it had spread past
# `dsrust` into the shared dependencies.
#
# The override is affordable because the *other* half of the setup still applies. `build-dir`
# deduplicates object files on disk; `sccache` deduplicates the compilation itself, and being
# content-addressed it cannot be poisoned — a mutated source hashes differently, so it gets its own
# entry rather than overwriting anyone's. A private build dir therefore costs disk for the length of
# the run and almost no rebuilding, and it is removed on the way out.
#
# Unique per run, not a fixed path. Several worktree sessions share this machine, and a fixed path
# put two of them in the same directory — measured, with two concrete failures rather than mere
# contention. `dsrust` depends on `dsrust-json-repair`, so one run's mutated rlib landed where the
# other's `dsrust` test binaries link from: the poisoning this override exists to prevent, one
# directory further out. And the cleanup below is `rm -rf`, so whichever run finished first deleted
# the other's build directory mid-run. Neither shows up as a failure; both show up as a number.
export CARGO_BUILD_BUILD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/dsrs-mutants-build.XXXXXX")"
trap 'rm -rf "$CARGO_BUILD_BUILD_DIR" "$ROOT/mutants.out" "$ROOT/mutants.out.old"' EXIT

if ! cargo mutants --version > /dev/null 2>&1; then
  echo "cargo-mutants is not installed: cargo install cargo-mutants --locked" >&2
  exit 127
fi

# One place that decides pass or fail, so a package run and a file-scoped run cannot drift apart.
check_ratchet() {
  local label="$1" allowed="$2" log="$3"
  # TIMEOUT counts too. A mutant that hangs was detected only in the sense that the suite never
  # finished — it is a line whose behaviour no assertion pins, same as a survivor, and letting it
  # go uncounted would let the number fall silently as tests got slower.
  local missed
  missed=$(grep -cE "^(MISSED|TIMEOUT)" "$log" || true)
  tail -1 "$log"
  if [ "$missed" -gt "$allowed" ]; then
    echo "  RATCHET BROKEN: $missed survivors, baseline $allowed" >&2
    grep -E "^(MISSED|TIMEOUT)" "$log" | sed 's/ in [0-9].*//' >&2
    return 1
  elif [ "$missed" -lt "$allowed" ]; then
    echo "  $missed survivors, below the baseline of $allowed — lower the baseline in this script"
  else
    echo "  $missed survivors, at the baseline"
  fi
  return 0
}

status=0

# The adapter slice, run whenever no package was named or `adapter` was.
if [ "$#" -eq 0 ] || [ "$1" = "adapter" ]; then
  echo "==> cargo mutants -p dsrust, adapter slice (baseline $ADAPTER_BASELINE)"
  log="target/mutants-adapter.log"
  files=()
  for file in "${ADAPTER_SLICE[@]}"; do files+=(--file "$file"); done
  set +e
  cargo mutants -p dsrust --timeout 120 "${files[@]}" > "$log" 2>&1
  set -e
  check_ratchet "adapter" "$ADAPTER_BASELINE" "$log" || status=1
fi

for entry in "${BASELINES[@]}"; do
  package="${entry%%:*}"
  allowed="${entry##*:}"
  if [ "$#" -gt 0 ] && [ "$1" != "$package" ]; then
    continue
  fi
  echo "==> cargo mutants -p $package (baseline $allowed)"
  log="target/mutants-$package.log"
  set +e
  cargo mutants -p "$package" --timeout 120 > "$log" 2>&1
  set -e
  check_ratchet "$package" "$allowed" "$log" || status=1
done
exit "$status"
