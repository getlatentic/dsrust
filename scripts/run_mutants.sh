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
#     ./scripts/run_mutants.sh            # every scoped crate
#     ./scripts/run_mutants.sh dsrust-tpe # one of them
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
#   dsrust-gepa 46 — 35 survivors and 11 non-terminating, and the largest gap measured so far. The
#                    survivors cluster in `engine.rs` (the optimize and propose loops), `merge.rs`
#                    and `instruction_proposal.rs`'s fence parsing; the non-terminating ones are all
#                    `pyset.rs` arithmetic where the mutant spins instead of answering. `pyset`'s
#                    intersection tie was in that list and is not any more — see the note in
#                    `generate_pyset_fixture.py`. This number is a floor to work down, not a
#                    finished state.
#   dsrust    — not run whole: 3619 mutants at roughly half a minute each is some five hours. Run it
#               scoped by file. The byte-critical adapter slice (chat, prompt, exchange, demos,
#               history, parse) measured 43 of 143 viable on 2026-08-01, 35 of them in `parse.rs`,
#               because nineteen fixtures pin the prompt this crate *sends* and none pin what it
#               *reads*. Filed as `parse-side-goldens`; no ratchet entry until that lands, since a
#               five-hour floor nobody runs is not a gate.
BASELINES=(
  "dsrust-tpe:1"
  "pyrng:4"
  "dsrust-gepa:46"
)

# This machine points `build.build-dir` at a shared cache outside the project, which defeats
# cargo-mutants' isolation: it copies the source tree, but every copy then compiles into the *same*
# build dir as the real one, and a mutated rlib gets linked into an ordinary `cargo test` afterwards.
# That is not hypothetical — it left `xyzzy`, the mutation marker, in a BAML prompt assertion and
# forty-one tests failing with the working tree clean. A per-run build dir keeps the copies to
# themselves.
export CARGO_BUILD_BUILD_DIR="${TMPDIR:-/tmp}/dsrs-mutants-build"
mkdir -p "$CARGO_BUILD_BUILD_DIR"

if ! cargo mutants --version > /dev/null 2>&1; then
  echo "cargo-mutants is not installed: cargo install cargo-mutants --locked" >&2
  exit 127
fi

status=0
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
  # TIMEOUT counts too. A mutant that hangs was detected only in the sense that the suite never
  # finished — it is a line whose behaviour no assertion pins, same as a survivor, and letting it
  # go uncounted would let the number fall silently as tests got slower.
  missed=$(grep -cE "^(MISSED|TIMEOUT)" "$log" || true)
  tail -1 "$log"
  if [ "$missed" -gt "$allowed" ]; then
    echo "  RATCHET BROKEN: $missed survivors, baseline $allowed" >&2
    grep -E "^(MISSED|TIMEOUT)" "$log" | sed 's/ in [0-9].*//' >&2
    status=1
  elif [ "$missed" -lt "$allowed" ]; then
    echo "  $missed survivors, below the baseline of $allowed — lower the baseline in this script"
  else
    echo "  $missed survivors, at the baseline"
  fi
done
exit "$status"
