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
#   dsrust-json-repair — no entry yet, deliberately. A run was started and abandoned: the crate was
#                        edited four times while it was in flight, so its copies held 217 schema
#                        cases against the tree's 220 and no `fuzz.rs` guard, and its log carried 80
#                        MISSED and 30 TIMEOUT against a tree that no longer existed. A floor has to
#                        come from a still tree. Freeze the crate, run it whole, then write the
#                        number the run produced.
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
# **It also means the runs below must stay serial.** One fixed build dir is exactly what cargo-mutants
# gives each parallel job its own of, and overriding it takes that away: every job writes the same
# `libjson_repair-<hash>.rlib`, because the metadata hash is the crate, not the mutation. Adding
# `-j` therefore has test binaries linking *someone else's* mutant, and the run reports survivors
# that are nothing of the sort — measured at 24 against a serial 0 for the same crate, all six
# escape arms of a `json.dumps` reimplementation among them, each of which fails on its own in
# under a second. A wrong number here is worse than a slow one: it is a ratchet nobody can trust.
#
# **And the path is per-run, not fixed.** This repo is worked in many worktrees over one machine, so
# a constant `dsrs-mutants-build` is the same collision one directory further out: two sessions
# mutating two crates wrote the same shared dir, each linking the other's mutants, and whichever
# finished first deleted it out from under the other — the trap below removes the directory it
# names. Both runs' numbers were rubbish and neither said so. `mktemp -d` keeps the isolation the
# override is for and takes the collision away; the cost is one crate's object code per concurrent
# run, which is what isolation costs.
#
# **A relative path restores `-j`.** Everything above is true of an *absolute* override, and that is
# the only kind this had. Cargo resolves a relative `build-dir` against the workspace root it is
# building — measured, not assumed — and cargo-mutants gives every parallel job its own copy of the
# tree. So a relative name lands inside each job's own copy, which is exactly the per-job isolation
# the absolute path took away, while still keeping the run out of `~/.cargo/build-dir` where an
# ordinary `cargo test` would link a mutant. It is also per-worktree for free, since two sessions
# have two trees. Validated by running one file serially and at `-j`, which must agree.
CARGO_BUILD_BUILD_DIR="mutants-build"
export CARGO_BUILD_BUILD_DIR

# How many mutants run at once, and how many rustc processes each may spawn.
#
# The product is what lands on the machine: `-j 3` with cargo's default inner parallelism is three
# concurrent builds each taking every core it can find. This box is shared with whoever is using it,
# so both ends are capped and the whole thing is niced.
JOBS=${DSRS_MUTANT_JOBS:-2}
export CARGO_BUILD_JOBS=${DSRS_MUTANT_BUILD_JOBS:-2}

trap 'rm -rf "$ROOT/mutants.out" "$ROOT/mutants.out.old"' EXIT

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
  echo "==> cargo mutants -p $package (baseline $allowed, -j $JOBS x $CARGO_BUILD_JOBS)"
  log="target/mutants-$package.log"
  set +e
  nice -n 10 cargo mutants -p "$package" --timeout 120 -j "$JOBS" > "$log" 2>&1
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
