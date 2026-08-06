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
#     ./scripts/run_mutants.sh adapter    # one dsrust slice (see SLICES; ~40-80 minutes each)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# package:missed:hangs. **Two floors, not one**, because they answer to different work and a corpus
# that improves can move a mutant from the first into the second.
#
# Measured: three comment cases took `lookahead.rs` from 55 silent survivors to 53, and its hangs
# from 6 to 10 — two of them the very mutants that stopped surviving, now driven into a loop that
# never advances by an input that finally reaches them. A single total called that a regression from
# 61 to 63 while the thing it is supposed to track had improved. A hang is still not a passing test,
# so both are floors; collapsing them is what hid the direction.
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
#   dsrust-json-repair 156 unpinned, 1 hanging — of 1932 viable in 56 minutes. The hang class is
#                    closed: the read counter lives in `Source::at` where no direct reader can
#                    drift out of its reach, the comment-strip and whitespace skips are
#                    iterator-shaped and have no cursor for a mutant to stall, and the counter's
#                    floor dropped to 2^16 after three escape mutants showed a slow-enough loop
#                    losing the race to the timeout. The one hang left is `char_here` replaced
#                    wholesale, which removes the counted call along with the function — the
#                    watcher-replaced class, which no guard survives by construction.
#                    value.rs holds exactly its two reasoned equivalents: `>` to `>=` in
#                    `Object::remove`'s index shift, where no position can equal the removed one
#                    because it just left the map; and `&&` to `||` in `Object::insert`'s
#                    threshold, which only rebuilds an index that already exists. Everything else
#                    the white-box tests see from inside, since the hybrid is invisible from
#                    outside on purpose.
#                    The clusters left are the lookahead cache (11 of lookahead.rs's 19, output-
#                    blind by construction), `empty.rs` at 16, and the long tail — worked down
#                    from 296 and 79 at the first honest measurement, across two optimisation
#                    arcs that each re-pinned what they touched.
#   dsrust    — not run whole: 3619 mutants at roughly half a minute each is some five hours. Run it
#               scoped by file. The byte-critical adapter slice (chat, prompt, exchange, demos,
#               history, parse) measured 43 of 143 viable on 2026-08-01, 35 of them in `parse.rs`,
#               because nineteen fixtures pin the prompt this crate *sends* and none pin what it
#               *reads*. Filed as `parse-side-goldens`; no ratchet entry until that lands, since a
#               five-hour floor nobody runs is not a gate.
BASELINES=(
  "dsrust-tpe:1:0"
  "pyrng:7:0"
  "dsrust-gepa:24:0"
  "dsrust-json-repair:156:1"
)

# The `dsrust` slices with floors. Scoped because the whole crate is a five-hour run, and a gate
# nobody runs is not a gate. Each entry is name:missed:hangs, with its files in SLICE_FILES below.
#
#   adapter 0:0 — as of 2026-08-06. The count held at 1 across the json-repair merge *by
#       coincidence*: the old survivor (`parse_json`'s `start < end`, an equivalent) moved into
#       dsrust-json-repair with the merge, and a new one appeared at the same count — the
#       `!text.is_empty()` guard on `section_value`'s literal fallback, now pinned by three
#       `note: Any` cases in the parse golden with dspy as the oracle. A floor that stays level is
#       not evidence the survivors are the same survivors.
#   anthropic 0:0 — 2026-08-06, from 20 of 80 at first measurement: thirteen usage mutants nothing
#       read, and one genuine fix — `block()`'s text rebuild stripped `cache_control`, so the
#       mutant deleting the arm was the better program and became the code.
#   ollama 0:0 — 2026-08-06, from 25+4 of 121 at first measurement, in three passes: the timeouts
#       were a stub that hung when never called (bounded now) and cursor arithmetic a mutant could
#       spin (flatten is a bounded `for` now — a stalled round runs out of rounds and the assert
#       names it). The second pass graded the first: a block-list test on a *user* turn covered
#       content_and_images while looking like it covered content_str, and the re-measurement was
#       the only thing that could tell. The predicted `<`/`<=` equivalent vanished with the
#       `while` it lived on.
#   openai 0:0 — 2026-08-06, from 27 of 173 at first measurement. The distribution was the
#       finding: the byte-verified chat wire contributed one survivor (`from_env`, which no
#       captured byte can see); eleven sat in the Responses API, the newest wire with the least
#       golden history. Byte-verification holds exactly where it exists.
SLICES=(
  "adapter:0:0"
  "anthropic:0:0"
  "ollama:0:0"
  "openai:0:0"
)
slice_files() {
  case "$1" in
    adapter) printf '%s\n' \
      crates/dsrust/src/adapter/parse.rs \
      crates/dsrust/src/adapter/chat.rs \
      crates/dsrust/src/adapter/prompt.rs \
      crates/dsrust/src/adapter/exchange.rs \
      crates/dsrust/src/adapter/demos.rs \
      crates/dsrust/src/adapter/types/history.rs ;;
    anthropic) printf '%s\n' 'crates/dsrust/src/lm/anthropic/**/*.rs' ;;
    ollama) printf '%s\n' 'crates/dsrust/src/lm/ollama/**/*.rs' ;;
    openai) printf '%s\n' 'crates/dsrust/src/lm/openai/**/*.rs' ;;
  esac
}

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
# One job by default rather than two: a run at two sat at 200% CPU for half an hour on this
# machine and had to be killed by hand, because a mutant that *hangs* holds its core for the
# whole timeout instead of finishing early. The hangs are gone, and the default stays at one —
# the box is shared with an editor and other worktree sessions, and a measurement is not worth
# making it unusable. Raise it deliberately when nothing else needs the cores.
JOBS=${DSRS_MUTANT_JOBS:-1}
export CARGO_BUILD_JOBS=${DSRS_MUTANT_BUILD_JOBS:-2}

trap 'rm -rf "$ROOT/mutants.out" "$ROOT/mutants.out.old"' EXIT

if ! cargo mutants --version > /dev/null 2>&1; then
  echo "cargo-mutants is not installed: cargo install cargo-mutants --locked" >&2
  exit 127
fi

# One place that decides pass or fail, so the package runs and the file-scoped slice cannot drift
# apart. Two floors, never one: MISSED is a line no assertion pins and wants a test; TIMEOUT is a
# mutant that spins instead of failing and wants an exit condition. Collapsing them hid a real
# improvement once (see the header), and it was also how this function went stale the first time —
# the two-floor rework was written inline in the loop, this single-total version was left defined
# and uncalled, and the adapter slice that only this could check silently stopped running.
check_floors() {
  local label="$1" allowed_missed="$2" allowed_hangs="$3" log="$4" status=0
  local missed hangs
  missed=$(grep -cE "^MISSED" "$log" || true)
  hangs=$(grep -cE "^TIMEOUT" "$log" || true)
  tail -1 "$log"
  local kind found floor word pattern
  for kind in missed hangs; do
    case "$kind" in
      missed) found=$missed; floor=$allowed_missed; word="unpinned"; pattern="^MISSED" ;;
      hangs)  found=$hangs;  floor=$allowed_hangs;  word="hanging";  pattern="^TIMEOUT" ;;
    esac
    if [ "$found" -gt "$floor" ]; then
      echo "  RATCHET BROKEN ($label): $found $word, floor $floor" >&2
      grep -E "$pattern" "$log" | sed 's/ in [0-9].*//' >&2
      status=1
    elif [ "$found" -lt "$floor" ]; then
      echo "  $found $word, below the floor of $floor — lower it in this script"
    else
      echo "  $found $word, at the floor"
    fi
  done
  return "$status"
}

status=0
matched=0

# The dsrust slices, each run when no scope was named or its name was.
for entry in "${SLICES[@]}"; do
  slice="${entry%%:*}"
  rest="${entry#*:}"
  slice_missed="${rest%%:*}"
  slice_hangs="${rest##*:}"
  if [ "$#" -gt 0 ] && [ "${1:-}" != "$slice" ]; then
    continue
  fi
  matched=1
  echo "==> cargo mutants -p dsrust, $slice slice (floors: $slice_missed missed, $slice_hangs hanging; -j $JOBS x $CARGO_BUILD_JOBS)"
  log="target/mutants-$slice.log"
  files=()
  while IFS= read -r file; do files+=(--file "$file"); done < <(slice_files "$slice")
  set +e
  nice -n 10 cargo mutants -p dsrust --timeout 120 -j "$JOBS" "${files[@]}" > "$log" 2>&1
  set -e
  check_floors "$slice" "$slice_missed" "$slice_hangs" "$log" || status=1
done

for entry in "${BASELINES[@]}"; do
  package="${entry%%:*}"
  rest="${entry#*:}"
  allowed_missed="${rest%%:*}"
  allowed_hangs="${rest##*:}"
  if [ "$#" -gt 0 ] && [ "$1" != "$package" ]; then
    continue
  fi
  matched=1
  echo "==> cargo mutants -p $package (floors: $allowed_missed missed, $allowed_hangs hanging; -j $JOBS x $CARGO_BUILD_JOBS)"
  log="target/mutants-$package.log"
  set +e
  nice -n 10 cargo mutants -p "$package" --timeout 120 -j "$JOBS" > "$log" 2>&1
  set -e
  check_floors "$package" "$allowed_missed" "$allowed_hangs" "$log" || status=1
done

# A named package with no entry ran nothing and exited 0, which reads as a clean run. A crate has to
# be measurable before it has a floor, so say what to do rather than succeed silently.
if [ "$#" -gt 0 ] && [ "$matched" -eq 0 ]; then
  echo "no baseline entry for '$1'. To measure a crate for the first time:" >&2
  echo "  CARGO_BUILD_BUILD_DIR=mutants-build CARGO_BUILD_JOBS=1 \\" >&2
  echo "    nice -n 15 cargo mutants -p $1 --timeout 120 -j 2" >&2
  echo "then add \"$1:<missed+timeout>\" to BASELINES with the survivors accounted for." >&2
  exit 2
fi
exit "$status"
