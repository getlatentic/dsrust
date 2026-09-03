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
# **A floor is only a floor over the code it was measured against.** This is not run by the gates,
# so a module added after a measurement is covered by nothing until someone re-runs it — which is
# how `pcg64.rs` sat outside pyrng's floor of 7 from the day it landed. Adding a file to a scoped
# crate means re-measuring that crate, and a floor that has not moved while the crate grew is a
# number about the past.
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
#   dsrust-tpe 11 — one of them the old floor: `n < 25` against `n <= 25` in `parzen`'s
#                  `default_weights`, where at n=25 the ramp is empty either way and both arms
#                  return twenty-five ones. The other ten arrived with `argsort` and `truncnorm`,
#                  and each was measured rather than assumed. Seven are in `argsort`: six on the
#                  choice of which partition to push and which to walk — the halves are disjoint and
#                  both get sorted, so the order changes only the stack depth (flipping it leaves
#                  all twenty tests green) — and one on the insertion sort's `pl + 1`, where
#                  starting a place earlier re-inserts an element into a prefix of one. Three are in
#                  `truncnorm`: `ndtr_single`'s *second* boundary, where the two arms agree to the
#                  last bit on this libm; `log_sum`'s big/small pick, which equal arguments have
#                  already returned past; and Newton's relative tolerance at exact equality. Every
#                  one carries its note in the source.
#                  Measured down from 29 missed + 2 hanging at first run: the hangs were manual
#                  cursors a mutant could stall (bounded searches now), and the survivors were a
#                  wrong `default_weights` — a second, incorrect copy of a function the crate
#                  already had right — plus untested boundaries in rounding, weights, the
#                  log-sum-exp shift, and a range whose low is not zero.
#
#                  The last hang was worth the three passes it took, because the first two treated
#                  it as a loop that needed a bound and it was not one. `argsort`'s partition walks
#                  ended in `unwrap_or(pr)` / `unwrap_or(pl)`, and an exhausted walk clamped *onto*
#                  the end of its range is a fixed point: on an inverted range — which the width
#                  test's mutant reaches and no real input does — both walks came up empty every
#                  pass, neither index moved, and `pi >= pj` never came true. Landing one past the
#                  end instead, where numpy's own pointer sits, makes the pair cross and the loop
#                  stop. Wrapping the whole thing in `for _ in 0..num` had hidden none of that and
#                  cost three survivors of its own.
#
#                  Both replacements are counted rather than clamped — `pi += 1 + take_while(..)
#                  .count()` — so the fallback is the arithmetic's own answer and not a constant no
#                  test can reach. That is the general shape: a defensive branch nothing exercises
#                  is a survivor by construction, and the two `unwrap_or` constants plus a
#                  `leading_zeros` depth bound were exactly that, five untestable mutants added
#                  while fixing a hang. What catches the width mutant now is one assertion at the
#                  push, `stack.len() < num`, true because each entry goes on at half the range
#                  below it, and written as the length rather than the log of it so the bound holds
#                  no arithmetic of its own.
#   pyrng      29 — **a floor is only a floor over the code it was measured against.** This said 7
#                  until 2026-08-28, and 7 was right when it was written: `pcg64.rs` did not exist
#                  yet. It arrived with SIMBA, nobody re-ran this, and a ratchet that is never
#                  re-measured cannot notice a module. It went 7 -> 37 on re-measurement, then 37 ->
#                  29 once the 37 were read rather than counted.
#
#                  Five *hangs* became zero on the way: `generate_state`'s shift-until-empty and
#                  both poisson loops were cursors a mutant could stall, and are bounded walks now —
#                  which is why the missed count rose rather than fell. A mutant that used to burn
#                  the timeout now runs and is counted, and the timeouts had been masking a whole
#                  cluster in `poisson_ptrs` and `loggam`.
#
#                  Those turned out to be two findings, not one. **Seven were reachable and nobody
#                  had looked**: three deep streams in the golden kill them, chosen as the shortest
#                  witnesses over 24 seeds — `k` landing exactly on zero, the `v > us` squeeze
#                  deciding, and one draw in 350,000 where `loggam`'s second coefficient flips an
#                  acceptance — and a second oracle kills three more, CPython's `lgamma` against the
#                  series over 138 points, which reaches coefficients poisson would need some 10^8
#                  draws to touch. Both are in the source with their numbers.
#
#                  **Twelve are equivalent for one reason**, written on `poisson_ptrs`: two of its
#                  three arms are squeezes, regions that sit strictly inside what the density test
#                  would decide anyway, and every surviving mutant on them makes a squeeze *smaller*
#                  — which only sends work to the density test and cannot move a draw. Verified, not
#                  argued: 360,000 draws with no acceptance outside the density's own region. The
#                  rest are below arithmetic's reach — `A[9]` changes the returned `f64` by exactly
#                  zero, and `n` is a free shift in a recurrence that is exact for any `n`.
#
#                  The corpus was widened first, and that is what made the split visible: 24 draws
#                  per stream to 400, lambdas out to 1000, then three streams of 1200 to 2400. It
#                  moved the count by two on its own. Widening blindly stops there; the remaining
#                  seven needed a witness search, and that is the difference between a bigger corpus
#                  and a better one.
#
#                  The other seventeen are outside poisson and are boundary cases of the same kind:
#                  `choices`' `len - 1` needs `random_double() * total` to reach `total`, which a
#                  draw strictly below one does not; `twist`'s `|` and `^` act on disjoint masks;
#                  `getrandbits` at exactly 32 shifts by zero either way; both `<=` spellings in the
#                  two searches need a draw landing exactly on a cumulative boundary; and
#                  `seed_values` and `generate_state` mix words with `|` where the operands cannot
#                  overlap.
#
#   Both re-measured 2026-08-07 under the fixed methodology and both reproduced exactly (151 and
#   203 mutants). They had stood since before that fix, and "probably unchanged" is not a
#   measurement — the campaign had just twice found a level number hiding a changed set.
#   dsrust-gepa 13 — every one a reasoned equivalent with its note in the source, worked down from 46
#                  across three passes. Four in the fence walks (`find` and `rfind` answer `None`
#                  together, so neither `-1` sentinel is reachable as a difference; and skipping the
#                  newline is invisible through the caller's `trim`). Two on the engine's anti-stall
#                  guard, which cannot fire because `batch.rs` refuses an empty trainset and every
#                  other path spends. Two in `merged_candidate` and the pair swap, both unreachable
#                  given the arm immediately above them — the `&&`/`||` confirmed by enumerating the
#                  whole triple space rather than argued. Two on `select_eval_subsample`'s `> 0`
#                  guards, where `sample` at `k = 0` neither draws nor appends, so the branch they
#                  skip is already a no-op. Three in `pyset`, the probe bound and the resize `+ 1`.
#
#                  The three passes each graded the one before it: pass two found that pass one's
#                  merge-cap case reproduced the uncapped run byte for byte, and pass three found
#                  that a `-` against a `+` in the subsample's take is invisible in the returned ids
#                  — truncation and `sample`'s stable prefix absorb it — and shows up only in where
#                  the generator was left. Both fixes came from sweeping the parameter against gepa
#                  for a configuration that separates the spellings, not from reasoning about which
#                  case would.
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
  "dsrust-tpe:11:0"
  "pyrng:29:0"
  "dsrust-gepa:13:0"
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
#   signature 0:0 — 2026-08-06, from 18 missed + 3 hanging of 250. Two of the eighteen were
#       branches that could not be told from their neighbours — `union`'s `Option<T>` case wrote
#       byte-identical JSON to its general arm, and `object`'s is_object guard could not change
#       what `node` already answers — so both were deleted rather than tested. The mutant count
#       fell 250 -> 222 with them and the prefix rewrites: code that cannot be wrong has mutation
#       sites, and removing it removes them.
#   optimize 0:0 — 2026-08-07, from 69 missed + 4 hanging of 572, in three passes, and the richest
#       scope by far. Two clusters were reachable only through a model call, and "needs an LM" had
#       been standing in for "cannot be tested" — `DummyLM` is a public export and drives the whole
#       batching walk. The second pass then graded the first and the third graded the second: a
#       three-row corpus cannot reach a bound that bites at batch ten, tests that never read the
#       config leave a temperature deletable, and `asked` — the function deciding who answers an
#       ensemble — had never run while `len` and `is_empty` had. Every residue was a limit of the
#       previous pass's tests rather than noise.
SLICES=(
  "adapter:0:0"
  "anthropic:0:0"
  "ollama:0:0"
  "openai:0:0"
  "signature:0:0"
  "optimize:0:0"
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
    signature) printf '%s\n' \
      'crates/dsrust/src/signature/**/*.rs' \
      crates/dsrust/src/signature.rs ;;
    optimize) printf '%s\n' \
      'crates/dsrust/src/optimize/**/*.rs' \
      crates/dsrust/src/optimize.rs ;;
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

# The half that covers a *hand*-run lives in `.cargo/mutants.toml`, which cargo-mutants reads on
# every invocation in this repo — the export above only protects runs that come through here, and
# the marker escaped the first time precisely because someone ran the tool directly. Refuse rather
# than run without it: a poisoned shared cache reads as a corrupted toolchain, `git status` shows
# nothing, and clearing it costs two `cargo clean`s.
if ! grep -q 'build.build-dir=\\"mutants-build\\"' "$ROOT/.cargo/mutants.toml" 2>/dev/null; then
  echo "refusing to run: .cargo/mutants.toml no longer pins a relative build-dir." >&2
  echo "Without it a hand-run of cargo mutants writes mutated rlibs into the shared" >&2
  echo "build directory every other project on this machine links from." >&2
  exit 2
fi

# How many mutants run at once, and how many rustc processes each may spawn.
#
# The product is what lands on the machine: `-j 3` with cargo's default inner parallelism is three
# concurrent builds each taking every core it can find. This box is shared with whoever is using it,
# so both ends are capped and the whole thing is niced.
# Three jobs by default. It was one for a long time, after a run at two sat at 200% CPU for half
# an hour and had to be killed by hand — a mutant that *hangs* holds its core for the whole
# timeout instead of finishing early. That reason expired twice over and the default was never
# revisited: the relative `build-dir` above restores the per-job isolation `-j` needs (validated
# by running one file serially and at `-j`, which agree), and the hang classes that caused the
# original pile-up are closed in every slice measured since.
#
# The cost of not revisiting it was real: the 2026-08-06 campaign ran some twelve hours serially
# on a ten-core machine, most of it avoidable. 3 x 2 leaves four cores for the editor and other
# worktree sessions, which is what the caution was actually protecting.
JOBS=${DSRS_MUTANT_JOBS:-3}
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
