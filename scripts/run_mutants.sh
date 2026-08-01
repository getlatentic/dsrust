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
BASELINES=(
  "dsrust-tpe:1"
)

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
  missed=$(grep -c "^MISSED" "$log" || true)
  tail -1 "$log"
  if [ "$missed" -gt "$allowed" ]; then
    echo "  RATCHET BROKEN: $missed survivors, baseline $allowed" >&2
    grep "^MISSED" "$log" | sed 's/ in [0-9].*//' >&2
    status=1
  elif [ "$missed" -lt "$allowed" ]; then
    echo "  $missed survivors, below the baseline of $allowed — lower the baseline in this script"
  else
    echo "  $missed survivors, at the baseline"
  fi
done
exit "$status"
