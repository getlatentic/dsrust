//! The one failure an optimisation can produce: time.
//!
//! Mutation testing proved the correctness suites cannot see a disabled optimisation — replacing
//! both guards of the lookahead cache with `false` turns the cache off entirely and every test
//! stays green, because an optimisation produces identical output by construction. So the fast
//! paths get their own floor here, and `run_rust_gates.sh` runs this file in **release** mode,
//! where the debug read counters compile out and the timing means something.
//!
//! The floor is a *ratio*, not wall-clock: parse time over the committed sweep against a
//! calibration loop of plain arithmetic measured in the same process. A faster machine speeds both
//! sides; a dead cache slows one. Like the mutation baselines, the number is a ratchet — lowering
//! it is a decision someone writes down, not a retry.

use std::time::Instant;

/// Parse time across the corpus, in calibration units.
///
/// Measured 1.22–1.28 over three runs on the tree this was written against; killing the lookahead
/// cache measures 11.6 on the same corpus. The floor sits at four times the clean reading and
/// half the broken one, so it catches the regression and never a noisy neighbour.
const FLOOR: f64 = 5.0;

#[test]
fn the_fast_paths_are_alive() {
    if cfg!(debug_assertions) {
        // Debug carries the read counters and no optimiser; its timings say nothing about the
        // release fast paths this exists to guard.
        eprintln!("throughput floor: skipped in debug — the gate runs this with --release");
        return;
    }
    let sweep = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/conformance/json_repair_sweep.json"),
    )
    .expect("the sweep is committed");
    let corpus: serde_json::Value = serde_json::from_str(&sweep).expect("json");
    let mut inputs: Vec<String> = corpus["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .map(|case| case["input"].as_str().expect("an input").to_owned())
        .collect();
    // The sweep's inputs are short, and a cache that turns a per-comma lookahead from quadratic
    // to linear is invisible on them — measured: killing it moved the sweep ratio from 1.1 to 1.2.
    // The shape it exists for is a long unquoted value, many commas, and a quote far ahead: each
    // comma asks "where is the next delimiter", and the cache answers all of them with one scan.
    // At this size the answer is 134µs cached against 4.3ms not, which no floor can miss.
    let commas = "www, ".repeat(1500);
    inputs.push(format!("{{\"a\": {commas}\"end\"}}"));

    let calibrate = || {
        let start = Instant::now();
        let mut sink = 0u64;
        for i in 0..2_000_000u64 {
            sink = sink.wrapping_mul(31).wrapping_add(i);
        }
        std::hint::black_box(sink);
        start.elapsed()
    };
    let unit = calibrate().min(calibrate()).min(calibrate());

    let parse_all = || {
        let start = Instant::now();
        let mut bytes = 0usize;
        for input in &inputs {
            bytes += json_repair::loads(input.as_str())
                .map(|value| value.to_string().len())
                .unwrap_or(0);
        }
        std::hint::black_box(bytes);
        start.elapsed()
    };
    parse_all();
    let parse = parse_all().min(parse_all()).min(parse_all());

    let ratio = parse.as_secs_f64() / unit.as_secs_f64();
    eprintln!(
        "  sweep: {n} inputs in {parse:?}; calibration {unit:?}; ratio {ratio:.2} (floor {FLOOR})",
        n = inputs.len()
    );
    assert!(
        ratio < FLOOR,
        "parsing the sweep took {ratio:.1} calibration units against a floor of {FLOOR} — a fast \
         path has died, or the corpus grew without the floor moving"
    );
}
