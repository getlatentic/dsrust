//! The gepa crate's engine, held to the real `gepa` package end to end.
//!
//! `tests/conformance/engine.json` is a set of full optimization runs recorded from `gepa.optimize`
//! (via `scripts/generate_gepa_engine_fixture.py`) under a scripted adapter whose scores are a pure
//! function of candidate component versions. Here the same scoring is mirrored and the runs are
//! reproduced with [`gepa::GepaEngine`]; a match pins the whole loop together — candidate selection
//! and minibatch sampling off one shared RNG in GEPA's order, the strict-improvement accept test, the
//! full-valset re-evaluation, the metric-call budget, and the state bookkeeping (parents, discovery
//! eval-counts, per-candidate mean valset score, and the best index).

use gepa::progress::Silent;
use gepa::{Candidate, EvalBatch, GepaAdapter, GepaEngine};
use gepa::{CandidateSelection, ComponentSelection};
use serde_json::Value;

/// The Rust mirror of the fixture's scripted adapter: a component text is "vN", a candidate's versions
/// are its component versions in sorted-name order (which `BTreeMap::values` yields directly), and an
/// example scores off the component it favors — monotonic with one component, a trade-off with two.
struct MirrorAdapter {
    cap: usize,
    weight: f64,
    valset_size: usize,
    /// The `merge` mode shifts the two-component score positive (dspy's `MERGE_BASE`), so a merge
    /// run's ancestor weights are never all zero. See the fixture generator.
    merge_mode: bool,
}

impl MirrorAdapter {
    fn versions(candidate: &Candidate) -> Vec<usize> {
        candidate
            .values()
            .map(|text| text[1..].parse().expect("a vN component text"))
            .collect()
    }

    fn score(&self, candidate: &Candidate, example_id: usize) -> f64 {
        let versions = Self::versions(candidate);
        let k = versions.len();
        let favored = versions[example_id % k];
        if k == 1 {
            favored as f64 * self.weight
        } else {
            let rival = versions[(example_id + 1) % k];
            let base = if self.merge_mode { 0.5 } else { 0.0 };
            (favored as f64 - rival as f64) * self.weight + base
        }
    }
}

impl GepaAdapter for MirrorAdapter {
    // Nothing here reports an output, so the type is the empty one.
    type Output = ();

    async fn evaluate_minibatch(
        &mut self,
        ids: &[usize],
        candidate: &Candidate,
        capture_traces: bool,
    ) -> EvalBatch<Self::Output> {
        let scores = ids.iter().map(|&id| self.score(candidate, id)).collect();
        if capture_traces {
            EvalBatch::traced(scores)
        } else {
            EvalBatch::scored(scores)
        }
    }

    async fn evaluate_valset(&mut self, candidate: &Candidate) -> EvalBatch<Self::Output> {
        EvalBatch::scored(
            (0..self.valset_size)
                .map(|id| self.score(candidate, id))
                .collect(),
        )
    }

    async fn evaluate_valset_ids(
        &mut self,
        ids: &[usize],
        candidate: &Candidate,
    ) -> EvalBatch<Self::Output> {
        EvalBatch::scored(ids.iter().map(|&id| self.score(candidate, id)).collect())
    }

    async fn propose_new_texts(
        &mut self,
        candidate: &Candidate,
        components: &[String],
        _captured: &EvalBatch<Self::Output>,
    ) -> Candidate {
        components
            .iter()
            .map(|name| {
                let version: usize = candidate[name][1..].parse().expect("a vN component text");
                (name.clone(), format!("v{}", (version + 1).min(self.cap)))
            })
            .collect()
    }
}

fn candidate_of(value: &Value) -> Candidate {
    value
        .as_object()
        .expect("a candidate object")
        .iter()
        .map(|(name, text)| {
            (
                name.clone(),
                text.as_str().expect("component text").to_string(),
            )
        })
        .collect()
}

/// A fixture parent list `[null]` (the seed) maps to no parents; `[0]`, `[1]`, ... to those indices.
fn parents_of(value: &Value) -> Vec<usize> {
    value
        .as_array()
        .expect("a parent list")
        .iter()
        .filter_map(|p| p.as_u64().map(|n| n as usize))
        .collect()
}

#[tokio::test]
async fn reproduces_the_runs_gepa_produces() {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/conformance/engine.json");
    let text = std::fs::read_to_string(&path).expect("the engine golden is committed");
    let fixture: Value = serde_json::from_str(&text).expect("the golden parses");
    let weight = fixture["weight"].as_f64().expect("weight");

    for case in fixture["cases"].as_array().expect("cases") {
        let label = case["label"].as_str().expect("label");
        let valset_size = case["valset_size"].as_u64().expect("valset_size") as usize;

        let merge_mode = case["mode"].as_str() == Some("merge");
        let engine = GepaEngine {
            adapter: MirrorAdapter {
                cap: case["cap"].as_u64().expect("cap") as usize,
                weight,
                valset_size,
                merge_mode,
            },
            trainset_size: case["trainset_size"].as_u64().expect("trainset_size") as usize,
            valset_size,
            minibatch_size: case["minibatch_size"].as_u64().expect("minibatch_size") as usize,
            max_metric_calls: case["max_metric_calls"].as_u64().expect("max_metric_calls") as usize,
            perfect_score: case["perfect_score"].as_f64().expect("perfect_score"),
            skip_perfect_score: true,
            use_merge: case["use_merge"].as_bool().unwrap_or(false),
            // From the case, not hard-coded: every case left this at gepa's default of 5 and
            // none accepted five merges, so the cap comparison, the `due` decrement and the
            // `total_tested` increment were all unreachable and five mutants of them survived.
            max_merge_invocations: case["max_merge_invocations"].as_u64().unwrap_or(5) as usize,
            seed: case["seed"].as_u64().expect("seed"),
            // The golden was recorded from gepa under its defaults; naming them keeps a later
            // change of default from silently re-pointing this at another strategy.
            candidate_selection_strategy: CandidateSelection::Pareto,
            track_best_outputs: false,
            progress: std::sync::Arc::new(Silent),
            component_selector: ComponentSelection::RoundRobin,
        };
        let outcome = engine.optimize(candidate_of(&case["seed_candidate"])).await;
        let result = &case["result"];

        let want_candidates: Vec<Candidate> = result["candidates"]
            .as_array()
            .expect("candidates")
            .iter()
            .map(candidate_of)
            .collect();
        assert_eq!(
            outcome.candidates, want_candidates,
            "{label}: candidate pool"
        );

        let want_parents: Vec<Vec<usize>> = result["parents"]
            .as_array()
            .expect("parents")
            .iter()
            .map(parents_of)
            .collect();
        assert_eq!(outcome.parents, want_parents, "{label}: parents");

        assert_eq!(
            outcome.best_idx,
            result["best_idx"].as_u64().expect("best_idx") as usize,
            "{label}: best_idx"
        );
        assert_eq!(
            outcome.total_num_evals,
            result["total_metric_calls"].as_u64().expect("total") as usize,
            "{label}: total_num_evals"
        );
        assert_eq!(
            outcome.num_full_ds_evals,
            result["num_full_val_evals"].as_u64().expect("full") as usize,
            "{label}: num_full_ds_evals"
        );

        let want_discovery: Vec<usize> = result["discovery_eval_counts"]
            .as_array()
            .expect("discovery")
            .iter()
            .map(|n| n.as_u64().unwrap() as usize)
            .collect();
        assert_eq!(
            outcome.num_metric_calls_by_discovery, want_discovery,
            "{label}: discovery eval counts"
        );

        let want_scores = result["val_aggregate_scores"].as_array().expect("scores");
        assert_eq!(
            outcome.val_aggregate_scores.len(),
            want_scores.len(),
            "{label}: score count"
        );
        for (idx, (got, want)) in outcome
            .val_aggregate_scores
            .iter()
            .zip(want_scores)
            .enumerate()
        {
            assert!(
                (got - want.as_f64().unwrap()).abs() < 1e-12,
                "{label}: val_aggregate_scores[{idx}] {got} vs {want}"
            );
        }
    }
}
