//! gepa's other two candidate selectors, against the package itself.
//!
//! dspy annotates `candidate_selection_strategy` as `Literal["pareto", "current_best"]` and passes
//! the string straight to `gepa.optimize`, whose factory map also holds `epsilon_greedy` (eps=0.1)
//! and `top_k_pareto` (k=5). Both are reachable from a dspy call.
//!
//! What is compared is the selection *and* where the generator was left, because these two differ in
//! how far they advance it: epsilon-greedy takes one draw or two depending on the coin, and
//! top-k-pareto takes none when the filtered mapping empties. A port that agrees on every selection
//! and advances the generator differently diverges on the round after.

use gepa::pyset::PyIntSet;
use pyrng::Random;
use serde_json::Value;

fn golden() -> Value {
    serde_json::from_str(include_str!("conformance/gepa_selectors.json"))
        .expect("the golden parses")
}

fn fronts_of(case: &Value) -> Vec<PyIntSet> {
    case["fronts"]
        .as_array()
        .expect("fronts")
        .iter()
        .map(|front| {
            PyIntSet::from_keys(
                front
                    .as_array()
                    .expect("a front")
                    .iter()
                    .map(|index| index.as_u64().expect("an index") as usize),
            )
        })
        .collect()
}

fn scores_of(case: &Value) -> Vec<f64> {
    case["scores"]
        .as_array()
        .expect("scores")
        .iter()
        .map(|score| score.as_f64().expect("a score"))
        .collect()
}

/// Both selectors, every case, every seed — the pick and the generator's position afterwards.
#[test]
fn both_selectors_choose_what_gepa_chooses() {
    let golden = golden();
    let seeds: Vec<u64> = golden["seeds"]
        .as_array()
        .expect("seeds")
        .iter()
        .map(|seed| seed.as_u64().expect("a seed"))
        .collect();

    for (case_index, case) in golden["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .enumerate()
    {
        let fronts = fronts_of(case);
        let scores = scores_of(case);

        for (arm, selection) in [
            // All four, not only the two that were new: `current_best` was built from reading
            // gepa's source and had never been compared against it.
            ("pareto", gepa::CandidateSelection::Pareto),
            ("current_best", gepa::CandidateSelection::CurrentBest),
            (
                "epsilon_greedy",
                gepa::CandidateSelection::EpsilonGreedy {
                    epsilon: case["epsilon_greedy"]["epsilon"].as_f64().expect("epsilon"),
                },
            ),
            (
                "top_k_pareto",
                gepa::CandidateSelection::TopKPareto {
                    k: case["top_k_pareto"]["k"].as_u64().expect("k") as usize,
                },
            ),
        ] {
            let expected = &case[arm];
            for (at, &seed) in seeds.iter().enumerate() {
                let mut rng = Random::seeded(seed);
                let picked = gepa::select_with(selection, &fronts, &scores, &mut rng);
                assert_eq!(
                    picked as u64,
                    expected["picks"][at].as_u64().expect("a pick"),
                    "{arm} case {case_index} seed {seed}"
                );
                // Where the generator was left. Agreeing on the pick and not on this is a port that
                // diverges on the *next* round rather than this one.
                assert_eq!(
                    rng.random(),
                    expected["after"][at].as_f64().expect("after"),
                    "{arm} case {case_index} seed {seed}: generator left in a different place"
                );
            }
        }
    }
}

/// Both component selectors, against the package: which components one reflection rewrites, and
/// where round-robin leaves the cursor.
///
/// `All` was built from reading gepa's source and had never been compared against it — the same gap
/// `current_best` had. The cursor matters as much as the choice: it is inherited by every candidate
/// a family produces, so a round that advances it differently diverges a generation later.
///
/// The golden's components are deliberately *not* alphabetical — sorted they are hints,
/// instructions, style — because that is the only arrangement in which a walk over a sorted map and
/// a walk over gepa's insertion-ordered dict disagree. They did, until `Candidate` stopped being a
/// `BTreeMap`; every other golden here is single-component or happens to sort into declaration
/// order, which is why nothing else caught it.
#[test]
fn both_component_selectors_choose_what_gepa_chooses() {
    let golden = golden();
    let components = &golden["components"];
    // Read from the recorded list, not from the `candidate` object: JSON keeps the order but
    // serde_json's default map does not, so parsing the object would silently sort it back and the
    // test would pass against exactly the bug it exists to catch.
    let declared: Vec<String> = components["declaration_order"]
        .as_array()
        .expect("the golden records the declaration order")
        .iter()
        .map(|name| name.as_str().expect("a component").to_owned())
        .collect();

    let rounds = components["rounds"].as_array().expect("rounds");
    let mut ours = Vec::new();
    let mut theirs = Vec::new();

    for round in rounds {
        let cursor = round["cursor"].as_u64().expect("cursor") as usize;
        let mut state = gepa::GepaState::for_components(declared.clone(), cursor);
        ours.push(state.select_component(0));
        theirs.push(
            round["round_robin"][0]
                .as_str()
                .expect("a component")
                .to_owned(),
        );

        // The cursor advances the same way whatever the order, which is the half that is *not*
        // diverging: a new candidate inherits it, so getting this wrong would move a generation.
        assert_eq!(
            state.next_component_for(0) as u64,
            round["advanced_to"].as_u64().expect("advanced_to"),
            "round-robin cursor after {cursor}"
        );

        // `All` is every component, so it differs from gepa's only in order — a different *set*
        // would be a real bug rather than this one.
        let every: Vec<String> = round["all"]
            .as_array()
            .expect("all")
            .iter()
            .map(|name| name.as_str().expect("a component").to_owned())
            .collect();
        assert_eq!(
            declared, every,
            "`all` selected the components in a different order"
        );
    }

    assert_eq!(ours, theirs, "the round-robin walk should be gepa's");
    // And it is declaration order that it walks, not some other order that happens to match here.
    assert_eq!(
        ours, declared,
        "the walk should follow the declaration order"
    );

    // Through the real constructor, which is what a run uses: `GepaState::new` reads the walk off
    // the seed candidate's keys. Handing `for_components` a list above proves the state walks what
    // it is given and says nothing about what a `Candidate` gives it — so a `Candidate` that sorted
    // its keys again would leave every assertion above green.
    let mut seed = gepa::Candidate::new();
    for name in &declared {
        seed.insert(name.clone(), String::new());
    }
    let mut from_seed = gepa::GepaState::new(seed, vec![1.0]);
    let walk: Vec<String> = (0..declared.len())
        .map(|_| from_seed.select_component(0))
        .collect();
    assert_eq!(
        walk, theirs,
        "the walk a seed candidate produces should be gepa's"
    );
}
