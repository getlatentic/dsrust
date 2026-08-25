//! dspy `teleprompt/random_search.py`: bootstrap several times and keep the best.
//!
//! [`BootstrapFewShot`] produces one demo set from one walk of the trainset. This runs that walk
//! nineteen times over — each with the trainset shuffled differently and a different number of
//! demos asked for — scores each result on a validation set, and keeps the winner. It is the
//! cheapest real optimizer in dspy: no proposal model, no search algorithm, just more attempts.
//!
//! The first three attempts are not random at all, and that is the point of the negative seeds:
//! zero-shot, labels only, and one unshuffled bootstrap. A random search that never tried the
//! obvious baselines could report an improvement over nothing.

use anyhow::{Result, bail};
use pyrng::Random;

use super::{BootstrapFewShot, LabeledFewShot};
use crate::example::{Example, Prediction};
use crate::module::{Module, ProgramState};

/// Reading a traced run: every attempt is kept with the program state that produced it, so a caller
/// can go back to one the search did not keep.
///
/// ```no_run
/// # use dsrust::optimize::Attempt;
/// # fn read(attempts: Vec<Attempt>) {
/// // The seeds are dspy's: -3 is the zero-shot program, -2 the labeled-only one, and 0 upward are
/// // the randomly sampled sets — so a negative seed is a baseline rather than a search result.
/// let searched: Vec<&Attempt> = attempts.iter().filter(|a| a.seed >= 0).collect();
/// let best = attempts.iter().max_by(|a, b| a.score.total_cmp(&b.score));
/// if let Some(best) = best {
///     println!("seed {} scored {:.1}% of {} attempts", best.seed, best.score, searched.len());
/// }
/// # }
/// ```
/// One attempt and what it scored, in the order dspy records them.
#[derive(Debug, Clone)]
pub struct Attempt {
    /// dspy's `seed`: `-3` zero-shot, `-2` labels only, `-1` an unshuffled bootstrap, then the
    /// shuffled ones from zero up.
    pub seed: i64,
    pub score: f64,
    /// The program this attempt produced, as the state that would restore it.
    pub state: ProgramState,
}

/// dspy `BootstrapFewShotWithRandomSearch`: bootstrap under several shuffles, keep the best.
pub struct BootstrapRandomSearch<M> {
    metric: M,
    max_bootstrapped_demos: usize,
    max_labeled_demos: usize,
    max_rounds: usize,
    num_candidate_programs: usize,
    stop_at_score: Option<f64>,
    metric_threshold: Option<f64>,
    /// dspy's `min_num_samples`, fixed at 1 on the instance and never exposed as a constructor
    /// argument — kept here because the size draw reads it.
    min_num_samples: usize,
    /// What a scoring pass is bounded by. See [`Scoring`](super::Scoring).
    scoring: super::Scoring,
    /// dspy `labeled_sample`: whether the labels-only candidate draws its demos or takes the first
    /// few. See [`labeled_sample`](Self::labeled_sample).
    labeled_sample: bool,
    /// dspy `restrict`: which seeds to attempt at all. See [`restrict`](Self::restrict).
    restrict: Option<Vec<i64>>,
}

impl<M> BootstrapRandomSearch<M>
where
    M: Fn(&Example, &Prediction) -> f64 + Send + Sync,
{
    /// dspy's defaults: four bootstrapped demos, sixteen labelled, sixteen candidate sets.
    pub fn new(metric: M) -> Self {
        Self {
            metric,
            max_bootstrapped_demos: 4,
            max_labeled_demos: 16,
            max_rounds: 1,
            num_candidate_programs: 16,
            stop_at_score: None,
            metric_threshold: None,
            min_num_samples: 1,
            labeled_sample: true,
            restrict: None,
            scoring: super::Scoring::default(),
        }
    }

    /// The upper end of the per-attempt demo draw — dspy's `max_bootstrapped_demos`, which it
    /// stores as `max_num_samples`.
    /// What each scoring pass is bounded by — dspy's `num_threads` and `max_errors`.
    pub fn scoring(mut self, scoring: super::Scoring) -> Self {
        self.scoring = scoring;
        self
    }

    pub fn max_bootstrapped_demos(mut self, demos: usize) -> Self {
        self.max_bootstrapped_demos = demos;
        self
    }

    pub fn max_labeled_demos(mut self, demos: usize) -> Self {
        self.max_labeled_demos = demos;
        self
    }

    pub fn max_rounds(mut self, rounds: usize) -> Self {
        self.max_rounds = rounds;
        self
    }

    /// How many *shuffled* attempts to make. The three baselines run regardless, so a search of
    /// `n` candidate programs makes `n + 3` attempts — as upstream's `range(-3, n)` does.
    pub fn num_candidate_programs(mut self, programs: usize) -> Self {
        self.num_candidate_programs = programs;
        self
    }

    /// Stop as soon as an attempt scores at least this. dspy compares with `>=`.
    pub fn stop_at_score(mut self, score: f64) -> Self {
        self.stop_at_score = Some(score);
        self
    }

    /// Read the bootstrap metric as a number against this bar rather than as a yes/no. See
    /// [`BootstrapFewShot::metric_threshold`].
    pub fn metric_threshold(mut self, threshold: f64) -> Self {
        self.metric_threshold = Some(threshold);
        self
    }

    /// dspy `labeled_sample`: whether the labels-only candidate draws its demos at random or takes
    /// the first few in order, on by default as upstream's is.
    ///
    /// It reaches one attempt of the search — seed `-2`, which is `LabeledFewShot` alone — and none
    /// of the others, so turning it off changes one candidate out of `n + 3` rather than the run.
    pub fn labeled_sample(mut self, labeled_sample: bool) -> Self {
        self.labeled_sample = labeled_sample;
        self
    }

    /// dspy `restrict`: attempt only these seeds, skipping the rest.
    ///
    /// The three baselines are `-3`, `-2` and `-1`; a candidate program is `0` upward. Unset
    /// attempts them all, which is upstream's `restrict=None`.
    ///
    /// Skipping does not move the others: each shuffled attempt seeds its own generator with its own
    /// seed, so what seed 4 draws is the same whether or not seed 3 ran. That is what makes this
    /// usable for resuming a search or narrowing one, rather than a way to get a different answer.
    pub fn restrict(mut self, seeds: impl IntoIterator<Item = i64>) -> Self {
        self.restrict = Some(seeds.into_iter().collect());
        self
    }

    /// dspy's seed sequence: the three baselines, then one per candidate program, less whatever
    /// `restrict` excludes.
    fn seeds(&self) -> impl Iterator<Item = i64> + '_ {
        (-3..self.num_candidate_programs as i64).filter(move |seed| {
            self.restrict
                .as_ref()
                .is_none_or(|only| only.contains(seed))
        })
    }

    /// The trainset this seed bootstraps from, and how many demos it asks for.
    ///
    /// Two separately seeded generators, which is upstream's `random.Random(seed).shuffle(...)`
    /// followed by `random.Random(seed).randint(...)`. Reusing one would draw the size from a
    /// stream the shuffle had already advanced, and every attempt would ask for a different number
    /// of demos than dspy's.
    fn shuffled(&self, seed: i64, trainset: &[Example]) -> (Vec<Example>, usize) {
        let mut shuffled = trainset.to_vec();
        Random::seeded(seed as u64).shuffle(&mut shuffled);
        let size = Random::seeded(seed as u64).randint(
            self.min_num_samples as u64,
            self.max_bootstrapped_demos as u64,
        ) as usize;
        (shuffled, size)
    }

    fn bootstrap(&self, demos: usize) -> BootstrapFewShot<&M> {
        let mut optimizer = BootstrapFewShot::new(&self.metric);
        optimizer.max_bootstrapped_demos = demos;
        optimizer.max_labeled_demos = self.max_labeled_demos;
        optimizer.max_rounds = self.max_rounds;
        optimizer.metric_threshold = self.metric_threshold;
        optimizer
    }

    /// Run every attempt, score each on `valset`, and leave `student` holding the best.
    ///
    /// Answers with every attempt in decreasing score, which is dspy's `candidate_programs`
    /// attached to the returned program. dspy defaults `valset` to the trainset; pass the trainset
    /// for both to match that.
    pub async fn compile<S: Module + ?Sized>(
        &self,
        student: &mut S,
        trainset: &[Example],
        valset: &[Example],
    ) -> Result<Vec<Attempt>> {
        // dspy's own guard, before any attempt is made: a `restrict` naming no seed in range leaves
        // nothing to evaluate, and the loop below would simply not run. Answering with an empty
        // list would hand the caller back their unchanged student as though the search had been
        // done and found nothing worth keeping — a typo'd seed and an exhausted search are not the
        // same answer, and only one of them is the caller's mistake.
        if let Some(only) = &self.restrict
            && self.seeds().next().is_none()
        {
            bail!(
                "`restrict` {only:?} does not match any candidate seed in \
                 -3..{}; no candidate programs would be evaluated.",
                self.num_candidate_programs
            );
        }
        // The state the student arrived in, so every attempt starts from the same program rather
        // than from whatever the previous one left behind — dspy gets that from `reset_copy`.
        let start = student.dump_state();
        let mut attempts: Vec<Attempt> = Vec::new();

        for seed in self.seeds() {
            student.load_state(&start)?;
            match seed {
                // Zero-shot: the student as it came, with no demos at all.
                -3 => {}
                -2 => LabeledFewShot {
                    sample: self.labeled_sample,
                    ..LabeledFewShot::new(self.max_labeled_demos)
                }
                .compile(student, trainset),
                // The unshuffled bootstrap, asking for the full demo budget rather than a draw.
                -1 => {
                    self.bootstrap(self.max_bootstrapped_demos)
                        .compile(student, trainset)
                        .await?;
                }
                _ => {
                    let (shuffled, demos) = self.shuffled(seed, trainset);
                    self.bootstrap(demos).compile(student, &shuffled).await?;
                }
            }

            let state = student.dump_state();
            let scored = self
                .scoring
                .apply(crate::evaluate::Evaluate::new(
                    valset.to_vec(),
                    |example| student.forward(example),
                    &self.metric,
                ))
                .run()
                .await
                .score;
            attempts.push(Attempt {
                seed,
                score: scored,
                state,
            });
            if self.stop_at_score.is_some_and(|bar| scored >= bar) {
                break;
            }
        }

        if let Some((at, _)) = winner(&attempts) {
            student.load_state(&attempts[at].state)?;
        }

        let mut ranked = attempts;
        // dspy sorts by score descending; a stable sort keeps ties in the order they were tried.
        ranked.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(ranked)
    }
}

/// dspy keeps the *first* attempt to beat every earlier one — `score > max(scores)` — so a later
/// tie does not displace an earlier winner. Its own function because the rule is the whole of
/// which program a search hands back, and three mutants of the inline fold survived unnoticed.
fn winner(attempts: &[Attempt]) -> Option<(usize, f64)> {
    attempts
        .iter()
        .enumerate()
        .fold(None::<(usize, f64)>, |best, (at, attempt)| match best {
            Some((_, high)) if attempt.score <= high => best,
            _ => Some((at, attempt.score)),
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimize::scripted::{Answers, Solver, trainset};

    /// The winner rule on its own: strictly-better replaces, a tie never does.
    #[test]
    fn the_winner_is_the_first_to_beat_every_earlier_attempt() {
        let attempt = |seed: i64, score: f64| Attempt {
            seed,
            score,
            state: ProgramState::new(Default::default()),
        };
        assert_eq!(winner(&[]), None);
        let tie = [attempt(-3, 1.0), attempt(-2, 1.0)];
        assert_eq!(
            winner(&tie).map(|(at, _)| at),
            Some(0),
            "a tie keeps the earlier"
        );
        let later_better = [attempt(-3, 0.5), attempt(-2, 0.9), attempt(-1, 0.9)];
        assert_eq!(winner(&later_better).map(|(at, _)| at), Some(1));
        let earlier_better = [attempt(-3, 0.9), attempt(-2, 0.5)];
        assert_eq!(winner(&earlier_better).map(|(at, _)| at), Some(0));
    }

    /// Each special seed, run alone through `restrict`, leaves the student in its own shape:
    /// `-3` zero-shot, `-2` the first `k` labels when `labeled_sample` is off — dspy's seeded
    /// draw would take rows 3 and 4, measured, so the deleted-field mutant cannot pass — and
    /// `-1` the unshuffled bootstrap, whose demos carry the `augmented` mark.
    #[tokio::test]
    async fn each_special_seed_leaves_its_own_shape() {
        let metric = crate::evaluate::exact_match;
        let rows = trainset();

        let mut student = Solver::new(Answers::Correctly);
        BootstrapRandomSearch::new(&metric)
            .restrict([-3])
            .compile(&mut student, &rows, &rows)
            .await
            .expect("compiles");
        assert!(student.demos.is_empty(), "zero-shot leaves no demos");

        let mut student = Solver::new(Answers::Correctly);
        BootstrapRandomSearch::new(&metric)
            .restrict([-2])
            .max_labeled_demos(2)
            .labeled_sample(false)
            .compile(&mut student, &rows, &rows)
            .await
            .expect("compiles");
        let questions: Vec<_> = student
            .demos
            .iter()
            .map(|demo| demo.get("question").cloned().expect("a question"))
            .collect();
        let first_two: Vec<_> = rows[..2]
            .iter()
            .map(|row| row.get("question").cloned().expect("a question"))
            .collect();
        assert_eq!(
            questions, first_two,
            "sample=false takes the first k in order"
        );

        let mut student = Solver::new(Answers::Correctly);
        BootstrapRandomSearch::new(&metric)
            .restrict([-1])
            .max_bootstrapped_demos(1)
            .max_labeled_demos(0)
            .compile(&mut student, &rows, &rows)
            .await
            .expect("compiles");
        // Only the bootstrap arm can earn a demo the metric accepted, and only an earned demo
        // carries the marker — so the two together say which arm ran and that it wrote what dspy
        // writes. `Solver` records no trace, which is the arm that used to drop the marker.
        assert_eq!(
            crate::optimize::scripted::answers(&student.demos),
            ["Paris"],
            "the bootstrap arm earned its demo: {:?}",
            student.demos
        );
        assert_eq!(
            student.demos[0].get("augmented"),
            Some(&serde_json::json!(true)),
            "an earned demo is marked: {:?}",
            student.demos
        );
    }

    /// `stop_at_score` reads `scored >= bar`: a met bar ends the search after that attempt, an
    /// unmet bar never does. The valset is the two capital rows `Correctly` actually solves, so
    /// the score is 1.0 and the metric-call count says how many attempts ran — the one observer
    /// that catches the comparison mutants in both directions.
    #[tokio::test]
    async fn stop_at_score_stops_exactly_when_the_bar_is_met() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let rows = trainset();
        let capitals = &rows[..2];
        let calls = AtomicUsize::new(0);
        let metric = |example: &Example, prediction: &Prediction| {
            calls.fetch_add(1, Ordering::SeqCst);
            crate::evaluate::exact_match(example, prediction)
        };

        let mut student = Solver::new(Answers::Correctly);
        BootstrapRandomSearch::new(&metric)
            .restrict([-3, -2])
            .stop_at_score(0.5)
            .compile(&mut student, &rows, capitals)
            .await
            .expect("compiles");
        let stopped = calls.swap(0, Ordering::SeqCst);
        assert_eq!(
            stopped,
            capitals.len(),
            "one attempt scored, then the bar ended it"
        );

        let mut student = Solver::new(Answers::Correctly);
        BootstrapRandomSearch::new(&metric)
            .restrict([-3, -2])
            .stop_at_score(2.0)
            .compile(&mut student, &rows, capitals)
            .await
            .expect("compiles");
        let ran_on = calls.load(Ordering::SeqCst);
        assert_eq!(
            ran_on,
            capitals.len() * 2,
            "an unreachable bar stops nothing"
        );
    }

    /// Which attempts each search makes, against the seeds dspy's own `score_data` recorded.
    ///
    /// `restrict` is only observable as which seeds are *absent*, so this compares the whole
    /// sequence rather than a count — and the golden carries a case naming a seed the range never
    /// reaches, which must be dropped rather than added or refused.
    #[test]
    fn the_attempts_are_the_ones_dspy_makes() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/conformance/optimize/random_search.json");
        let text = std::fs::read_to_string(&path).expect("the random-search golden is committed");
        let golden: serde_json::Value = serde_json::from_str(&text).expect("the golden parses");
        let cases = golden["cases"].as_array().expect("cases");
        assert!(!cases.is_empty(), "the golden records no cases");

        for case in cases {
            let mut search = BootstrapRandomSearch::new(|_: &Example, _: &Prediction| 0.0)
                .num_candidate_programs(
                    case["num_candidate_programs"].as_u64().expect("programs") as usize
                );
            if let Some(only) = case["restrict"].as_array() {
                search = search.restrict(only.iter().map(|seed| seed.as_i64().expect("a seed")));
            }
            let theirs: Vec<i64> = case["seeds"]
                .as_array()
                .expect("seeds")
                .iter()
                .map(|seed| seed.as_i64().expect("a seed"))
                .collect();
            assert_eq!(
                search.seeds().collect::<Vec<_>>(),
                theirs,
                "attempts for {case}"
            );
        }
    }

    /// `labeled_sample` reaches exactly one attempt — seed `-2`, the labels-only one — and is only
    /// observable in the demos it keeps. The golden records both arms on a trainset where drawing
    /// and taking-in-order disagree, and its generator refuses to write one where they do not.
    #[tokio::test]
    async fn labeled_sample_decides_what_the_labels_only_attempt_keeps() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/conformance/optimize/random_search.json");
        let text = std::fs::read_to_string(&path).expect("the random-search golden is committed");
        let golden: serde_json::Value = serde_json::from_str(&text).expect("the golden parses");
        let trainset: Vec<Example> = golden["trainset"]
            .as_array()
            .expect("trainset")
            .iter()
            .map(|row| {
                crate::example! {
                    question: row["question"].clone(),
                    answer: row["answer"].clone(),
                }
                .with_inputs(["question"])
            })
            .collect();

        let mut compared = 0;
        for case in golden["cases"].as_array().expect("cases") {
            if case["restrict"].as_array().map(Vec::as_slice) != Some(&[serde_json::json!(-2)]) {
                continue;
            }
            compared += 1;
            let sample = case["labeled_sample"].as_bool().expect("labeled_sample");
            let mut student = crate::predict::Predict::parse("question -> answer").expect("parses");
            LabeledFewShot {
                sample,
                ..LabeledFewShot::new(2)
            }
            .compile(&mut student, &trainset);

            let theirs: Vec<&str> = case["labels_only_demos"]
                .as_array()
                .expect("demos")
                .iter()
                .map(|demo| demo["question"].as_str().expect("a question"))
                .collect();
            let ours: Vec<&str> = student
                .demos
                .iter()
                .map(|demo| demo.get("question").and_then(|q| q.as_str()).unwrap_or(""))
                .collect();
            assert_eq!(ours, theirs, "labels-only demos at labeled_sample={sample}");
        }
        assert_eq!(
            compared, 2,
            "the golden should record both arms of labeled_sample"
        );
    }

    /// A `restrict` that matches no seed in range is refused, not answered with nothing.
    ///
    /// The loop over `seeds()` simply does not run in that case, so `compile` returned an empty
    /// list of attempts and left the student exactly as it arrived — indistinguishable from a
    /// search that ran and found nothing better. dspy raises here, and the golden beside this test
    /// never covered it: its one out-of-range case is `restrict: [1, 99]`, where `1` is in range
    /// and something still runs.
    #[tokio::test]
    async fn a_restrict_matching_no_seed_is_refused_rather_than_answered_with_nothing() {
        let trainset =
            vec![crate::example! { question: "q", answer: "a" }.with_inputs(["question"])];
        let mut student = crate::predict::Predict::parse("question -> answer").expect("parses");
        let refused = BootstrapRandomSearch::new(|_: &Example, _: &Prediction| 0.0)
            .num_candidate_programs(2)
            .restrict([99])
            .compile(&mut student, &trainset, &trainset)
            .await;
        let why = refused
            .expect_err("no seed to attempt is an error")
            .to_string();
        assert!(
            why.contains("restrict"),
            "the message names the knob: {why}"
        );
        assert!(
            why.contains("-3..2"),
            "and the range it had to fall in: {why}"
        );
    }

    /// What a shuffled attempt draws depends on its own seed and nothing else, so restricting the
    /// search does not change what the surviving attempts do. That is the property that makes
    /// `restrict` usable for resuming a search rather than a way to get a different answer.
    #[test]
    fn restricting_does_not_move_what_the_remaining_seeds_draw() {
        let trainset: Vec<Example> = (0..8)
            .map(|n| crate::example! { question: format!("q{n}") })
            .collect();
        let whole =
            BootstrapRandomSearch::new(|_: &Example, _: &Prediction| 0.0).num_candidate_programs(5);
        let narrowed = BootstrapRandomSearch::new(|_: &Example, _: &Prediction| 0.0)
            .num_candidate_programs(5)
            .restrict([3]);
        assert_eq!(
            whole.shuffled(3, &trainset),
            narrowed.shuffled(3, &trainset)
        );
    }

    /// The seed sequence is three baselines then one per candidate program, which is what makes a
    /// "16 candidate" search 19 attempts.
    #[test]
    fn the_seeds_are_three_baselines_then_the_candidates() {
        let search = BootstrapRandomSearch::new(|_: &Example, _: &Prediction| 0.0);
        assert_eq!(search.seeds().collect::<Vec<_>>()[..4], [-3, -2, -1, 0]);
        assert_eq!(
            search.seeds().count(),
            19,
            "dspy's default 16 candidates plus 3 baselines"
        );
        assert_eq!(
            search.num_candidate_programs(2).seeds().collect::<Vec<_>>(),
            [-3, -2, -1, 0, 1]
        );
    }

    /// The size draw is a *fresh* generator, not the one the shuffle advanced. Sharing one would
    /// ask for a different number of demos on every seed than dspy asks for.
    #[test]
    fn the_size_is_drawn_from_a_fresh_generator() {
        let search =
            BootstrapRandomSearch::new(|_: &Example, _: &Prediction| 0.0).max_bootstrapped_demos(8);
        let trainset: Vec<Example> = (0..20)
            .map(|n| Example::new([("q", serde_json::json!(n))]))
            .collect();

        for seed in 0..5 {
            let (_, drawn) = search.shuffled(seed, &trainset);
            let expected = Random::seeded(seed as u64).randint(1, 8) as usize;
            assert_eq!(
                drawn, expected,
                "seed {seed} draws from a generator of its own"
            );
            assert!((1..=8).contains(&drawn), "and stays within the range");
        }
    }

    /// The same seed shuffles the same way, and different seeds differently — the whole reason the
    /// search is worth running.
    #[test]
    fn a_seed_shuffles_reproducibly() {
        let search = BootstrapRandomSearch::new(|_: &Example, _: &Prediction| 0.0);
        let trainset: Vec<Example> = (0..10)
            .map(|n| Example::new([("q", serde_json::json!(n))]))
            .collect();
        let order = |seed| {
            search
                .shuffled(seed, &trainset)
                .0
                .iter()
                .map(|example| example.get("q").cloned().unwrap_or_default())
                .collect::<Vec<_>>()
        };
        assert_eq!(order(1), order(1));
        assert_ne!(order(1), order(2));
        assert_ne!(
            order(1),
            order(-1_i64.abs()),
            "and none of them is the original order"
        );
    }
}
