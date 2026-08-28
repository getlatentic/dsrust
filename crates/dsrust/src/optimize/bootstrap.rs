//! dspy `BootstrapFewShot` (`teleprompt/bootstrap.py`): demos a program earned rather than
//! demos it was handed.

use anyhow::{Result, anyhow};

use crate::example::{Example, Prediction};
use crate::lm::Sampling;
use crate::module::Module;

use super::Optimizer;
use super::earned::{Bootstrapped, Solved};
use super::labeled::LabeledFewShot;
use super::rng::Rng;

/// Keep only the demos the program can actually solve.
///
/// A teacher runs the trainset, a metric scores each attempt, and the attempts that pass become
/// the student's demos. The difference from [`LabeledFewShot`] is the difference between showing
/// a program any labelled example and showing it examples of its own successful behaviour.
///
/// The teacher is the program that runs; the student is the program that keeps the result. dspy
/// defaults the teacher to a deep copy of the student, which is what [`Self::compile`] does in
/// place: the student teaches itself, and nothing is written back to it until every example has
/// been attempted. [`Self::compile_with_teacher`] takes a separate program, which is how a
/// strong model teaches a cheap one.
///
/// Whichever program teaches is first primed with labelled demos by [`LabeledFewShot`], out of
/// the same [`max_labeled_demos`](Self::max_labeled_demos) budget the student's own labelled
/// demos come from.
pub struct BootstrapFewShot<M> {
    /// Scores one attempt against the example it came from. dspy allows no metric at all,
    /// meaning every attempt that ran is kept; that is [`Self::without_metric`].
    ///
    /// dspy passes the trace as a third argument, which is how a metric tells optimizing from
    /// evaluating: the convention is `if trace is not None: return score > bar`, so a metric
    /// hands an optimizer a yes/no where it hands an evaluator a number. There is no trace to
    /// pass, so a metric here always returns the number and this reads it for truth — the same
    /// answer the convention arrives at, decided on this side of the call.
    pub metric: Option<M>,
    /// Read the metric as a number against this bar rather than as a yes/no.
    ///
    /// dspy guards this with `if self.metric_threshold`, and Python reads `0.0` as false, so a
    /// threshold of zero upstream means *no threshold* rather than a bar at zero. `Some(0.0)`
    /// keeps that reading, because a config carried over from Python would otherwise change
    /// meaning on the way across.
    pub metric_threshold: Option<f64>,
    /// Stop once this many examples have been solved. dspy's default is 4.
    pub max_bootstrapped_demos: usize,
    /// The demo budget per predictor. Bootstrapped demos are spent from it first and labelled
    /// examples fill whatever is left. dspy's default is 16.
    pub max_labeled_demos: usize,
    /// How many times one example may be attempted before the walk moves on. dspy's default
    /// is 1.
    ///
    /// Each round after the first is asked as a fresh rollout at `temperature = 1.0`, matching
    /// dspy, so a second attempt at an example is a genuinely different ask rather than a repeat
    /// of the one that just failed. The teacher is put back the way it was found afterwards.
    pub max_rounds: usize,
    /// How many failures the compile absorbs before giving up and returning the last one.
    ///
    /// dspy reads `max_errors=None` off `dspy.settings.max_errors`, whose default is 10. There
    /// is no settings object here, so 10 is the default outright.
    pub max_errors: usize,
}

impl<M> BootstrapFewShot<M> {
    fn configured(metric: Option<M>) -> Self {
        Self {
            metric,
            metric_threshold: None,
            max_bootstrapped_demos: 4,
            max_labeled_demos: 16,
            max_rounds: 1,
            max_errors: 10,
        }
    }

    /// Score every attempt with `metric` and keep the ones it accepts.
    pub fn new(metric: M) -> Self {
        Self::configured(Some(metric))
    }
}

impl BootstrapFewShot<fn(&Example, &Prediction) -> f64> {
    /// dspy's `metric=None`: keep every attempt the teacher completed without erroring.
    pub fn without_metric() -> Self {
        Self::configured(None)
    }
}

impl<M> BootstrapFewShot<M>
where
    M: crate::evaluate::Metric,
{
    /// dspy `compile(student, teacher=None, trainset=...)`: the student teaches itself.
    ///
    /// Answers with the number of examples that were solved, which is how many bootstrapped
    /// demos each predictor received.
    pub async fn compile<S: Module + ?Sized>(
        &self,
        student: &mut S,
        trainset: &[Example],
    ) -> Result<usize> {
        let bootstrapped = self.bootstrap(student, trainset).await?;
        Ok(self.train(student, bootstrapped))
    }

    /// dspy `compile(student, teacher=teacher, trainset=...)`: a separate program produces the
    /// demos and the student keeps them.
    pub async fn compile_with_teacher<S, T>(
        &self,
        student: &mut S,
        teacher: &mut T,
        trainset: &[Example],
    ) -> Result<usize>
    where
        S: Module + ?Sized,
        T: Module + ?Sized,
    {
        same_shape(student, teacher)?;
        let bootstrapped = self.bootstrap(teacher, trainset).await?;
        Ok(self.train(student, bootstrapped))
    }

    /// dspy `_bootstrap`: walk the trainset until the budget is spent, keeping what solved.
    async fn bootstrap<T: Module + ?Sized>(
        &self,
        teacher: &mut T,
        trainset: &[Example],
    ) -> Result<Bootstrapped> {
        // dspy skips this when `max_labeled_demos` is zero, leaving the teacher whatever demos
        // it arrived with. It also skips it for an already-compiled teacher; `Module` carries no
        // `_compiled` flag to read, so a teacher that was compiled elsewhere is primed again.
        if self.max_labeled_demos > 0 {
            LabeledFewShot::new(self.max_labeled_demos).compile(teacher, trainset);
        }

        let mut earned = Bootstrapped::empty();
        let mut solved = vec![false; trainset.len()];
        let mut errors = 0;
        // dspy counts the examples it solved, not the demos they produced. With several
        // predictors one example earns several demos, so the two part company here.
        let mut kept = 0;
        for (index, example) in trainset.iter().enumerate() {
            if kept >= self.max_bootstrapped_demos {
                break;
            }
            // What the teacher asks for when left alone, put back once this example is done so
            // one example's rollout cannot leak into the next one's first round.
            let resting = resting_config(teacher);
            for round in 0..self.max_rounds {
                // dspy makes every round after the first a fresh rollout at temperature 1.0. The
                // rollout id is what misses the cache, and the temperature is what makes the
                // answer differ once it does; without both, a re-ask is the same ask.
                if round > 0 {
                    teacher.set_config(Sampling::rollout(round as u64));
                }
                match self.attempt(teacher, example).await {
                    Ok(Some(demo)) => {
                        earned.file(demo);
                        solved[index] = true;
                        kept += 1;
                        break;
                    }
                    // Scored too low to keep. Not a failure, and not charged to the budget.
                    Ok(None) => {}
                    Err(error) => {
                        errors += 1;
                        if errors >= self.max_errors {
                            restore_config(teacher, &resting);
                            return Err(error);
                        }
                        tracing::error!(%error, "failed to run or to evaluate an example");
                    }
                }
            }
            restore_config(teacher, &resting);
        }

        let mut validation: Vec<Example> = trainset
            .iter()
            .zip(&solved)
            .filter(|(_, solved)| !**solved)
            .map(|(example, _)| example.clone())
            .collect();
        Rng::seeded(0).shuffle(&mut validation);
        earned.validation = validation;
        Ok(earned)
    }

    /// dspy `_bootstrap_one_example`: one round of asking the teacher to solve one example.
    ///
    /// `Ok(None)` is an attempt that ran and scored too low. `Err` is dspy's `except`: the
    /// program raised, or the example never declared its inputs — dspy raises there too, inside
    /// the same `try`, so it is charged to the same budget.
    async fn attempt<T: Module + ?Sized>(
        &self,
        teacher: &mut T,
        example: &Example,
    ) -> Result<Option<Solved>> {
        let inputs = example.inputs()?;
        let withheld = withhold(teacher, example);
        let mut trace = Vec::new();
        let prediction = teacher.forward_traced(inputs.clone(), &mut trace).await;
        // Unconditionally, before the failure is handed on. dspy's restore sits after the call
        // inside the same `try`, so a program that raises leaves the example struck out of the
        // teacher's demos for the whole rest of the compile. Nothing decides that; it leaks.
        restore(teacher, withheld);

        let prediction = prediction?;
        let accepted = match &self.metric {
            None => true,
            Some(metric) => self.accepts(metric.score(example, &prediction).await),
        };
        Ok(accepted.then(|| Solved {
            program: augmented_turn(&inputs, &prediction.example),
            per_predictor: trace
                .into_iter()
                // An unparsed step earns no demo. dspy cannot reach one here — a parse failure
                // raises out of the whole forward and `_bootstrap_one_example` marks the example
                // unsuccessful, so nothing is recorded for that call — and skipping is what its
                // trace holds. Only `bootstrap_trace_data` keeps a failure, and only GEPA calls it.
                .filter_map(|step| {
                    let outputs = step.outputs.answered()?;
                    Some((
                        step.predictor.clone(),
                        augmented_turn(&step.inputs, outputs),
                    ))
                })
                .collect(),
        }))
    }

    /// dspy reads the metric for truth the way Python reads a float — anything but zero passes —
    /// unless a threshold turns it into a comparison. Both halves are load-bearing: without a
    /// threshold a metric that scores 0.5 counts as a success, which a bar at 1.0 would reject.
    fn accepts(&self, score: f64) -> bool {
        match self.metric_threshold {
            Some(threshold) if threshold != 0.0 => score >= threshold,
            _ => score != 0.0,
        }
    }

    /// dspy `_train`: bootstrapped demos first, then labelled ones to fill the budget out.
    fn train<S: Module + ?Sized>(&self, student: &mut S, mut bootstrapped: Bootstrapped) -> usize {
        let mut rng = Rng::seeded(0);
        let mut raw = std::mem::take(&mut bootstrapped.validation);
        let mut widest = 0;
        for predictor in student.named_predictors() {
            let augmented = bootstrapped.earned(&predictor.name, self.max_bootstrapped_demos);
            widest = widest.max(augmented.len());
            // The bootstrapped demos are spent from the labelled budget, not added on top of it.
            let room = self.max_labeled_demos.saturating_sub(augmented.len());
            // dspy rebinds `raw_demos` to the sample it just drew, so the next predictor draws
            // from this predictor's sample rather than from the validation set. It reads like a
            // slip, but it is upstream's, and it shows the moment a program has two predictors.
            raw = rng.sample(&raw, room);
            *predictor.demos = augmented.iter().chain(&raw).cloned().collect();
        }
        widest
    }
}

impl<M> Optimizer for BootstrapFewShot<M>
where
    M: crate::evaluate::Metric,
{
    async fn compile<'a>(
        &'a self,
        student: &'a mut dyn Module,
        teacher: Option<&'a mut dyn Module>,
        trainset: &'a [Example],
        valset: Option<&'a [Example]>,
    ) -> Result<()> {
        if valset.is_some() {
            return Err(anyhow!(
                "BootstrapFewShot keeps the traces a teacher produced and never scores a \
                 candidate, so it has no valset to score on"
            ));
        }
        match teacher {
            Some(teacher) => {
                self.compile_with_teacher(student, teacher, trainset)
                    .await?
            }
            None => BootstrapFewShot::compile(self, student, trainset).await?,
        };
        Ok(())
    }
}

/// What each predictor asks for before a round overrides it, in `named_predictors` order.
fn resting_config<T: Module + ?Sized>(teacher: &mut T) -> Vec<Sampling> {
    teacher
        .named_predictors()
        .iter()
        .map(|predictor| predictor.config.clone())
        .collect()
}

/// Put back what [`resting_config`] read.
///
/// A teacher outlives the compile that borrowed it, so leaving a rollout on it would silently
/// change how it answers everything afterwards — and at `temperature = 1.0`, which is nobody's
/// idea of a default.
fn restore_config<T: Module + ?Sized>(teacher: &mut T, resting: &[Sampling]) {
    for (predictor, was) in teacher.named_predictors().into_iter().zip(resting) {
        *predictor.config = was.clone();
    }
}

/// dspy `Example(augmented=True, **inputs, **outputs)`: a demo the teacher earned, marked as such.
///
/// The marker is read, not decoration. MIPROv2's grounded proposer gathers *only* augmented
/// demos, and a compiled program carries the key to disk — so a program compiled here and opened
/// in Python must have it where dspy's would. It comes first because dspy passes it first and a
/// Python dict keeps insertion order.
///
/// Every earned demo carries it, the untraced program's included. dspy has no unmarked earned
/// demo to match: `dspy.settings.trace` fills from `Predict.__call__`, so a program that answered
/// has a trace, and `name2traces` is the only place `_train` reads demos from. The untraced arm of
/// [`Bootstrapped`](super::earned::Bootstrapped) stands in for that trace, so it stands in for the
/// marker too. Only labelled demos drawn from the trainset go unmarked, on both sides.
fn augmented_turn(inputs: &Example, outputs: &Example) -> Example {
    let mut marked = Example::new([("augmented", serde_json::Value::Bool(true))]);
    for (name, value) in inputs.fields().chain(outputs.fields()) {
        marked.set(name, value.clone());
    }
    let keys: Vec<String> = inputs
        .fields()
        .filter(|(name, _)| inputs.is_input(name))
        .map(|(name, _)| name.to_owned())
        .collect();
    marked.with_inputs(keys)
}

/// Hide the example being solved from the teacher's own demos, and answer with what was hidden.
///
/// dspy filters it out for the length of the call: a teacher shown the answer would copy it back
/// and the demo would prove nothing about whether the program can solve anything.
fn withhold<T: Module + ?Sized>(teacher: &mut T, example: &Example) -> Vec<Vec<Example>> {
    teacher
        .named_predictors()
        .into_iter()
        .map(|predictor| {
            let held = predictor.demos.clone();
            predictor.demos.retain(|demo| demo != example);
            held
        })
        .collect()
}

/// Put back what [`withhold`] took, so the next example is asked of the same teacher.
fn restore<T: Module + ?Sized>(teacher: &mut T, held: Vec<Vec<Example>>) {
    for (predictor, demos) in teacher.named_predictors().into_iter().zip(held) {
        *predictor.demos = demos;
    }
}

/// dspy `_prepare_predictor_mappings`: a teacher that is not the student's twin produces demos
/// the student has no place for, and dspy refuses to compile rather than write them somewhere.
///
/// dspy also asserts the two programs are not the same object. Taking each as its own `&mut`
/// makes that the borrow checker's job, so there is nothing left here to check.
fn same_shape<S, T>(student: &mut S, teacher: &mut T) -> Result<()>
where
    S: Module + ?Sized,
    T: Module + ?Sized,
{
    let taught = teacher.named_predictors();
    let learned = student.named_predictors();
    if learned.len() != taught.len() {
        return Err(anyhow!(
            "student and teacher must have the same number of predictors, got {} and {}",
            learned.len(),
            taught.len()
        ));
    }
    for (student_side, teacher_side) in learned.iter().zip(&taught) {
        if student_side.name != teacher_side.name {
            return Err(anyhow!(
                "student and teacher must have the same program structure, got predictor {:?} against {:?}",
                student_side.name,
                teacher_side.name
            ));
        }
        if student_side.signature != teacher_side.signature {
            return Err(anyhow!(
                "student and teacher must have the same signatures, and differ at predictor {:?}",
                student_side.name
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `max_labeled_demos == 0` skips the labeling entirely, leaving the teacher whatever demos it
    /// arrived with — dspy's own skip. The `>=` mutant labeled anyway, which *wipes* those demos:
    /// LabeledFewShot at zero sets an empty set, and the difference is only visible on a teacher
    /// that arrived with some.
    #[tokio::test]
    async fn zero_labeled_demos_leaves_the_teachers_own_demos_standing() {
        let metric = crate::evaluate::exact_match;
        let mut teacher = Solver::new(Answers::Correctly);
        teacher
            .demos
            .push(example! { question: "kept?", answer: "kept" });
        let optimizer = BootstrapFewShot {
            max_labeled_demos: 0,
            max_bootstrapped_demos: 1,
            ..BootstrapFewShot::new(&metric)
        };
        optimizer
            .bootstrap(&mut teacher, &trainset()[..1])
            .await
            .expect("bootstraps");
        assert_eq!(
            teacher.demos.len(),
            1,
            "the teacher's own demo survived the zero-labeled run"
        );
        assert_eq!(
            teacher.demos[0].get("question"),
            Some(&serde_json::json!("kept?"))
        );
    }
    use crate::evaluate::exact_match;
    use crate::example;
    use crate::optimize::scripted::{Answers, Lopsided, Pair, Solver, TwoHop, answers, trainset};
    use crate::signature::{OutField, Signature};
    use serde_json::json;

    /// A predictor called twice in one example earns two demos and keeps one, and which one is
    /// dspy's seeded coin rather than the last.
    ///
    /// The unit test beside [`collapse`](crate::optimize::earned::collapse) holds the coin itself
    /// to upstream's recorded answers. This holds the *wiring*: that `file` groups a repeated
    /// predictor's demos in trace order and asks the coin, rather than keeping whichever arrived
    /// last. With `hop` differing between the calls the two demos are distinguishable, so taking
    /// the last would answer `"second answer"` every time.
    #[tokio::test]
    async fn a_predictor_called_twice_keeps_the_demo_the_coin_names() {
        let metric = exact_match;
        let mut student = TwoHop::new();
        let optimizer = BootstrapFewShot {
            max_bootstrapped_demos: 4,
            max_labeled_demos: 0,
            ..BootstrapFewShot::new(&metric)
        };
        optimizer
            .compile(&mut student, &trainset()[..1])
            .await
            .expect("compiles");
        assert_eq!(student.demos.len(), 1, "two traces collapse to one demo");
        let kept = student.demos[0]
            .get("hop")
            .and_then(|hop| hop.as_str())
            .expect("the demo records which hop earned it");
        // Which hop the coin lands on is upstream's business; that the *choice* happened is this
        // crate's. A collapse that ignored the coin would take the trace order's last.
        assert_eq!(kept, "first", "the seeded coin picked the earlier hop here");
        assert_eq!(
            student.demos[0].get("augmented"),
            Some(&json!(true)),
            "the kept demo is still marked as earned"
        );
    }

    /// The two examples the capital table solves, in trainset order.
    const SOLVABLE: [&str; 2] = ["Paris", "Berlin"];

    fn bootstrap<M>(metric: M) -> BootstrapFewShot<M> {
        BootstrapFewShot {
            // Off by default so a test says which limit it is exercising.
            max_labeled_demos: 0,
            ..BootstrapFewShot::new(metric)
        }
    }

    #[tokio::test]
    async fn bootstrap_keeps_only_the_attempts_the_metric_accepts() {
        let mut student = Solver::new(Answers::Correctly);
        let kept = bootstrap(exact_match)
            .compile(&mut student, &trainset())
            .await
            .expect("compile succeeds");

        // Four of the six trainset examples are unanswerable, so only two survive scoring.
        assert_eq!(kept, 2);
        assert_eq!(answers(&student.demos), SOLVABLE);
        // A bootstrapped demo carries the inputs asked and the outputs that worked.
        assert_eq!(
            student.demos[0].get("question").unwrap(),
            &json!("capital of France?")
        );
    }

    #[tokio::test]
    async fn a_program_that_never_succeeds_bootstraps_nothing() {
        // The honest outcome: no demos, rather than demos of wrong behaviour.
        let mut student = Solver::new(Answers::Wrongly);
        let kept = bootstrap(exact_match)
            .compile(&mut student, &trainset())
            .await
            .expect("compile succeeds");
        assert_eq!(kept, 0);
        assert!(student.demos.is_empty());
    }

    /// The budget stops the walk, it does not merely trim the result: dspy checks it at the top
    /// of the loop, so the examples past the budget are never asked at all.
    #[tokio::test]
    async fn max_bootstrapped_demos_stops_the_walk_rather_than_truncating_it() {
        let mut student = Solver::new(Answers::Correctly);
        let optimizer = BootstrapFewShot {
            max_bootstrapped_demos: 1,
            ..bootstrap(exact_match)
        };
        let kept = optimizer
            .compile(&mut student, &trainset())
            .await
            .expect("compile succeeds");

        assert_eq!(kept, 1);
        assert_eq!(
            student.calls().len(),
            1,
            "the walk asked past the budget instead of stopping at it"
        );
    }

    /// The bootstrapped demos come first and are spent from the labelled budget, so a budget of
    /// three with two solved examples leaves room for exactly one labelled demo.
    #[tokio::test]
    async fn bootstrapped_demos_lead_and_labeled_ones_fill_the_rest_of_the_budget() {
        let mut student = Solver::new(Answers::Correctly);
        let optimizer = BootstrapFewShot {
            max_labeled_demos: 3,
            ..bootstrap(exact_match)
        };
        optimizer
            .compile(&mut student, &trainset())
            .await
            .expect("compile succeeds");

        let kept = answers(&student.demos);
        assert_eq!(kept.len(), 3);
        assert_eq!(kept[..2], SOLVABLE);
        assert!(
            kept[2].starts_with("riddle"),
            "the tail should be a labelled example the teacher could not solve, got {:?}",
            kept[2]
        );
    }

    /// dspy's `max(0, ...)`: more bootstrapped demos than the labelled budget allows leaves no
    /// room rather than a negative amount of it.
    #[tokio::test]
    async fn a_labeled_budget_smaller_than_the_bootstrapped_demos_adds_nothing() {
        let mut student = Solver::new(Answers::Correctly);
        let optimizer = BootstrapFewShot {
            max_labeled_demos: 1,
            ..bootstrap(exact_match)
        };
        optimizer
            .compile(&mut student, &trainset())
            .await
            .expect("compile succeeds");
        assert_eq!(answers(&student.demos), SOLVABLE);
    }

    /// dspy's `_train` reuses the trainset examples nothing solved as labelled demos, which is
    /// what upstream's `test_validation_set_usage` pins.
    #[tokio::test]
    async fn the_examples_that_were_never_solved_become_labeled_demos() {
        let mut student = Solver::new(Answers::Wrongly);
        let optimizer = BootstrapFewShot {
            max_labeled_demos: 16,
            ..bootstrap(exact_match)
        };
        optimizer
            .compile(&mut student, &trainset())
            .await
            .expect("compile succeeds");
        assert_eq!(student.demos.len(), trainset().len());
    }

    #[tokio::test]
    async fn max_rounds_reattempts_an_example_until_one_round_succeeds() {
        let optimizer = BootstrapFewShot {
            max_rounds: 2,
            max_bootstrapped_demos: 1,
            ..bootstrap(exact_match)
        };
        // Right on the second ask of the same question, and never on the first.
        let mut student = Solver::new(Answers::RightOnRound(2));
        let kept = optimizer
            .compile(&mut student, &trainset())
            .await
            .expect("compile succeeds");
        assert_eq!(kept, 1);

        let mut once = Solver::new(Answers::RightOnRound(2));
        let kept = BootstrapFewShot {
            max_rounds: 1,
            ..optimizer
        }
        .compile(&mut once, &trainset())
        .await
        .expect("compile succeeds");
        assert_eq!(kept, 0, "one round should not get a second ask");
    }

    /// What `b2` was open on. A round that only re-asks is worthless against a deterministic
    /// model: it sends byte-identical bytes and gets the identical answer back. dspy makes each
    /// round after the first `lm.copy(rollout_id=round, temperature=1.0)`, and this is that —
    /// the rollout id to miss the cache, the temperature to make the answer differ once it does.
    #[tokio::test]
    async fn every_round_after_the_first_is_asked_as_a_fresh_rollout() {
        let mut student = Solver::new(Answers::RightOnRound(3));
        BootstrapFewShot {
            max_rounds: 3,
            max_bootstrapped_demos: 1,
            ..bootstrap(exact_match)
        }
        .compile(&mut student, &trainset())
        .await
        .expect("compile succeeds");

        let asks: Vec<Sampling> = student
            .calls()
            .into_iter()
            .take(3)
            .map(|call| call.config)
            .collect();
        assert_eq!(
            asks[0],
            Sampling::default(),
            "the first round is asked however the teacher already asks"
        );
        assert_eq!(asks[1], Sampling::rollout(1));
        assert_eq!(asks[2], Sampling::rollout(2));
        assert_ne!(asks[1], asks[2], "two rounds are two different asks");
    }

    /// A teacher outlives the compile, so a rollout left on it would quietly re-sample every
    /// later call at `temperature = 1.0`.
    #[tokio::test]
    async fn the_teacher_is_left_asking_the_way_it_was_found() {
        let mut student = Solver::new(Answers::Wrongly);
        let resting = Sampling {
            temperature: Some(0.2),
            ..Sampling::default()
        };
        student.set_config(resting.clone());

        BootstrapFewShot {
            max_rounds: 3,
            ..bootstrap(exact_match)
        }
        .compile(&mut student, &trainset())
        .await
        .expect("compile succeeds");

        assert_eq!(
            student.named_predictors()[0].config.clone(),
            resting,
            "the rounds borrowed the teacher's config and gave it back"
        );
    }

    #[tokio::test]
    async fn a_round_that_succeeds_ends_the_rounds_for_that_example() {
        let mut student = Solver::new(Answers::Correctly);
        BootstrapFewShot {
            max_rounds: 5,
            max_bootstrapped_demos: 1,
            ..bootstrap(exact_match)
        }
        .compile(&mut student, &trainset())
        .await
        .expect("compile succeeds");
        assert_eq!(student.calls().len(), 1, "asked again after succeeding");
    }

    #[tokio::test]
    async fn the_error_budget_ends_the_compile_and_hands_back_the_failure() {
        let mut student = Solver::new(Answers::Failing);
        let optimizer = BootstrapFewShot {
            max_errors: 1,
            ..bootstrap(exact_match)
        };
        let error = optimizer
            .compile(&mut student, &trainset())
            .await
            .expect_err("the budget should have been spent");
        assert!(error.to_string().contains("the provider is down"));
        assert_eq!(
            student.calls().len(),
            1,
            "the walk continued past the error budget"
        );
    }

    /// Below the budget a failure is absorbed: one bad example must not discard the evidence
    /// from every other one.
    #[tokio::test]
    async fn failures_under_the_budget_are_absorbed_and_the_walk_carries_on() {
        let mut student = Solver::new(Answers::Failing);
        let optimizer = BootstrapFewShot {
            max_errors: 99,
            ..bootstrap(exact_match)
        };
        let kept = optimizer
            .compile(&mut student, &trainset())
            .await
            .expect("compile survives the failures");
        assert_eq!(kept, 0);
        assert_eq!(student.calls().len(), trainset().len());
    }

    /// An example that never declared its inputs raises inside dspy's own `try`, so it spends
    /// error budget rather than passing silently.
    #[tokio::test]
    async fn an_undeclared_example_is_charged_to_the_error_budget() {
        let mut student = Solver::new(Answers::Correctly);
        let optimizer = BootstrapFewShot {
            max_errors: 1,
            ..bootstrap(exact_match)
        };
        let undeclared = vec![example! { question: "capital of France?", answer: "Paris" }];
        let error = optimizer
            .compile(&mut student, &undeclared)
            .await
            .expect_err("an undeclared example cannot be asked");
        assert!(error.to_string().contains("inputs have not been set"));
    }

    /// Without a threshold dspy reads the metric for truth, so a half score is a success. A port
    /// that compared against 1.0 would silently keep fewer demos than upstream.
    #[tokio::test]
    async fn without_a_threshold_any_non_zero_score_is_a_success() {
        let mut student = Solver::new(Answers::Wrongly);
        let kept = bootstrap(|_: &Example, _: &Prediction| 0.5)
            .compile(&mut student, &trainset())
            .await
            .expect("compile succeeds");
        assert_eq!(
            kept, 4,
            "a 0.5 score should pass, bounded by the demo budget"
        );
    }

    #[tokio::test]
    async fn a_threshold_turns_the_metric_into_a_comparison() {
        let half = |_: &Example, _: &Prediction| 0.5;
        let below = BootstrapFewShot {
            metric_threshold: Some(0.8),
            ..bootstrap(half)
        };
        let mut student = Solver::new(Answers::Correctly);
        assert_eq!(
            below.compile(&mut student, &trainset()).await.unwrap(),
            0,
            "0.5 is under a bar of 0.8"
        );

        let above = BootstrapFewShot {
            metric_threshold: Some(0.4),
            ..bootstrap(half)
        };
        let mut student = Solver::new(Answers::Correctly);
        assert_eq!(above.compile(&mut student, &trainset()).await.unwrap(), 4);
    }

    /// Python reads `0.0` as false, so dspy's `if self.metric_threshold` skips a zero threshold
    /// and falls back to reading the metric for truth. A negative score passes there and would
    /// fail a real bar at zero, which is what separates the two readings.
    #[tokio::test]
    async fn a_threshold_of_zero_means_no_threshold_at_all() {
        let mut student = Solver::new(Answers::Correctly);
        let optimizer = BootstrapFewShot {
            metric_threshold: Some(0.0),
            ..bootstrap(|_: &Example, _: &Prediction| -1.0)
        };
        assert_eq!(
            optimizer.compile(&mut student, &trainset()).await.unwrap(),
            4
        );
    }

    #[tokio::test]
    async fn a_zero_score_is_a_failure_whichever_reading_applies() {
        let mut student = Solver::new(Answers::Correctly);
        let kept = bootstrap(|_: &Example, _: &Prediction| 0.0)
            .compile(&mut student, &trainset())
            .await
            .expect("compile succeeds");
        assert_eq!(kept, 0);
    }

    /// dspy's `metric=None`: nothing scores the attempts, so every one that ran is kept.
    #[tokio::test]
    async fn without_a_metric_every_attempt_that_ran_is_kept() {
        let mut student = Solver::new(Answers::Wrongly);
        let kept = BootstrapFewShot {
            max_labeled_demos: 0,
            ..BootstrapFewShot::without_metric()
        }
        .compile(&mut student, &trainset())
        .await
        .expect("compile succeeds");
        assert_eq!(kept, 4);
        assert_eq!(answers(&student.demos), ["no idea"; 4]);
    }

    #[tokio::test]
    async fn the_teacher_is_shown_labeled_demos_before_it_is_asked_to_solve_anything() {
        let mut student = Solver::new(Answers::Correctly);
        BootstrapFewShot {
            max_labeled_demos: 4,
            ..bootstrap(exact_match)
        }
        .compile(&mut student, &trainset())
        .await
        .expect("compile succeeds");

        let first = student.calls().first().cloned().expect("the teacher ran");
        assert_eq!(
            first.demos.len(),
            3,
            "the teacher should hold its four labelled demos, less the one being solved"
        );
    }

    /// dspy hides the example under test from the teacher's demos for the length of the call.
    /// Handing a teacher the answer would make every attempt succeed and prove nothing.
    #[tokio::test]
    async fn the_example_being_solved_is_withheld_from_the_teachers_own_demos() {
        let mut student = Solver::new(Answers::Correctly);
        BootstrapFewShot {
            max_labeled_demos: 16,
            ..bootstrap(exact_match)
        }
        .compile(&mut student, &trainset())
        .await
        .expect("compile succeeds");

        for call in student.calls() {
            let held: Vec<String> = call
                .demos
                .iter()
                .filter_map(|demo| demo.get("question")?.as_str().map(str::to_owned))
                .collect();
            assert!(
                !held.contains(&call.question),
                "the teacher was shown the answer to {:?}",
                call.question
            );
            assert_eq!(held.len(), trainset().len() - 1, "more than one was hidden");
        }
    }

    /// The withheld demos go back afterwards, so every example after the first is asked of the
    /// same teacher rather than one that has been quietly emptied out.
    #[tokio::test]
    async fn the_withheld_demos_are_put_back_after_the_call() {
        let mut teacher = Solver::new(Answers::Correctly);
        let mut student = Solver::new(Answers::Correctly);
        BootstrapFewShot {
            max_labeled_demos: 16,
            ..bootstrap(exact_match)
        }
        .compile_with_teacher(&mut student, &mut teacher, &trainset())
        .await
        .expect("compile succeeds");
        assert_eq!(teacher.demos.len(), trainset().len());
    }

    /// The teacher runs and the student keeps the result: the student is never asked anything,
    /// and the teacher never takes the compiled demos.
    #[tokio::test]
    async fn the_teacher_runs_and_the_student_keeps_the_result() {
        let mut teacher = Solver::new(Answers::Correctly);
        let mut student = Solver::new(Answers::Wrongly);
        let kept = BootstrapFewShot {
            max_labeled_demos: 0,
            ..bootstrap(exact_match)
        }
        .compile_with_teacher(&mut student, &mut teacher, &trainset())
        .await
        .expect("compile succeeds");

        assert_eq!(kept, 2);
        assert!(
            student.calls().is_empty(),
            "the student answered instead of the teacher"
        );
        assert_eq!(teacher.calls().len(), trainset().len());
        assert_eq!(answers(&student.demos), SOLVABLE);
        assert!(teacher.demos.is_empty(), "the teacher took the result");
    }

    #[tokio::test]
    async fn a_teacher_of_a_different_shape_is_refused() {
        let mut student = Solver::new(Answers::Correctly);
        let mut teacher = Pair::new();
        let error = bootstrap(exact_match)
            .compile_with_teacher(&mut student, &mut teacher, &trainset())
            .await
            .expect_err("one predictor cannot learn from two");
        assert!(error.to_string().contains("same number of predictors"));
    }

    #[tokio::test]
    async fn a_teacher_with_a_different_signature_is_refused() {
        let mut student = Solver::new(Answers::Correctly);
        let mut teacher = Solver::new(Answers::Correctly);
        for predictor in teacher.named_predictors() {
            *predictor.signature = Signature::single_input(
                "Answer, but differently.",
                vec![OutField {
                    name: "answer".into(),
                    ..Default::default()
                }],
            );
        }
        let error = bootstrap(exact_match)
            .compile_with_teacher(&mut student, &mut teacher, &trainset())
            .await
            .expect_err("the signatures differ");
        assert!(error.to_string().contains("same signatures"));
    }

    /// dspy rebinds `raw_demos` to the sample it just drew, so each predictor draws from the
    /// previous one's sample and the labelled demos can only shrink along the walk.
    #[tokio::test]
    async fn the_labeled_pool_shrinks_as_each_predictor_is_filled() {
        let mut student = Pair::new();
        BootstrapFewShot {
            max_labeled_demos: 3,
            ..bootstrap(exact_match)
        }
        .compile(&mut student, &trainset())
        .await
        .expect("compile succeeds");

        // Two bootstrapped demos each, then one labelled demo drawn from the four unsolved
        // examples, then one drawn from that single-example pool.
        assert_eq!(student.first_demos.len(), 3);
        assert_eq!(student.second_demos.len(), 3);
        assert_eq!(
            answers(&student.first_demos[2..]),
            answers(&student.second_demos[2..]),
            "the second predictor's only choice is the first predictor's sample"
        );
    }

    /// Each predictor is taught by the calls it made, not by the program's result.
    ///
    /// `Pair` drafts in its first half and answers from that draft in its second, so the two
    /// earn demos that the other could not have: a misattribution shows up as the wrong fields
    /// rather than as a different ordering.
    #[tokio::test]
    async fn each_predictor_is_taught_by_its_own_calls() {
        let mut student = Pair::new();
        bootstrap(exact_match)
            .compile(&mut student, &trainset())
            .await
            .expect("compile succeeds");

        assert_eq!(student.first_demos.len(), 2);
        assert_eq!(student.second_demos.len(), 2);
        assert!(
            student
                .first_demos
                .iter()
                .all(|demo| demo.get("question").is_some() && demo.get("answer").is_none()),
            "the drafting half is taught by what it was asked and what it drafted"
        );
        assert!(
            student
                .second_demos
                .iter()
                .all(|demo| demo.get("draft").is_some() && demo.get("question").is_none()),
            "the answering half is taught by the draft it read, never by the question"
        );
        assert_eq!(answers(&student.second_demos), SOLVABLE);
    }

    /// A predictor that never ran is taught by nothing, which is dspy starting every name at an
    /// empty list rather than falling back to what the program as a whole managed.
    #[tokio::test]
    async fn a_predictor_that_never_ran_is_taught_by_nothing() {
        let mut student = Lopsided::new();
        bootstrap(exact_match)
            .compile(&mut student, &trainset())
            .await
            .expect("compile succeeds");

        assert_eq!(answers(&student.ran_demos), SOLVABLE);
        assert!(
            student.idle_demos.is_empty(),
            "an idle predictor should not inherit its sibling's demos"
        );
    }

    /// The walk stops once enough *examples* are solved, not once enough demos are collected.
    ///
    /// With two predictors one example earns two demos, so counting demos would carry the walk
    /// past the budget. `train` caps what each predictor keeps either way, which hides the
    /// difference in the demos themselves; where it shows is the validation set, because an
    /// example the walk never reached is one more the labelled tail can draw from.
    #[tokio::test]
    async fn the_budget_counts_solved_examples_rather_than_demos() {
        let mut student = Pair::new();
        BootstrapFewShot {
            max_bootstrapped_demos: 1,
            // Room for the whole validation set, so its size is read off the demos directly
            // rather than through which members a partial draw happened to take.
            max_labeled_demos: 6,
            ..bootstrap(exact_match)
        }
        .compile(&mut student, &trainset())
        .await
        .expect("compile succeeds");

        // France alone was solved, leaving Germany and the four riddles to be drawn whole.
        assert_eq!(student.first_demos.len(), 6);
        assert!(
            answers(&student.first_demos).contains(&"Berlin".to_owned()),
            "the walk stopped before Germany, so it stays available to the labelled draw"
        );
    }

    #[tokio::test]
    async fn an_empty_trainset_compiles_to_nothing() {
        // dspy raises a NameError here, reading a loop variable the empty loop never bound.
        // Nothing to port: a compile with nothing to learn from writes no demos.
        let mut student = Solver::new(Answers::Correctly);
        let kept = bootstrap(exact_match)
            .compile(&mut student, &[])
            .await
            .expect("compile succeeds");
        assert_eq!(kept, 0);
        assert!(student.demos.is_empty());
    }
}
