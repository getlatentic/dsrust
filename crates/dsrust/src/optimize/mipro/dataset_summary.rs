//! dspy `propose/dataset_summary_generator.py`: describe the trainset to the proposer.
//!
//! `data_aware_proposer=True` shows the instruction proposer a summary of the data the program will
//! see, written by the prompt model reading the trainset a batch at a time. Three signatures and a
//! loop: observe the first batch, add to those observations batch by batch, then summarise the lot.
//!
//! **The samples are shown as Python source, not as JSON.** The `examples` field carries
//! `repr(trainset[a:b])` — a list of `Example({...}) (input_keys={...})` — put through
//! `order_input_keys_in_string`, which sorts the key set with a regex. That sort is load-bearing and
//! it is upstream's: `input_keys` is a Python `set`, whose iteration order CPython randomises per
//! process, so without it the prompt would differ between two runs of dspy itself.
//!
//! Two bounds are upstream's own and look like bugs from outside: at most ten batches are ever read
//! (`max_calls = 10`, checked *before* the call, so nine actually run), and five replies beginning
//! `COMPLETE` end the loop. Both are reproduced.

use anyhow::Result;
use serde_json::Value;

use crate::example::Example;
use crate::lm::{DynChatModel, Sampling};
use crate::module::Module;
use crate::predict::Predict;
use crate::signature::Signature;

use super::signatures::{input, output};

/// dspy's `max_calls = 10`, tested at the top of the loop body — so the tenth batch breaks out
/// before it is read and nine are described.
const MAX_CALLS: usize = 10;

/// How many `COMPLETE` replies end the loop early.
const MAX_SKIPS: usize = 5;

/// dspy `DatasetDescriptor`: observe trends in a first batch of examples.
pub(crate) fn dataset_descriptor() -> Signature {
    Signature {
        instructions: "Given several examples from a dataset please write observations about trends that hold for most or all of the samples. \
            Some areas you may consider in your observations: topics, content, syntax, conciseness, etc. \
            It will be useful to make an educated guess as to the nature of the task this dataset will enable. Don't be afraid to be creative".into(),
        inputs: vec![input("examples", "Sample data points from the dataset")],
        outputs: vec![output(
            "observations",
            "Somethings that holds true for most or all of the data you observed",
        )],
    }
}

/// dspy `DatasetDescriptorWithPriorObservations`: the same, given what has been observed already.
pub(crate) fn dataset_descriptor_with_prior_observations() -> Signature {
    Signature {
        instructions: "Given several examples from a dataset please write observations about trends that hold for most or all of the samples. \
            I will also provide you with a few observations I have already made.  Please add your own observations or if you feel the observations are comprehensive say 'COMPLETE' \
            Some areas you may consider in your observations: topics, content, syntax, conciceness, etc. \
            It will be useful to make an educated guess as to the nature of the task this dataset will enable. Don't be afraid to be creative".into(),
        inputs: vec![
            input("examples", "Sample data points from the dataset"),
            input(
                "prior_observations",
                "Some prior observations I made about the data",
            ),
        ],
        outputs: vec![output(
            "observations",
            "Somethings that holds true for most or all of the data you observed or COMPLETE if you have nothing to add",
        )],
    }
}

/// dspy `ObservationSummarizer`: the observations, cut to two or three sentences.
pub(crate) fn observation_summarizer() -> Signature {
    Signature {
        instructions: "Given a series of observations I have made about my dataset, please summarize them into a brief 2-3 sentence summary which highlights only the most important details.".into(),
        inputs: vec![input(
            "observations",
            "Observations I have made about my dataset",
        )],
        outputs: vec![output(
            "summary",
            "Two to Three sentence summary of only the most significant highlights of my observations",
        )],
    }
}

/// dspy `order_input_keys_in_string(repr(trainset[a:b]))`: the slice as Python source, with each
/// example's `input_keys` set sorted.
///
/// Sorting is upstream's, and it is what makes this reproducible at all: `input_keys` is a Python
/// `set`, so its iteration order is randomised per process and two runs of dspy would otherwise
/// disagree with each other, never mind with this.
pub(crate) fn examples_repr(examples: &[Example]) -> String {
    let rendered: Vec<String> = examples.iter().map(one_example).collect();
    format!("[{}]", rendered.join(", "))
}

fn one_example(example: &Example) -> String {
    let fields: Vec<String> = example
        .fields()
        .map(|(name, value)| {
            format!(
                "{}: {}",
                crate::python::quoted(name),
                crate::python::repr(value)
            )
        })
        .collect();
    let mut keys: Vec<&str> = example
        .fields()
        .map(|(name, _)| name)
        .filter(|name| example.is_input(name))
        .collect();
    keys.sort_unstable();
    let keys: Vec<String> = keys.iter().map(|key| crate::python::quoted(key)).collect();
    format!(
        "Example({{{}}}) (input_keys={{{}}})",
        fields.join(", "),
        keys.join(", ")
    )
}

/// dspy `create_dataset_summary`: read the trainset a batch at a time and summarise what was seen.
///
/// Every call is `Predict(..., n=1, temperature=1.0)` on the prompt model, as upstream's are. An
/// error anywhere in the observation loop is swallowed and the observations gathered so far are
/// summarised — upstream's `except Exception`, which is why a flaky batch does not lose the run.
pub(crate) async fn create_dataset_summary(
    trainset: &[Example],
    view_data_batch_size: usize,
    prompt_model: &std::sync::Arc<dyn DynChatModel>,
) -> Result<String> {
    let hot = Sampling {
        temperature: Some(1.0),
        ..Sampling::default()
    };
    let describe = |signature: Signature| {
        Predict::from_signature(signature)
            .set_lm(prompt_model.clone())
            .config(hot.clone())
    };

    let upper = view_data_batch_size.min(trainset.len());
    let first = describe(dataset_descriptor())
        .forward(Example::new([(
            "examples",
            Value::String(examples_repr(&trainset[..upper])),
        )]))
        .await?;
    let mut observations = text(&first, "observations");

    let mut skips = 0;
    // `chunks` hands over the same `[start..start+batch]` windows the cursor loop cut, clamped at
    // the end the same way, and `take` is upstream's `calls >= MAX_CALLS` bound: the increment ran
    // before the check, so batch number MAX_CALLS was never asked. The iterator owns the
    // progress — every bounds-and-step mutant of the old walk survived, because nothing ran it.
    let batches = trainset[view_data_batch_size.min(trainset.len())..]
        .chunks(view_data_batch_size.max(1)) // .max(1): upstream at zero asks nine empty batches and gives up — a bug declined, not reproduced
        .take(MAX_CALLS - 1);
    for batch in batches {
        let asked = Example::new([
            ("examples", Value::String(examples_repr(batch))),
            ("prior_observations", Value::String(observations.clone())),
        ]);
        // Upstream wraps the whole loop in `except Exception` and summarises what it already has,
        // so a batch that fails ends the reading rather than the run.
        let Ok(answered) = describe(dataset_descriptor_with_prior_observations())
            .forward(asked)
            .await
        else {
            break;
        };
        let added = text(&answered, "observations");
        if added.len() >= 8 && added[..8].eq_ignore_ascii_case("COMPLETE") {
            skips += 1;
            if skips >= MAX_SKIPS {
                break;
            }
            continue;
        }
        observations.push_str(&added);
    }

    let summary = describe(observation_summarizer())
        .forward(Example::new([(
            "observations",
            Value::String(observations),
        )]))
        .await?;
    Ok(super::proposer::strip_prefix(&text(&summary, "summary")))
}

fn text(prediction: &crate::example::Prediction, field: &str) -> String {
    prediction
        .get(field)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every summarising ask runs hot — dspy's `temperature=1.0` — and the walk stops at
    /// `MAX_CALLS`: with fourteen rows at a batch size of one, only nine loop batches are asked
    /// (upstream increments before it checks, so batch ten was never asked), plus the opener and
    /// the summariser. The temperature-field mutant and both arithmetic mutants of the bound need
    /// a corpus this long to differ at all — the three-row tests above cannot see them.
    #[tokio::test]
    async fn the_walk_runs_hot_and_stops_at_max_calls() {
        let rows: Vec<Example> = (0..14)
            .map(|at| Example::new([("question", Value::String(format!("row {at}")))]))
            .collect();
        let script: Vec<Example> = (0..10)
            .map(|at| Example::new([("observations", Value::String(format!("saw {at}. ")))]))
            .chain([Example::new([(
                "summary",
                Value::String("Rows.".to_owned()),
            )])])
            .collect();
        let lm = std::sync::Arc::new(crate::DummyLM::new(script));

        create_dataset_summary(&rows, 1, &(lm.clone() as _))
            .await
            .expect("the script answers");
        let asked = lm.asked();
        assert_eq!(
            asked.len(),
            1 + (MAX_CALLS - 1) + 1,
            "the opener, nine loop batches, and the summariser"
        );
        for ask in &asked {
            assert_eq!(ask.config.temperature, Some(1.0), "dspy summarises hot");
        }
        let ninth = asked[MAX_CALLS - 1].last_message().to_owned();
        assert!(
            ninth.contains("row 9"),
            "the last asked batch is row 9: {ninth}"
        );
        assert!(
            !asked
                .iter()
                .any(|ask| ask.last_message().contains("row 10")),
            "row 10 sits past the call budget and is never asked"
        );
    }

    /// Five COMPLETE answers end the reading — dspy's MAX_SKIPS — and the batches past the fifth
    /// are never asked. The skip counter's `*=` mutant never reaches five and reads on.
    #[tokio::test]
    async fn five_complete_answers_end_the_reading() {
        let rows: Vec<Example> = (0..9)
            .map(|at| Example::new([("question", Value::String(format!("row {at}")))]))
            .collect();
        let script: Vec<Example> = std::iter::once(Example::new([(
            "observations",
            Value::String("opening. ".to_owned()),
        )]))
        .chain(
            (0..MAX_SKIPS)
                .map(|_| Example::new([("observations", Value::String("COMPLETE".to_owned()))])),
        )
        .chain([Example::new([(
            "summary",
            Value::String("Rows.".to_owned()),
        )])])
        .collect();
        let lm = std::sync::Arc::new(crate::DummyLM::new(script));

        create_dataset_summary(&rows, 1, &(lm.clone() as _))
            .await
            .expect("the script answers");
        assert_eq!(
            lm.asked().len(),
            1 + MAX_SKIPS + 1,
            "the opener, five skipped batches, and the summariser — nothing after the fifth"
        );
    }

    /// The batching walk, driven by a scripted LM: three rows at a batch size of one is a first
    /// call plus two loop calls plus the summariser, each batch carrying only its own row and the
    /// observations accumulated so far.
    ///
    /// Twenty mutants lived in this function — every bound and every `+=` of the walk, plus the
    /// whole body replaceable by `Ok("")` — because nothing ever ran it. It needs an LM, and
    /// "needs an LM" had been standing in for "cannot be tested".
    #[tokio::test]
    async fn the_walk_batches_the_trainset_and_folds_each_answer_in() {
        let rows: Vec<Example> = ["alpha", "beta", "gamma"]
            .iter()
            .map(|word| Example::new([("question", Value::String((*word).to_owned()))]))
            .collect();
        let lm = std::sync::Arc::new(crate::DummyLM::new([
            Example::new([("observations", Value::String("saw alpha. ".to_owned()))]),
            Example::new([("observations", Value::String("saw beta. ".to_owned()))]),
            Example::new([("observations", Value::String("saw gamma. ".to_owned()))]),
            Example::new([("summary", Value::String("Three greek words.".to_owned()))]),
        ]));

        let summary = create_dataset_summary(&rows, 1, &(lm.clone() as _))
            .await
            .expect("the script answers");
        assert_eq!(summary, "Three greek words.");

        let asked: Vec<String> = lm
            .asked()
            .iter()
            .map(|a| a.last_message().to_owned())
            .collect();
        assert_eq!(asked.len(), 4, "one per batch, plus the summariser");
        // Each batch carries its own row and no other: the bounds mutants all widened or narrowed
        // this window, and every one of them survived.
        assert!(asked[0].contains("alpha"), "{}", asked[0]);
        assert!(!asked[0].contains("beta"), "the first batch is one row");
        assert!(asked[1].contains("beta") && !asked[1].contains("gamma"));
        assert!(asked[2].contains("gamma"));
        // And the observations accumulate, which is what `prior_observations` is for.
        assert!(asked[2].contains("saw alpha."), "{}", asked[2]);
        assert!(
            asked[3].contains("saw gamma."),
            "the summariser sees them all"
        );
    }

    /// A batch answering `COMPLETE` contributes nothing and the walk goes on — upstream's own
    /// skip. The `>= 8` length guard and the case-insensitive compare both had mutants standing.
    #[tokio::test]
    async fn a_complete_answer_is_skipped_rather_than_folded_in() {
        let rows: Vec<Example> = ["alpha", "beta"]
            .iter()
            .map(|word| Example::new([("question", Value::String((*word).to_owned()))]))
            .collect();
        let lm = std::sync::Arc::new(crate::DummyLM::new([
            Example::new([("observations", Value::String("saw alpha. ".to_owned()))]),
            Example::new([(
                "observations",
                Value::String("complete, nothing new".to_owned()),
            )]),
            Example::new([("summary", Value::String("One word.".to_owned()))]),
        ]));

        create_dataset_summary(&rows, 1, &(lm.clone() as _))
            .await
            .expect("the script answers");
        let asked: Vec<String> = lm
            .asked()
            .iter()
            .map(|a| a.last_message().to_owned())
            .collect();
        let summariser = asked.last().expect("a summariser call");
        assert!(summariser.contains("saw alpha."), "{summariser}");
        assert!(
            !summariser.contains("nothing new"),
            "a COMPLETE batch is skipped, not folded in: {summariser}"
        );
    }
    use serde_json::json;

    fn golden() -> serde_json::Value {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/conformance/optimize/dataset_summary.json");
        let text = std::fs::read_to_string(&path).expect("the golden is committed");
        serde_json::from_str(&text).expect("the golden parses")
    }

    fn rows(golden: &serde_json::Value) -> Vec<Example> {
        golden["rows"]
            .as_array()
            .expect("rows")
            .iter()
            .map(|row| {
                let fields = row["fields"].as_object().expect("fields");
                let keys: Vec<&str> = row["input_keys"]
                    .as_array()
                    .expect("input keys")
                    .iter()
                    .map(|key| key.as_str().expect("a key"))
                    .collect();
                Example::new(
                    fields
                        .iter()
                        .map(|(name, value)| (name.as_str(), value.clone())),
                )
                .with_inputs(keys)
            })
            .collect()
    }

    /// Every slice, against `order_input_keys_in_string(repr(...))` from the pinned dspy.
    ///
    /// The corpus is chosen for what it does to `repr`: an apostrophe alone, both quotes, a
    /// backslash, a float, `None`, `True`, a list, and input keys that are not alphabetical in
    /// field order so the sort has something to do.
    #[test]
    fn a_trainset_slice_renders_as_python_prints_it() {
        let golden = golden();
        let rows = rows(&golden);
        let slices = golden["slices"].as_array().expect("slices");
        assert!(!slices.is_empty(), "the golden records no slices");
        for slice in slices {
            let start = slice["start"].as_u64().expect("start") as usize;
            let stop = slice["stop"].as_u64().expect("stop") as usize;
            assert_eq!(
                examples_repr(&rows[start..stop]),
                slice["repr"].as_str().expect("a repr"),
                "slice {start}..{stop}"
            );
        }
    }

    /// The three signatures, as rendered system prompts — the same standard the proposer's own are
    /// held to.
    #[test]
    fn the_three_signatures_render_as_dspys_do() {
        let golden = golden();
        let recorded = &golden["signatures"];
        for (name, signature) in [
            ("dataset_descriptor", dataset_descriptor()),
            (
                "dataset_descriptor_with_prior_observations",
                dataset_descriptor_with_prior_observations(),
            ),
            ("observation_summarizer", observation_summarizer()),
        ] {
            let inputs: Vec<crate::adapter::Input<'_>> = signature
                .inputs
                .iter()
                .map(|field| crate::adapter::Input::new(field.name.as_str(), json!("")))
                .collect();
            let rendered = crate::adapter::Adapter::format(
                &crate::adapter::ChatAdapter::default(),
                &signature,
                &[],
                &inputs,
            )
            .expect("renders");
            let system = rendered[0].text().expect("a system message");
            assert_eq!(
                system,
                recorded[name].as_str().expect("a rendered signature"),
                "{name}"
            );
        }
    }
}
