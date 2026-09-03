//! dspy's KNN family, held to `tests/conformance/knn/knn.json`, recorded by
//! `scripts/generate_knn_fixture.py`: the dummy vectorizer's float32 rows, `KNN`'s selections,
//! the `Embeddings` retriever's answers, and the bytes numpy writes for a saved index.

use base64::Engine;
use dsrust::lm::dummy_vectorizer::DummyVectorizer;
use dsrust::retrievers::npy;
use serde_json::Value;

fn fixture() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/conformance/knn/knn.json"
    );
    serde_json::from_str(&std::fs::read_to_string(path).expect("committed")).expect("parses")
}

fn rows(value: &Value) -> Vec<Vec<f32>> {
    value
        .as_array()
        .expect("rows")
        .iter()
        .map(|row| {
            row.as_array()
                .expect("row")
                .iter()
                .map(|v| v.as_f64().expect("number") as f32)
                .collect()
        })
        .collect()
}

fn bytes(value: &Value) -> Vec<u8> {
    base64::engine::general_purpose::STANDARD
        .decode(value.as_str().expect("base64"))
        .expect("decodes")
}

#[test]
fn the_vectorizer_draws_pythons_coefficients_and_rounds_as_numpy_does() {
    let fixture = fixture();
    let recorded = &fixture["vectorizer"];
    let vectorizer = DummyVectorizer::new(
        recorded["max_length"].as_u64().unwrap() as usize,
        recorded["n_gram"].as_u64().unwrap() as usize,
    );
    let coeffs: Vec<u64> = recorded["coeffs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c.as_u64().unwrap())
        .collect();
    assert_eq!(
        vectorizer.coeffs(),
        coeffs.as_slice(),
        "random.seed(123) then randrange(1, P), twice"
    );
    let texts: Vec<&str> = recorded["texts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t.as_str().unwrap())
        .collect();
    let ours = vectorizer.vectorize(&texts);
    let theirs = rows(&recorded["vectors"]);
    for (at, (mine, its)) in ours.iter().zip(&theirs).enumerate() {
        assert_eq!(
            mine, its,
            "row {at} ({:?}) differs from numpy's float32 row",
            texts[at]
        );
    }
}

#[test]
fn a_saved_matrix_is_byte_for_byte_what_numpy_writes() {
    let fixture = fixture();
    let saved = &fixture["embeddings"]["saved"];
    let matrix = rows(&saved["corpus_embeddings"]);
    assert_eq!(
        npy::encode_f32(&matrix),
        bytes(&saved["corpus_embeddings_npy"])
    );
    let wide = rows(&fixture["npy"]["wide_values"]);
    assert_eq!(
        npy::encode_f32(&wide),
        bytes(&fixture["npy"]["wide"]),
        "a 2x50 header pads differently"
    );
}

#[test]
fn numpys_bytes_read_back_as_the_matrix() {
    let fixture = fixture();
    let saved = &fixture["embeddings"]["saved"];
    assert_eq!(
        npy::decode_f32(&bytes(&saved["corpus_embeddings_npy"])).expect("reads"),
        rows(&saved["corpus_embeddings"])
    );
    assert_eq!(
        npy::decode_f32(&bytes(&saved["unnormalized_npy"]))
            .expect("reads")
            .len(),
        3
    );
    assert_eq!(
        npy::decode_f32(&bytes(&fixture["npy"]["wide"])).expect("reads"),
        rows(&fixture["npy"]["wide_values"])
    );
    assert!(npy::decode_f32(b"not numpy").is_err());
}

mod selection {
    use std::sync::Arc;

    use dsrust::lm::dummy_vectorizer::DummyVectorizer;
    use dsrust::lm::embedding::Embedder;
    use dsrust::predict::knn::Knn;
    use dsrust::retrievers::{Embeddings, EmbeddingsWithScores};
    use dsrust::{Example, example};
    use serde_json::{Value, json};

    use super::{fixture, rows};

    fn dummy_vectorizer() -> Arc<Embedder> {
        let vectorizer = DummyVectorizer::default();
        Arc::new(Embedder::callable(move |texts, _| {
            Ok(vectorizer.vectorize(texts))
        }))
    }

    /// Upstream's `dummy_embedder` in `tests/retrievers/test_embeddings.py`.
    fn dummy_embedder() -> Arc<Embedder> {
        Arc::new(Embedder::callable(|texts, _| {
            Ok(texts
                .iter()
                .map(|text| match (text.contains("cat"), text.contains("dog")) {
                    (true, _) => vec![1.0, 0.0, 0.0],
                    (_, true) => vec![0.0, 1.0, 0.0],
                    _ => vec![0.0, 0.0, 1.0],
                })
                .collect())
        }))
    }

    fn trainset(recorded: &Value) -> Vec<Example> {
        recorded
            .as_array()
            .expect("trainset")
            .iter()
            .map(|e| {
                example! { question: e["question"].clone(), answer: e["answer"].clone() }
                    .with_inputs(["question"])
            })
            .collect()
    }

    #[tokio::test]
    async fn knn_selects_what_dspy_selects() {
        let fixture = fixture();
        let recorded = &fixture["knn"];
        let knn = Knn::build(
            recorded["k"].as_u64().unwrap() as usize,
            trainset(&recorded["trainset"]),
            dummy_vectorizer(),
        )
        .await
        .expect("embeds");
        assert_eq!(
            knn.trainset_vectors(),
            rows(&recorded["trainset_vectors"]).as_slice(),
            "the trainset rows, `question: ...` rendered and embedded"
        );
        for (query, expected) in recorded["selections"].as_object().unwrap() {
            let nearest = knn
                .call(&example! { question: query.as_str() })
                .await
                .expect("selects");
            let indices: Vec<usize> = nearest
                .iter()
                .map(|found| {
                    knn.trainset()
                        .iter()
                        .position(|e| e == found)
                        .expect("from the trainset")
                })
                .collect();
            assert_eq!(
                json!(indices),
                *expected,
                "{query}: numpy's argsort, last k reversed"
            );
        }
    }

    #[tokio::test]
    async fn the_retriever_answers_what_dspy_answers() {
        let fixture = fixture();
        let recorded = &fixture["embeddings"];
        let corpus: Vec<String> = recorded["corpus"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p.as_str().unwrap().to_owned())
            .collect();
        let normalized = EmbeddingsWithScores::build(corpus.clone(), dummy_embedder(), 2, true)
            .await
            .expect("builds");
        let unnormalized = EmbeddingsWithScores::build(corpus.clone(), dummy_embedder(), 3, false)
            .await
            .expect("builds");
        for (query, expected) in recorded["searches"].as_object().unwrap() {
            for (label, retriever) in [("normalized", &normalized), ("unnormalized", &unnormalized)]
            {
                let found = retriever.0.search(query).await.expect("searches");
                let theirs = &expected[label];
                assert_eq!(
                    json!(found.passages),
                    theirs["passages"],
                    "{query} ({label}) passages"
                );
                assert_eq!(
                    json!(found.indices),
                    theirs["indices"],
                    "{query} ({label}) indices"
                );
                assert_eq!(
                    json!(found.scores),
                    theirs["scores"],
                    "{query} ({label}) scores"
                );
            }
        }
        let prediction = normalized
            .forward("A dog is barking.")
            .await
            .expect("answers");
        assert_eq!(prediction.get("indices").unwrap(), &json!([1, 0]));
        assert!(prediction.get("scores").is_some(), "with scores");
        let plain = Embeddings::build(corpus, dummy_embedder(), 1, true)
            .await
            .expect("builds");
        let answered = plain
            .forward("I saw a dog running.")
            .await
            .expect("answers");
        assert_eq!(
            answered.get("passages").unwrap(),
            &json!(["The dog barked at the mailman."])
        );
        assert!(answered.get("scores").is_none(), "without scores");
    }

    #[tokio::test]
    async fn a_saved_index_round_trips_and_loads_dspys_files() {
        let fixture = fixture();
        let recorded = &fixture["embeddings"];
        let corpus: Vec<String> = recorded["corpus"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p.as_str().unwrap().to_owned())
            .collect();
        let original = EmbeddingsWithScores::build(corpus.clone(), dummy_embedder(), 2, false)
            .await
            .expect("builds");
        let folder = tempfile::tempdir().expect("a folder");
        original.0.save(folder.path().join("index")).expect("saves");
        let config: Value = serde_json::from_str(
            &std::fs::read_to_string(folder.path().join("index/config.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(
            config,
            json!({ "k": 2, "normalize": false, "corpus": corpus, "has_faiss_index": false })
        );
        let loaded =
            EmbeddingsWithScores::from_saved(folder.path().join("index"), dummy_embedder())
                .expect("loads");
        assert_eq!(loaded.0.k(), 2);
        assert!(!loaded.0.normalize());
        assert_eq!(loaded.0.corpus(), original.0.corpus());
        let theirs = original.0.search("cat sitting").await.unwrap();
        assert_eq!(loaded.0.search("cat sitting").await.unwrap(), theirs);
        // dspy's own files: the config it writes and numpy's bytes, read back as an index.
        let dspy_dir = folder.path().join("from_dspy");
        std::fs::create_dir_all(&dspy_dir).unwrap();
        std::fs::write(
            dspy_dir.join("config.json"),
            serde_json::to_string_pretty(&recorded["saved"]["config"]).unwrap(),
        )
        .unwrap();
        std::fs::write(
            dspy_dir.join("corpus_embeddings.npy"),
            super::bytes(&recorded["saved"]["corpus_embeddings_npy"]),
        )
        .unwrap();
        let from_dspy =
            Embeddings::from_saved(&dspy_dir, dummy_embedder()).expect("loads dspy's files");
        assert_eq!(
            from_dspy.corpus_embeddings(),
            rows(&recorded["saved"]["corpus_embeddings"]).as_slice()
        );
        assert!(Embeddings::from_saved("/nonexistent/path", dummy_embedder()).is_err());
    }
}

mod numpy_facts {
    use dsrust::numpy::{argsort_f32, dot_f32, l2_norm_f32, mean_f32};

    use super::{fixture, rows};

    fn floats(value: &serde_json::Value) -> Vec<f32> {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect()
    }

    /// `np.mean` and `np.linalg.norm` over float32 rows of every length the pairwise sum treats
    /// differently: bit for bit.
    #[test]
    fn mean_and_norm_round_as_numpy_rounds() {
        let facts = &fixture()["numpy"];
        let means = floats(&facts["means"]);
        let norms = floats(&facts["norms"]);
        for (at, row) in rows(&facts["rows"]).iter().enumerate() {
            assert_eq!(mean_f32(row), means[at], "mean of a row of {}", row.len());
            assert_eq!(
                l2_norm_f32(row),
                norms[at],
                "norm of a row of {}",
                row.len()
            );
        }
    }

    /// `np.dot` goes to BLAS, whose accumulation order is the platform's; a left-to-right float32
    /// sum agrees with it to within the rounding of the last few additions.
    #[test]
    fn a_dot_product_agrees_with_numpys_to_rounding() {
        let facts = &fixture()["numpy"];
        for case in facts["dots"].as_array().unwrap() {
            let (a, b) = (floats(&case["a"]), floats(&case["b"]));
            let theirs = case["dot"].as_f64().unwrap() as f32;
            let ours = dot_f32(&a, &b);
            let tolerance = 1e-5 * a.len() as f32 * (theirs.abs().max(1.0));
            assert!(
                (ours - theirs).abs() <= tolerance,
                "dot over {}: {ours} vs numpy's {theirs}",
                a.len()
            );
        }
    }

    /// `np.argsort` over eighteen scores with ties: numpy's quicksort orders the ties, and this
    /// is its permutation.
    #[test]
    fn argsort_breaks_ties_as_numpy_does() {
        let facts = &fixture()["numpy"];
        let scores = floats(&facts["tied_scores"]);
        let theirs: Vec<usize> = facts["tied_argsort"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        assert_eq!(argsort_f32(&scores), theirs);
    }
}

mod surfaces {
    use std::sync::Arc;

    use dsrust::lm::DynChatModel;
    use dsrust::lm::embedding::{EmbedCall, EmbeddingFn};
    use dsrust::optimize::BootstrapFewShot;
    use dsrust::retrievers::EmbeddingsWithScores;
    use dsrust::{
        ChainOfThought, DummyLM, DummyVectorizer, Embedder, EmbedderModel, KnnFewShot,
        KnnFewShotProgram, Module, Retrieved, example,
    };

    /// An embedder says what it embeds with.
    #[test]
    fn an_embedder_names_its_model() {
        let named = Embedder::new("openai/text-embedding-3-small");
        assert!(
            matches!(named.model(), EmbedderModel::Named(model) if model == "openai/text-embedding-3-small")
        );
        let function: Arc<EmbeddingFn> =
            Arc::new(|texts, _| Ok(texts.iter().map(|_| vec![1.0]).collect()));
        let callable = Embedder::callable(move |texts, kwargs| function(texts, kwargs));
        assert!(matches!(callable.model(), EmbedderModel::Callable(_)));
    }

    /// A call's overrides ride on `EmbedCall`, and a cached batch is not asked again.
    #[tokio::test]
    async fn a_call_may_override_the_batch_size_and_the_caching() {
        let asked = Arc::new(std::sync::Mutex::new(0usize));
        let counting = Arc::clone(&asked);
        let embedder = Embedder::callable(move |texts, _| {
            *counting.lock().unwrap() += 1;
            Ok(texts.iter().map(|t| vec![t.len() as f32]).collect())
        });
        let inputs = ["a", "bb", "ccc"];
        let once = embedder
            .call_with(
                &inputs,
                EmbedCall {
                    batch_size: Some(1),
                    ..EmbedCall::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(once, vec![vec![1.0], vec![2.0], vec![3.0]]);
        assert_eq!(*asked.lock().unwrap(), 3, "three batches of one");
        embedder
            .call_with(
                &inputs,
                EmbedCall {
                    batch_size: Some(1),
                    ..EmbedCall::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(
            *asked.lock().unwrap(),
            3,
            "every batch answered from the cache"
        );
        embedder
            .call_with(
                &inputs,
                EmbedCall {
                    batch_size: Some(1),
                    caching: Some(false),
                    ..EmbedCall::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(
            *asked.lock().unwrap(),
            6,
            "`caching=False` on the call asks again"
        );
    }

    /// The search answers a `Retrieved`, scores included.
    #[tokio::test]
    async fn a_search_is_a_retrieved() {
        let embedder = Arc::new(Embedder::callable(|texts, _| {
            Ok(texts.iter().map(|t| vec![t.len() as f32, 1.0]).collect())
        }));
        let index = EmbeddingsWithScores::build(
            vec!["ab".to_owned(), "abcd".to_owned()],
            embedder,
            1,
            false,
        )
        .await
        .unwrap();
        let found: Retrieved = index.0.search("abcd").await.unwrap();
        assert_eq!(found.indices, vec![1]);
        assert_eq!(found.passages, vec!["abcd".to_owned()]);
        assert_eq!(found.scores.len(), 1);
    }

    /// `KNNFewShot.compile`'s program: fresh student each call, bootstrapped on the nearest.
    #[tokio::test]
    async fn the_compiled_program_bootstraps_on_the_nearest_each_call() {
        let model = Arc::new(DummyLM::new([
            example! { reasoning: "France's capital is Paris.", answer: "Paris" },
            example! { reasoning: "Belgium's capital is Brussels.", answer: "Brussels" },
        ]));
        let lm = model.clone() as Arc<dyn DynChatModel>;
        let student = move || {
            ChainOfThought::parse("question -> answer")
                .expect("parses")
                .set_lm(lm.clone())
        };
        let vectorizer = DummyVectorizer::default();
        let embedder = Arc::new(Embedder::callable(move |texts, _| {
            Ok(vectorizer.vectorize(texts))
        }));
        let trainset = vec![
            example! { question: "What is the capital of France?", answer: "Paris" }
                .with_inputs(["question"]),
        ];
        let knn_few_shot =
            KnnFewShot::build(3, trainset, embedder, BootstrapFewShot::without_metric())
                .await
                .unwrap();
        assert_eq!(knn_few_shot.knn().k(), 3);
        let program: KnnFewShotProgram<ChainOfThought, _> =
            knn_few_shot.compile(student, None::<fn() -> ChainOfThought>);
        let answered = program
            .forward(example! { question: "What is the capital of Belgium?" })
            .await
            .unwrap();
        assert_eq!(answered.get("answer").unwrap(), "Brussels");
        assert_eq!(
            model.call_count(),
            2,
            "one call to bootstrap the nearest example, one to answer with it as a demo"
        );
    }
}
