//! dspy's KNN family under upstream's own tests: `Embedder`, `KNN`, `Embeddings`.
//!
//! An embedder here batches and caches through the crate and reaches a Python callable for the
//! vectors — dspy's `DummyVectorizer`, or a test's own function — so the decision under test is
//! the crate's while the vectors are whatever Python's vectorizer makes of the texts the crate
//! rendered. A KNN selection and a retriever search cross the same way.

use std::sync::Arc;

use dsrust::Example;
use dsrust::lm::embedding::{EmbedCall, Embedder};
use dsrust::predict::knn;
use dsrust::retrievers::Embeddings;
use pyo3::exceptions::{PyFileNotFoundError, PyValueError};
use pyo3::prelude::*;
use serde_json::{Map, Value};

fn to_value_error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn kwargs_of(kwargs: Option<String>) -> PyResult<Map<String, Value>> {
    match kwargs {
        None => Ok(Map::new()),
        Some(text) => serde_json::from_str(&text).map_err(to_value_error),
    }
}

/// `dspy.Embedder`, backed by the crate: a Python callable for the vectors, or a hosted model.
#[pyclass]
pub(crate) struct RustEmbedder {
    pub(crate) inner: Arc<Embedder>,
}

#[pymethods]
impl RustEmbedder {
    #[new]
    #[pyo3(signature = (callable=None, model=None, batch_size=200, caching=true, kwargs=None))]
    fn new(
        callable: Option<Py<PyAny>>,
        model: Option<String>,
        batch_size: usize,
        caching: bool,
        kwargs: Option<String>,
    ) -> PyResult<Self> {
        let default_kwargs = kwargs_of(kwargs)?;
        let embedder = match (callable, model) {
            (Some(callable), _) => Embedder::callable(move |batch, kwargs| {
                Python::attach(|py| {
                    let kwargs_json = serde_json::to_string(kwargs)?;
                    let answered = callable
                        .call1(py, (batch.to_vec(), kwargs_json))
                        .map_err(|error| anyhow::anyhow!("{error}"))?;
                    answered.extract::<Vec<Vec<f32>>>(py).map_err(|error| {
                        anyhow::anyhow!(
                            "the embedder answered something other than rows of floats: {error}"
                        )
                    })
                })
            }),
            (None, Some(model)) => Embedder::new(model),
            (None, None) => {
                return Err(PyValueError::new_err(
                    "`model` in `dspy.Embedder` must be a string or a callable",
                ));
            }
        };
        Ok(Self {
            inner: Arc::new(
                embedder
                    .batch_size(batch_size)
                    .caching(caching)
                    .kwargs(default_kwargs),
            ),
        })
    }

    /// `embedder(inputs, batch_size=..., caching=..., **kwargs)`, inputs already a list.
    #[pyo3(signature = (inputs, batch_size=None, caching=None, kwargs=None))]
    fn call(
        &self,
        py: Python<'_>,
        inputs: Vec<String>,
        batch_size: Option<usize>,
        caching: Option<bool>,
        kwargs: Option<String>,
    ) -> PyResult<Vec<Vec<f32>>> {
        let call = EmbedCall {
            batch_size,
            caching,
            kwargs: kwargs_of(kwargs)?,
        };
        let inner = Arc::clone(&self.inner);
        py.detach(move || pollster::block_on(inner.call_with(&inputs, call)))
            .map_err(to_value_error)
    }
}

/// One serialised example: its fields and, when marked, its input keys.
fn example_of(value: &Value) -> Example {
    let fields = value["fields"]
        .as_object()
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .collect::<Vec<(String, Value)>>();
    let example = Example::new(fields);
    match value["input_keys"].as_array() {
        Some(keys) => example.with_inputs(keys.iter().filter_map(Value::as_str).map(str::to_owned)),
        None => example,
    }
}

/// `KNN.__init__`'s rendering of each training example, before it is embedded.
#[pyfunction]
fn knn_texts(trainset_json: &str) -> PyResult<Vec<String>> {
    let trainset: Vec<Value> = serde_json::from_str(trainset_json).map_err(to_value_error)?;
    trainset
        .iter()
        .map(|value| knn::example_text(&example_of(value)).map_err(to_value_error))
        .collect()
}

/// `KNN.__call__`'s rendering of the call's inputs, before they are embedded.
#[pyfunction]
fn knn_query_text(inputs_json: &str) -> PyResult<String> {
    let inputs: Map<String, Value> = serde_json::from_str(inputs_json).map_err(to_value_error)?;
    Ok(knn::query_text(&Example::new(
        inputs.into_iter().collect::<Vec<_>>(),
    )))
}

/// `KNN.__call__`'s selection: the k nearest rows, nearest first.
#[pyfunction]
fn knn_select(trainset_vectors: Vec<Vec<f32>>, query: Vec<f32>, k: usize) -> Vec<usize> {
    knn::nearest_indices(&trainset_vectors, &query, k)
}

/// `dspy.retrievers.Embeddings`, backed by the crate: the index, its search, and its files.
#[pyclass]
pub(crate) struct RustIndex {
    inner: Embeddings,
}

#[pymethods]
impl RustIndex {
    #[new]
    fn new(
        py: Python<'_>,
        corpus: Vec<String>,
        embedder: PyRef<'_, RustEmbedder>,
        k: usize,
        normalize: bool,
    ) -> PyResult<Self> {
        let embedder = Arc::clone(&embedder.inner);
        let inner = py
            .detach(move || pollster::block_on(Embeddings::build(corpus, embedder, k, normalize)))
            .map_err(to_value_error)?;
        Ok(Self { inner })
    }

    /// `Embeddings.from_saved(path, embedder)`.
    #[staticmethod]
    fn load(path: String, embedder: PyRef<'_, RustEmbedder>) -> PyResult<Self> {
        // A path with no index at it is upstream's `FileNotFoundError`, which `open` raises there.
        let config = std::path::Path::new(&path).join("config.json");
        if !config.exists() {
            return Err(PyFileNotFoundError::new_err(format!(
                "[Errno 2] No such file or directory: '{}'",
                config.display()
            )));
        }
        let inner =
            Embeddings::from_saved(&path, Arc::clone(&embedder.inner)).map_err(to_value_error)?;
        Ok(Self { inner })
    }

    fn search(
        &self,
        py: Python<'_>,
        query: String,
    ) -> PyResult<(Vec<String>, Vec<usize>, Vec<f32>)> {
        let found = py
            .detach(|| pollster::block_on(self.inner.search(&query)))
            .map_err(to_value_error)?;
        Ok((found.passages, found.indices, found.scores))
    }

    fn save(&self, path: String) -> PyResult<()> {
        self.inner.save(&path).map_err(to_value_error)
    }

    fn k(&self) -> usize {
        self.inner.k()
    }

    fn normalize(&self) -> bool {
        self.inner.normalize()
    }

    fn corpus(&self) -> Vec<String> {
        self.inner.corpus().to_vec()
    }

    fn corpus_embeddings(&self) -> Vec<Vec<f32>> {
        self.inner.corpus_embeddings().to_vec()
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<RustEmbedder>()?;
    module.add_class::<RustIndex>()?;
    module.add_function(wrap_pyfunction!(knn_texts, module)?)?;
    module.add_function(wrap_pyfunction!(knn_query_text, module)?)?;
    module.add_function(wrap_pyfunction!(knn_select, module)?)?;
    Ok(())
}
