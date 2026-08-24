//! The two asks that are not the first one: an extraction, and a retry carrying the error.
//!
//! Both are places where a reply did not settle the matter and the module asks again. dspy puts
//! the extraction in `TwoStepAdapter` and the retry in `Predict`; here they sit together because
//! they are the same shape — a second exchange only the module can make, since an adapter that
//! could call a model would not stay object-safe.

use anyhow::Result;
use serde_json::Value;

use super::{Predict, Steering, Validated};
use crate::adapter::parse::FieldMismatch;
use crate::adapter::{Extraction, Feedback, Input};
use crate::error::Explained;
use crate::lm::{DynChatModel, LmUsage, Sampling, api};

impl<S> Predict<S> {
    /// The second ask of a two-step adapter: hand the first reply to the extraction model and
    /// read the signature's fields out of what it answers.
    ///
    /// The extraction speaks its own adapter over its own signature — `text` in, the task's
    /// outputs out — so nothing here knows it is reading prose rather than a fresh request.
    pub(super) async fn extract(
        &self,
        extraction: Extraction<'_>,
        raw: String,
        asking: Option<LmUsage>,
    ) -> Result<Validated> {
        let text = [Input::new("text", Value::String(raw.clone()))];
        let answered = self
            .ask_extractor(extraction.adapter, &extraction, &text)
            .await?;
        let mut spent = answered.spend();
        let mut extracted_text = answered.first_text();
        let mut parsed = extraction
            .adapter
            .parse(&extraction.signature, &extracted_text);
        // dspy's extraction is a whole `ChatAdapter()(…)` call rather than a `parse`, so it carries
        // that adapter's `__call__` — and a reply the markers cannot read is re-asked through the
        // JSON adapter against the *extraction* model. Reading the reply directly skipped that
        // recovery: an extractor answering in JSON succeeds upstream and failed here.
        if parsed.is_err()
            && let Some(fallback) = extraction.adapter.json_fallback()
        {
            let retried = self
                .ask_extractor(fallback.as_ref(), &extraction, &text)
                .await?;
            spent = LmUsage::merge(spent, retried.spend());
            extracted_text = retried.first_text();
            parsed = fallback.parse(&extraction.signature, &extracted_text);
        }
        // dspy raises its own `AdapterParseError` here rather than letting the extraction
        // adapter's escape, and the two halves go in different places: the *message* carries the
        // failure (`f"…: {e}"`), and `lm_response` carries the **first** reply, not the
        // extraction's. That is the one a caller can act on — an extraction that found nothing
        // usually means the prose never carried the fields, and the prose is what they would go
        // and look at. This had the raw reply written where upstream writes the error, so the
        // sentence named the text and never said what was wrong with it.
        let mut value = parsed.map_err(|error| {
            anyhow::Error::new(FieldMismatch {
                // Upstream passes no `parsed_result`, which is the arm that omits the trailing
                // line rather than printing an empty `[]`.
                parsed: Value::Null,
                adapter_name: "TwoStepAdapter".to_owned(),
                lm_response: raw.clone(),
                expected_fields: self
                    .signature
                    .outputs
                    .iter()
                    .map(|field| field.name.clone())
                    .collect(),
                message: Some(format!(
                    "Failed to parse response from the original completion: {error:#}"
                )),
            })
        })?;
        self.signature.coerce(&mut value)?;
        self.signature.ensure(&value)?;
        Ok(Validated {
            // A recovery ask answers once; there are no alternatives beside it.
            siblings: Vec::new(),
            usage: LmUsage::merge(asking, spent),
            raw: extracted_text,
            value,
        })
    }

    /// One ask of the extraction model through the adapter given — the first attempt and, when the
    /// markers fail, the JSON re-ask, which upstream reaches through `ChatAdapter.__call__`.
    async fn ask_extractor(
        &self,
        adapter: &dyn crate::Adapter,
        extraction: &Extraction<'_>,
        text: &[Input<'_>],
    ) -> Result<api::LmResponse> {
        let messages = adapter.format(&extraction.signature, &[], text)?;
        let schema = extraction.signature.schema();
        let mode = adapter.output_mode(&schema);
        // Left at the provider's defaults rather than given the module's config: this call
        // rewrites prose the model already produced into fields, so a temperature chosen to vary
        // the *answer* would only vary the transcription of one.
        let request = api::request_of(messages, mode, &Sampling::default());
        extraction
            .model
            .forward_dyn(&request)
            .await
            .explain("the extraction model did not answer")
    }

    /// One more ask on the same adapter carrying the rejected reply and its error; every
    /// failure past this point is final.
    pub(crate) async fn feedback_ask(
        &self,
        lm: &dyn DynChatModel,
        inputs: &[Input<'_>],
        feedback: &Feedback,
        steering: &Steering,
    ) -> Result<(String, Value, Option<LmUsage>)> {
        let answered = self.ask(lm, inputs, Some(feedback), steering).await?;
        let answered_text = answered.response.first_text();
        let mut value = crate::observe::parsing(self.adapter.as_ref(), &answered_text, || {
            self.adapter.parse(&self.signature, &answered_text)
        })?;
        self.signature.coerce(&mut value)?;
        self.signature.ensure(&value)?;
        let usage = answered.response.spend();
        Ok((answered_text, value, usage))
    }
}
