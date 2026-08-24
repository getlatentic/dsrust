//! The two asks that are not the first one: an extraction, and a retry carrying the error.
//!
//! Both are places where a reply did not settle the matter and the module asks again. dspy puts
//! the extraction in `TwoStepAdapter` and the retry in `Predict`; here they sit together because
//! they are the same shape — a second exchange only the module can make, since an adapter that
//! could call a model would not stay object-safe.

use anyhow::{Context, Result};
use serde_json::Value;

use super::{Predict, Steering, Validated};
use crate::adapter::{Extraction, Feedback, Input};
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
        let messages = extraction
            .adapter
            .format(&extraction.signature, &[], &text)?;
        let schema = extraction.signature.schema();
        let mode = extraction.adapter.output_mode(&schema);
        // Left at the provider's defaults rather than given the module's config: this call
        // rewrites prose the model already produced into fields, so a temperature chosen to
        // vary the *answer* would only vary the transcription of one.
        let request = api::request_of(messages, mode, &Sampling::default());
        let extracted = extraction
            .model
            .forward_dyn(&request)
            .await
            .context("the extraction model did not answer")?;
        let extracted_text = extracted.first_text();
        let mut value = extraction
            .adapter
            .parse(&extraction.signature, &extracted_text)
            // dspy names the *first* reply here, not the extraction's. That is the one a
            // caller can act on: the extraction failing usually means the prose never carried
            // the fields, and the prose is what they would go and look at.
            .with_context(|| {
                format!("Failed to parse response from the original completion: {raw}")
            })?;
        self.signature.coerce(&mut value)?;
        self.signature.ensure(&value)?;
        Ok(Validated {
            // A recovery ask answers once; there are no alternatives beside it.
            siblings: Vec::new(),
            usage: LmUsage::merge(asking, extracted.spend()),
            raw: extracted_text,
            value,
        })
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
