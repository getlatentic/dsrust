//! Reading several completions from one call — the shape an instruction optimizer's proposal step
//! returns. dspy's `Predict(sig, n=N)(...)`, whose `Prediction.completions.<field>` is a list of the
//! candidates the model proposed; a coordinate-ascent optimizer walks it to collect them.

use anyhow::Result;

use super::{Predict, Steering, rendered_inputs};
use crate::example::{Example, Prediction};

impl<S> Predict<S> {
    /// Ask the model for several candidate answers in one call and read every one, not just the
    /// first — dspy's `Predict(sig, n=N)(...)`. How many is
    /// [`config.completions`](crate::lm::LmConfig) (set with [`with_config`](Self::with_config));
    /// each candidate is parsed on its own, and one the model malformed is dropped rather than
    /// failing the batch, since a proposal step wants the candidates that came through, not all or
    /// nothing.
    pub async fn forward_completions(&self, inputs: Example) -> Result<Vec<Prediction>> {
        let lm = self.asking()?;
        let (pairs, predicted_output) = rendered_inputs(&inputs);
        let steering = Steering {
            predicted_output,
            ..Steering::default()
        };
        let answered = self.ask(lm.as_ref(), &pairs, None, &steering).await?;
        let mut usage = answered.response.spend();
        let mut predictions = Vec::new();
        for output in &answered.response.outputs {
            let text = output.as_text();
            let Ok(mut value) = self.adapter.parse(&self.signature, &text) else {
                continue;
            };
            let _ = self.signature.coerce(&mut value);
            let mut prediction = Prediction::new(super::prediction_example(&value), text);
            // The call's usage is charged once, to the first candidate that came through.
            if usage.is_some() {
                prediction = prediction.with_usage(usage.take());
            }
            predictions.push(prediction);
        }
        Ok(predictions)
    }
}
