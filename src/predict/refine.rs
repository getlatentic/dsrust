//! dspy `Refine` (`predict/refine.py`): ask again, having been told what went wrong.

// `describe` and the advice half of `feedback` are written and tested ahead of the loop that
// will call them. The allow goes when `Refine::forward` lands, and suppresses nothing: both are
// exercised by their own tests today.
#[allow(dead_code)]
mod describe;
#[allow(dead_code)]
pub mod feedback;
