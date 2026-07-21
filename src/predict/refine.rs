//! dspy `Refine` (`predict/refine.py`): ask again, having been told what went wrong.

// `describe` is written and tested ahead of the loop that will call it, so nothing outside this
// module reaches it yet. The allow goes when `Refine::forward` lands and is not a suppression of
// anything: the functions are exercised by their own tests today.
#[allow(dead_code)]
mod describe;
