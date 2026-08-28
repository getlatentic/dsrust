//! SIMBA's `OfferFeedback` against dspy's own, rendered.
//!
//! Two signatures in dspy carry this name and they are different: `refine.py`'s assigns blame
//! against a threshold over one trajectory, and `simba_utils.py`'s contrasts a worse trajectory
//! against a better one. Thirteen input fields against nine. Porting one and reusing it for the
//! other would send a model the wrong question, which no test of the search loop would catch —
//! the loop would still make the same *decisions*, and only the prompt would be wrong.
//!
//! Compared as bytes: the golden is the signature rendered through dspy's `ChatAdapter`.

use dsrust::optimize::simba::feedback::offer_feedback;
use dsrust::{Adapter, ChatAdapter};
use serde_json::Value;

fn golden() -> Value {
    serde_json::from_str(include_str!(
        "conformance/optimize/simba_offer_feedback.json"
    ))
    .expect("the golden parses")
}

#[test]
fn the_fields_are_dspys() {
    let fixture = golden();
    let ours = offer_feedback();
    let theirs = &fixture["signature"];

    assert_eq!(
        ours.instructions,
        theirs["instructions"].as_str().expect("instructions"),
        "instructions"
    );
    for (ours, theirs) in ours
        .inputs
        .iter()
        .zip(theirs["inputs"].as_array().expect("inputs"))
    {
        assert_eq!(
            ours.name,
            theirs["name"].as_str().expect("a name"),
            "input name"
        );
        assert_eq!(
            ours.desc,
            theirs["desc"].as_str().expect("a desc"),
            "`{}` desc",
            ours.name
        );
    }
    assert_eq!(
        ours.inputs.len(),
        theirs["inputs"].as_array().expect("inputs").len(),
        "input count"
    );
    for (ours, theirs) in ours
        .outputs
        .iter()
        .zip(theirs["outputs"].as_array().expect("outputs"))
    {
        assert_eq!(
            ours.name,
            theirs["name"].as_str().expect("a name"),
            "output name"
        );
        assert_eq!(
            ours.desc,
            theirs["desc"].as_str().expect("a desc"),
            "`{}` desc",
            ours.name
        );
    }

    // And it really is a different signature from refine's, rather than the same one twice.
    assert_ne!(
        ours.inputs.len(),
        fixture["refine_signature_field_count"]
            .as_u64()
            .expect("refine's count") as usize,
        "if these ever match, check that the two OfferFeedbacks have not converged"
    );
}

/// The prompt itself, byte for byte against what dspy's own adapter rendered.
#[test]
fn the_rendered_prompt_is_dspys() {
    let fixture = golden();
    let rendered = ChatAdapter::default()
        .format(&offer_feedback(), &[], &[])
        .expect("the signature renders");
    let theirs = fixture["rendered"].as_array().expect("messages")[0]["content"]
        .as_str()
        .expect("dspy's system message");
    let ours = rendered
        .first()
        .expect("a system message")
        .text()
        .expect("text");
    assert_eq!(ours, theirs, "system message");
}

/// SIMBA's `inspect_modules` and refine's differ only by an unused `enumerate`, so the crate shares
/// one helper. Asserted from dspy rather than read off the source.
#[test]
fn inspect_modules_is_the_one_refine_uses() {
    assert!(
        golden()["inspect_modules_matches_refines"]
            .as_bool()
            .expect("a verdict"),
        "dspy's two `inspect_modules` have diverged; the shared Rust helper is no longer justified"
    );
}
