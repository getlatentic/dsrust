//! The `/api/generate` route against a real daemon. Ignored by default like the other live tests.
use dsrust::lm::{ChatModel, LM, api};

#[tokio::test]
#[ignore = "needs a live ollama with llama3.2:1b pulled"]
async fn the_generate_route_answers() {
    let lm = LM::new("ollama/llama3.2:1b").expect("ref").without_cache();
    let req = api::LmRequest::new(
        "llama3.2:1b",
        vec![api::LmMessage::new("user", vec![api::LmPart::text("Reply with the single word: ready")])],
    );
    let answered = lm.forward(&reqwest::Client::new(), &req).await.expect("a reply");
    assert!(!answered.first_text().is_empty(), "got: {:?}", answered.first_text());
    eprintln!("generate route replied: {:?}", answered.first_text());
}
