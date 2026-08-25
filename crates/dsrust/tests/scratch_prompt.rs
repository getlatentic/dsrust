#[test]
fn literal_prompt() {
    use dsrust::adapter::{Adapter, ChatAdapter};
    use dsrust::signature::{OutField, Signature};
    let s = Signature::single_input(
        "Pick a colour.",
        vec![OutField {
            name: "colour".into(),
            values: Some(vec!["red".into(), "blue".into()]),
            ..Default::default()
        }],
    );
    println!(
        "{}",
        ChatAdapter::default().system_message(&s).expect("renders")
    );
}
