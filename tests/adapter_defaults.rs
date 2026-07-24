//! The adapter constructor defaults, held to dspy 3.3.0b1's own.
//!
//! These are read off `dspy.ChatAdapter()` / `dspy.JSONAdapter()` at the Python REPL, not inferred:
//! upstream's JSONAdapter deliberately differs from every other adapter, and a `#[derive(Default)]`
//! that happened to say `false` looked right while contradicting it.

use dsrust::{ChatAdapter, JsonAdapter};

#[test]
fn chat_defaults_match_dspys_chat_adapter() {
    let chat = ChatAdapter::default();
    // dspy: use_json_adapter_fallback=True, use_native_function_calling=False,
    // parallel_tool_calls=None.
    assert!(chat.use_json_adapter_fallback);
    assert!(!chat.use_native_function_calling);
    assert_eq!(chat.parallel_tool_calls, None);
}

#[test]
fn json_defaults_to_native_function_calling_as_dspy_does() {
    // dspy's JSONAdapter takes `use_native_function_calling: bool = True` — "JSONAdapter uses
    // native function calling by default" — where ChatAdapter takes False.
    let json = JsonAdapter::default();
    assert!(
        json.use_native_function_calling,
        "dspy.JSONAdapter() defaults native function calling on"
    );
    assert_eq!(json.parallel_tool_calls, None);
}

#[test]
fn a_chat_fallback_carries_the_chat_adapters_settings_not_the_json_defaults() {
    // dspy `_make_json_adapter_fallback` passes its own settings down, so a re-ask from a plain
    // ChatAdapter must not silently pick up JSONAdapter's native-on default.
    let fallback = ChatAdapter::default().json_fallback_adapter().expect("falls back");
    assert!(!fallback.use_native_function_calling);

    let native = ChatAdapter::default()
        .with_native_function_calling(true)
        .with_parallel_tool_calls(Some(true));
    let fallback = native.json_fallback_adapter().expect("falls back");
    assert!(fallback.use_native_function_calling);
    assert_eq!(fallback.parallel_tool_calls, Some(true));

    assert!(ChatAdapter::without_json_fallback().json_fallback_adapter().is_none());
}
