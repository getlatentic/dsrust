//! Six probes against a real provider: thinking, JSON, both tool channels, images, and one thing
//! the provider refuses.
//!
//!     export OPENAI_BASE_URL=https://api.deepseek.com
//!     export OPENAI_API_KEY=sk-...
//!     cargo run --example deepseek
//!
//! DeepSeek speaks OpenAI's chat-completions shape, so the standard variables reach it and no
//! builder call is needed for host or key — see `bedrock.rs` for that argument in full. What this
//! file is for is the layer above: whether the *features* a program actually uses survive a
//! provider that is not OpenAI.
//!
//! Most work. The last is the interesting one:
//!
//!   1. **Thinking.** `reasoning_effort` is the OpenAI-shaped control and DeepSeek honours it —
//!      `"low"` thinks briefly, `"none"` not at all. Left unset, `deepseek-v4-flash` reasons
//!      anyway, so the ceiling has to cover the thinking as well as the reply.
//!   2. **JSON.** `JsonAdapter` asks for `{"type": "json_object"}`, which is the default here for
//!      the reason below, and parses the reply into the signature's fields.
//!   3. **Tool calls, over both channels** — and they are not the same thing. A tool *running* is
//!      not a tool call *travelling natively*: `ReAct` parses the tool name out of the model's
//!      prose, and the loop works either way, so watching a tool execute proves nothing about the
//!      wire. Two gates decide, and both must open. The adapter's — `ChatAdapter` defaults native
//!      **off** and `JsonAdapter` **on**, as upstream's do — and the model's, since `native_tools`
//!      refuses when `capabilities.function_calling` is false. Those come from litellm's registry,
//!      so a model that is not in it silently takes the text channel however it is asked. Verified
//!      against a capturing server: `ReAct` sends no `tools` array, `ReActV2` with `JsonAdapter`
//!      sends one.
//!   4. **Images.** Built in memory and sent inline as base64. DeepSeek fetches a URL *server
//!      side* and could not reach Wikipedia from its own network, which says nothing about the
//!      caller — inline is the path a caller controls, and `Image::from_path`/`from_bytes` take it.
//!   5. **Schema-constrained decoding is refused**, with `This response_format type is unavailable
//!      now`. That is why [`JsonFormat::Object`] is the default and [`JsonFormat::Schema`] is
//!      opt-in: strict mode is an OpenAI feature that OpenAI-shaped services need not implement,
//!      and a port that assumed it would fail against most of them. Shown here rather than hidden,
//!      because the error naming the model and quoting the provider is the useful behaviour.

use dsrust::lm::api::{LmConfig, LmReasoningConfig};
use dsrust::lm::{JsonFormat, configure};
use dsrust::{
    FnTool, Image, JsonAdapter, LM, Module, Predict, Prediction, ReAct, ReActV2, Tool, call, input,
    make_signature,
};
use serde_json::Value;

const CHAT_MODEL: &str = "deepseek-v4-flash";
const VISION_MODEL: &str = "deepseek-v4-flash-vision-exp";

fn model(name: &str, effort: Option<&str>) -> anyhow::Result<LM> {
    let mut config = LmConfig {
        // Room for the thinking as well as the answer.
        max_tokens: Some(2048),
        ..Default::default()
    };
    if let Some(effort) = effort {
        config.reasoning = Some(LmReasoningConfig {
            effort: Some(effort.to_owned()),
            ..Default::default()
        });
    }
    LM::builder(format!("openai/{name}"))
        .config(config)
        // Off so a second run asks the provider again rather than answering from `~/.dsrs_cache`,
        // which is what a capability probe has to do to mean anything.
        .cache(false)
        .build()
}

fn answer(prediction: &Prediction) -> Option<&str> {
    prediction.get("answer").and_then(Value::as_str)
}

fn weather() -> Box<dyn Tool> {
    Box::new(FnTool::new(
        "get_weather",
        "look up the weather for a city",
        serde_json::json!({ "city": { "type": "string" } }),
        |args: &Value| {
            let city = args.get("city").and_then(Value::as_str).unwrap_or("?");
            println!("      [this process ran get_weather({city})]");
            Ok(format!("The weather in {city} is sunny, 21C."))
        },
    ))
}

/// A red square with one blue quadrant, so "what colours" has a checkable answer.
fn two_colour_image() -> anyhow::Result<Image> {
    let (width, height) = (64u32, 64u32);
    let mut pixels = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            let blue = x < width / 2 && y < height / 2;
            pixels.extend_from_slice(if blue { &[0u8, 0, 255] } else { &[255u8, 0, 0] });
        }
    }
    Image::from_rgb(width, height, &pixels)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    configure(model(CHAT_MODEL, Some("low"))?);
    let asked = call!(
        Predict!("question -> answer"),
        question = "What is 17 * 23?"
    )
    .await?;
    println!("1 thinking     {:?}", answer(&asked));

    let json = Predict!("question -> city, country").adapter(JsonAdapter::default());
    let asked = call!(json, question = "Name a city in Peru.").await?;
    println!(
        "2 json         city={:?} country={:?}",
        asked.get("city").and_then(Value::as_str),
        asked.get("country").and_then(Value::as_str)
    );

    // `ReAct` answers through `Module::forward` rather than `call!`: `Ask` is written per module,
    // and an agent is not one of the three that has it.
    let agent = ReAct!("request -> answer", vec![weather()], max_iters = 4);
    let asked = agent
        .forward(input! { request: "What is the weather in Paris?" })
        .await?;
    println!("3 tools (text) {:?}", answer(&asked));

    // Instructions live on the signature, as they do upstream where they are a signature class's
    // docstring, so this one is built rather than named. `JsonAdapter` is what opens the native
    // tool channel, since `ChatAdapter` defaults it off exactly as dspy's does.
    let guided = make_signature!("request -> answer")
        .with_instructions("Answer the question, using tools when they help.");
    let agent = ReActV2::new(guided, vec![weather()]).adapter(JsonAdapter::default());
    let asked = agent
        .forward(input! { request: "What is the weather in Berlin?" })
        .await?;
    println!("4 tools (wire) {:?}", answer(&asked));

    configure(model(VISION_MODEL, None)?);
    let asked = call!(
        Predict!("question, photo: Image -> answer"),
        question = "What two colours are in this image? Answer in three words.",
        photo = serde_json::to_value(two_colour_image()?)?
    )
    .await?;
    println!("5 image        {:?}", answer(&asked));

    configure(model(CHAT_MODEL, None)?.openai_json_format(JsonFormat::Schema));
    let strict = Predict!("question -> city").adapter(JsonAdapter::default());
    match call!(strict, question = "Name a city.").await {
        Ok(asked) => println!("6 json_schema  {:?}", asked.get("city")),
        Err(error) => println!(
            "6 json_schema  refused: {}",
            error.to_string().lines().next().unwrap_or_default()
        ),
    }
    Ok(())
}
