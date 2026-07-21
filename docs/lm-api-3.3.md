# dspy 3.3's normalized LM API, in full

Read from `dspy-3.3.0b1/dspy/core/types.py`. This is the reference for the LM layer; **3.2.1 stays
the pin for prompt bytes**. Regenerate with:

```sh
curl -sL "$(curl -sL https://pypi.org/pypi/dspy/3.3.0b1/json \
  | python3 -c "import json,sys;print([u['url'] for u in json.load(sys.stdin)['urls'] if u['packagetype']=='bdist_wheel'][0])")" \
  -o dspy33.whl && unzip -o -q dspy33.whl -d dspy33
```

## The request

```python
class LMRequest:
    model: str
    messages: list[LMMessage]
    tools: list[LMToolSpec]
    config: LMConfig
    metadata: dict[str, Any]
```

## LMConfig — 12 fields, and the alias rule

```python
class LMConfig:
    temperature: float | None
    max_tokens: int | None
    top_p: float | None
    stop: list[str] | None
    n: int | None
    logprobs: bool | int | None
    response_format: Any | None
    reasoning: LMReasoningConfig | None
    tool_choice: LMToolChoice | None
    cache: LMCacheConfig | None
    prompt_cache: LMPromptCacheConfig | None
    extensions: dict[str, Any]
    model_config = ConfigDict(extra="forbid")
```

**It does not enumerate provider parameters, and does not try to.** `_KNOWN_CONFIG_KEYS` holds 16
entries; anything outside them goes to `extensions`. So `top_k`, `seed`, `logit_bias`,
`frequency_penalty` and whatever a provider adds next all land in the bag.

Four recognised keys are *not* fields. They are flat input spellings folded into structured fields
before construction, which `extra="forbid"` then requires — an unfolded key raises:

| input key | folds into |
|---|---|
| `reasoning_effort` | `reasoning.effort` |
| `parallel_tool_calls` | `tool_choice.parallel` |
| `rollout_id` | **`cache.rollout_id`** |
| `prompt_cache_key` | `prompt_cache.key` |

### The nested configs

```python
class LMReasoningConfig:    effort: str|None; max_tokens: int|None; summary: str|None
class LMToolChoice:         mode: "auto"|"required"|"none" = "auto"; allowed: list[str]|None; parallel: bool|None
class LMCacheConfig:        enabled: bool|None; rollout_id: int|str|None
class LMPromptCacheConfig:  enabled: bool|None; key: str|None
```

## The response

```python
class LMResponse:
    model: str | None
    outputs: list[LMOutput]          # min_length=1
    usage: LMUsage | dict | None
    cost: float | None
    cache_hit: bool = False
    response_id: str | None
    provider_response: Any | None    # excluded from serialization
    provider_data: dict[str, Any]
    metadata: dict[str, Any]

class LMOutput:
    parts: list[LMPart]
    finish_reason: str | None
    truncated: bool = False
    logprobs: Any | None
    provider_output: Any | None      # excluded
    provider_data: dict[str, Any]
    metadata: dict[str, Any]

class LMUsage:
    input_tokens, output_tokens, total_tokens,
    prompt_tokens, completion_tokens, reasoning_tokens,
    cache_read_tokens, cache_write_tokens,
    input_audio_tokens, output_audio_tokens: int | None
    details: dict[str, Any]
    model_config = ConfigDict(extra="allow")     # allow, unlike LMConfig's forbid
```

`LMUsage` keeps **both** naming conventions rather than normalizing to one — "Both DSPy token
names and provider token names are populated because both are existing user-visible interfaces."
A `fill_aliases` validator mirrors them after construction, in both directions, and computes the
total:

    input_tokens  <-> prompt_tokens
    output_tokens <-> completion_tokens
    total_tokens   = input_tokens + output_tokens   (when both are known and total is not)

Two things separate it from `LMConfig`: `extra="allow"` rather than `forbid`, so an unknown
counter attaches rather than raising, and `details` for structured provider breakdowns. This crate
normalizes to one pair of names, which is a deliberate simplification that s13-1 has to unpick —
a caller reading `prompt_tokens` upstream finds nothing here.

## Messages and parts

```python
class LMMessage:   role: str; parts: list[LMPart]; name: str|None; metadata: dict
class LMToolSpec:  type: "function"; name: str; description: str|None; parameters: dict; metadata; provider_data
```

`LMPart` is a hierarchy, not a string, and it has **eleven** members — read off
`dspy/core/types.py` at 3.3.0b1, not summarised:

```python
LMBasePart(type: str, metadata: dict)                       # extra="forbid"
├─ LMTextPart(text)                                          type="text"
├─ LMSourcePart(media_type, data|url|file_id|path)           @validate_one_source
│  ├─ LMImagePart(detail: low|high|auto|None)                type="image",  media_type="image/png"
│  ├─ LMAudioPart                                            type="audio",  media_type="audio/wav"
│  ├─ LMVideoPart                                            type="video",  media_type="video/mp4"
│  └─ LMBinaryPart(filename)                                 type="binary", media_type="application/octet-stream"
├─ LMDocumentPart(data|url|file_id|path|source, citations,   type="document", media_type="application/pdf"
│                 title, context)                            @validate_source — source XOR a media source
├─ LMToolCallPart(id, name, args, provider_data)             type="tool_call"
├─ LMToolResultPart(call_id, name, content: list[LMPart],    type="tool_result" — recursive
│                   is_error, provider_data)
├─ LMThinkingPart(text, redacted)                            type="thinking"
├─ LMCitationPart(text|title|url)                            type="citation" — @validate_has_content
└─ LMRefusalPart(text)                                       type="refusal"

LMPart = Annotated[<the eleven>, Field(discriminator="type")]
```

`LMDocumentPart` notably does **not** extend `LMSourcePart` — it carries the same four source
fields plus a `source` dict, and its validator is the stricter `source` XOR media-source rule.
Modelling it as an image sibling would be wrong.

### The two mechanisms that keep the bytes still

A typed part tree does not itself reach a provider. `dspy/clients/openai_format.py` converts it
back to the OpenAI content blocks 3.2.1 already sent, and two rules do all the work:

- **`parts_to_openai_content`** returns a *bare string* when the message is exactly one
  `LMTextPart` with no `legacy_content_block`, and a block list otherwise. That is precisely this
  crate's `Content::Text` vs `Content::Blocks` split, restated as a function of the parts — so the
  split stops being a type and becomes a rendering rule.
- **`metadata["legacy_content_block"]`** holds a provider-shaped block verbatim, and
  `part_to_openai_blocks` returns it untouched ahead of every other branch.
  `adapters/_legacy_type_markers.py` — the 3.3 successor to this crate's `adapter/blocks.rs` —
  parks any block it cannot classify on `LMTextPart(text="", metadata={...})`. That is upstream's
  own answer to "a custom type wrote JSON we do not model", and it is what makes the port
  lossless rather than best-effort.

Measured, not argued — 3.2.1's rendered blocks through 3.3's
`_legacy_content_block_to_lm_part` → `parts_to_openai_content` come back identical, including for
an unmodelled `{"type": "wildcard_v9", …}` block.

## Where this crate stands against it

| upstream | ours | gap |
|---|---|---|
| `LMRequest{model, messages, tools, config, metadata}` | `LmRequest{system, turns, mode, config}` | no `model`, no `tools`, no `metadata`; `mode` is upstream's `config.response_format` |
| `LMConfig` — 12 fields | `LmConfig` — 4 | missing 8 typed fields and `extensions` |
| `rollout_id` at `cache.rollout_id`, typed `int｜str` | flat `rollout_id: Option<u64>` | wrong home and narrower type |
| `LMResponse` — 9 fields | `LmResponse` — 4 | missing `model`, `cost`, `response_id`, `provider_response`, `metadata` |
| `outputs: list[LMOutput]`, min 1 | `outputs: Vec<String>` | a candidate is a structure, not a string: no `finish_reason`, `truncated`, `logprobs` |
| `LMUsage` — 10 counters | `Usage` — 2 | no totals, reasoning, cache read/write, or audio counters |
| `LMMessage{role, parts, name, metadata}` | `ChatTurn{role, content}` | no `name`, no `metadata`; `Content` is text-or-blocks rather than a typed part hierarchy |

The largest single divergence is `outputs`: ours is a list of strings, upstream's is a list of
structured candidates. `finish_reason` and `truncated` in particular are things a caller currently
cannot see at all.

## Porting pydantic to serde

Most of it maps directly.

| pydantic | serde |
|---|---|
| `x: int \| None = None` | `Option<u32>` + `#[serde(default, skip_serializing_if = "Option::is_none")]` |
| `Field(default_factory=dict)` | `#[serde(default)]` |
| `extra="forbid"` | `#[serde(deny_unknown_fields)]` |
| `extra="allow"` | `#[serde(flatten)] extra: Map<String, Value>` — cannot combine with `deny_unknown_fields`, and does not need to |
| an input key under another name | `#[serde(alias = "…")]`, which accepts several spellings on the way in |

Two things it does not do.

**Validators do not run on construction.** `@model_validator(mode="after")` fires every time
pydantic builds the model; a Rust struct literal has no hook at all. `LMUsage::fill_aliases` —
mirroring `input_tokens` ↔ `prompt_tokens` and deriving `total_tokens` — therefore cannot be a
derive. Two options, and the sprint should pick one and keep to it:

- a constructor (`LmUsage::new(...)` / `from_parts`) that normalizes, with the literal left
  un-normalized. Cheap, and a caller who writes the literal silently skips it.
- `#[serde(from = "ShadowLmUsage")]`, deserializing into a private twin and converting through the
  same normalizer. Covers the deserialize path — the disk cache — but still not direct literals.

Both are needed for full cover: the shadow for data arriving, the constructor for code building.
`LmConfig::from_kwargs` is the same shape, folding the four flat aliases into their nested homes.

**Some validators should not be ported at all.** `LMSourcePart` validates that exactly one of
`data`, `url`, `file_id`, `path` is set. That constraint exists because Python cannot say it in a
type — Rust can:

```rust
enum LmSource { Data(String), Url(String), FileId(String), Path(PathBuf) }
```

which makes the invalid state unrepresentable rather than rejected. Prefer that wherever a
validator is enforcing a shape the type system can carry, and keep a runtime check only where the
rule is genuinely about values.
