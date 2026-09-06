# Cloudflare Workers support

DsRust supports text inference on Cloudflare Workers through the `inference` capability. Worker
consumers must disable the native defaults:

```toml
dsrust = { version = "0.1.0-alpha.3", default-features = false, features = ["inference"] }
```

This build includes Predict, Chain of Thought, ReAct, local tools, streaming, retries, tracing,
request-scoped callbacks and usage tracking, and the Anthropic, OpenRouter, and OpenAI-compatible
HTTP providers. Use `lm::Cached` when a request-owned in-memory cache is required. Process-global
models, callbacks, usage trackers, and caches are deliberately unavailable in Worker builds.

The executable fixture at [`examples/cloudflare-worker`](../examples/cloudflare-worker) is the
deployment contract. It builds with `worker-build`, runs under local `workerd` through Wrangler,
and tests Predict, streaming, ReAct tool use, and a request-local cache hit without credentials.

## Capability phases

| Phase | Status | Scope |
| --- | --- | --- |
| 1 | Implemented | Existing HTTP providers, text programs, tools, streaming, retry, tracing, scoped usage, request-owned memory cache |
| 2 | Planned | Injected Workers AI binding and KV-backed cache using the Phase 1 cache key |
| 3 | Planned | URL/data/byte image inputs, followed by byte/encoded audio inputs behind codec features |
| 4 | Planned | Worker-native retrieval and storage integrations; assessment of Vectorize, R2, KV, Durable Objects, and external sandboxes |

Ollama, Deno and local interpreters, filesystem persistence, optimizers and training, native batch
evaluation, thread-based parallelism, and native retriever persistence remain native-only. Enabling
the `native` or `process-global` feature on `wasm32-unknown-unknown` produces a targeted compile
error instead of an indirect dependency failure.

The Phase 1 artifact uses a size-focused Cargo profile and `wasm-opt -Os`. CI checks dsrust's
inference package independently, builds the final Worker consumer, runs all fixture routes, rejects
artifacts above Cloudflare's 64 MiB uncompressed limit, and rejects growth above 5% of the committed
baseline.
