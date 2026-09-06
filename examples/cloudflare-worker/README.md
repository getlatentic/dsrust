# dsrust Cloudflare Worker — Phase 1

This executable fixture proves dsrust text inference, streaming responses, ReAct tools, and an
explicitly owned in-memory cache under the Cloudflare Workers runtime. It uses `DummyLM`, so local
verification requires no API key and makes no paid provider request.

```sh
cargo install worker-build --version 0.7.4 --locked
npm install
npm run build
npm run check:size
npm run test:local
npm run dev
```

Exercise `POST /predict`, `/stream`, `/react`, and `/cache-check` with a JSON body such as
`{"question":"capital of France?"}`. The cache route performs two calls inside one request and
reports `second_cache_hit: true`; it deliberately does not retain request-owned state globally.

Existing OpenAI-compatible, OpenRouter, and Anthropic providers compile into the same inference
feature. Supply credentials through Worker secrets when replacing the deterministic fixture model.
Ollama, filesystem persistence, interpreters, optimizers, and other native-only predictors are not
part of Phase 1.

The fixture pins `worker` and `worker-build` to 0.7.4 because reqwest 0.12 and that workers-rs
release share `wasm-streams` 0.4. Newer workers-rs releases currently introduce a second
`wasm-streams` ABI whose duplicate wasm-bindgen exports prevent a final consumer build. The pin is
covered by the real `worker-build` and `wrangler dev` checks and should be removed when those
dependency graphs converge.
