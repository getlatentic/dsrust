import { spawn } from "node:child_process";
import net from "node:net";

async function freePort(preferred) {
  for (let port = preferred; port < preferred + 100; port += 1) {
    const available = await new Promise((resolve) => {
      const server = net
        .createServer()
        .once("error", () => resolve(false))
        .once("listening", () => server.close(() => resolve(true)))
        .listen(port, "127.0.0.1");
    });
    if (available) return port;
  }
  throw new Error(`No free local port found from ${preferred}`);
}

const port = await freePort(8787);
const baseUrl = `http://127.0.0.1:${port}`;
const command = process.platform === "win32" ? "npm.cmd" : "npm";
const server = spawn(
  command,
  ["exec", "wrangler", "--", "dev", "--local", "--port", String(port), "--ip", "127.0.0.1"],
  { stdio: ["ignore", "pipe", "pipe"] },
);

let output = "";
server.stdout.on("data", (chunk) => {
  output += chunk;
});
server.stderr.on("data", (chunk) => {
  output += chunk;
});

async function waitUntilReady() {
  const deadline = Date.now() + 120_000;
  while (!output.includes(`Ready on ${baseUrl}`)) {
    if (server.exitCode !== null) {
      throw new Error(`wrangler exited before becoming ready\n${output}`);
    }
    if (Date.now() > deadline) {
      throw new Error(`wrangler did not become ready within 120 seconds\n${output}`);
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
}

async function post(path, question) {
  const response = await fetch(`${baseUrl}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ question }),
  });
  const body = await response.text();
  if (!response.ok) throw new Error(`${path} returned ${response.status}: ${body}`);
  return body;
}

try {
  await waitUntilReady();

  const predict = JSON.parse(await post("/predict", "capital of France"));
  if (predict.answer !== "Paris") throw new Error(`/predict returned ${JSON.stringify(predict)}`);

  const streamed = await post("/stream", "hello");
  if (!streamed.includes('\"type\":\"start\"') || !streamed.includes("echo: hello") || !streamed.includes('\"type\":\"end\"')) {
    throw new Error(`/stream returned an invalid event stream: ${streamed}`);
  }

  const reacted = JSON.parse(await post("/react", "lookup worker"));
  if (reacted.answer !== "value-for-worker") {
    throw new Error(`/react returned ${JSON.stringify(reacted)}`);
  }

  const cached = JSON.parse(await post("/cache-check", "same"));
  if (cached.first_cache_hit || !cached.second_cache_hit || cached.entries !== 1) {
    throw new Error(`/cache-check returned ${JSON.stringify(cached)}`);
  }

  console.log(JSON.stringify({ baseUrl, routes: ["predict", "stream", "react", "cache-check"] }));
} finally {
  server.kill("SIGTERM");
}
