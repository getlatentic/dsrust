import { readFile, stat } from "node:fs/promises";

const workerPath = new URL("../build/index_bg.wasm", import.meta.url);
const baselinePath = new URL("../size-baseline.json", import.meta.url);
const maximumBytes = 64 * 1024 * 1024;
const maximumGrowth = 1.05;
const { size } = await stat(workerPath);
const baseline = JSON.parse(await readFile(baselinePath, "utf8"));
const growthLimit = Math.floor(baseline.inference.bytes * maximumGrowth);

console.log(
  JSON.stringify({
    capability: "inference",
    artifact: "index_bg.wasm",
    bytes: size,
    baselineBytes: baseline.inference.bytes,
    growthLimit,
    maximumBytes,
  }),
);
if (size > maximumBytes) {
  throw new Error(`Worker WASM is ${size} bytes; limit is ${maximumBytes} bytes`);
}
if (size > growthLimit) {
  throw new Error(
    `Worker WASM is ${size} bytes; the committed 5% growth limit is ${growthLimit} bytes`,
  );
}
