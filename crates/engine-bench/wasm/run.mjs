import { readFile } from "node:fs/promises";
import { WASI } from "node:wasi";

const wasmPath = process.argv[2];
if (!wasmPath) {
  throw new Error("usage: node run.mjs <ferment_engine_bench.wasm>");
}

const bytes = await readFile(wasmPath);
const wasi = new WASI({ version: "preview1", args: [], env: {} });
const { instance } = await WebAssembly.instantiate(bytes, {
  wasi_snapshot_preview1: wasi.wasiImport,
});
wasi.start(instance);
