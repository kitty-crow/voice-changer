# Serverless browser voice changer

This directory is the browser-native, zero-inference-server runtime for Voice Changer.

The production target is a static GitHub Pages site. The host serves files only. Microphone audio, feature extraction, pitch extraction, RVC inference and playback remain on the visiting device.

## Runtime architecture

```text
Microphone
  -> AudioWorklet
  -> SharedArrayBuffer input ring
  -> dedicated inference Worker
     -> resample to 16 kHz
     -> ContentVec / HuBERT ONNX
     -> RMVPE ONNX when the RVC model requires F0
     -> RVC ONNX
     -> resample to browser audio rate
  -> SharedArrayBuffer output ring
  -> AudioWorklet
  -> speakers
```

There is no WebSocket, Socket.IO, localhost server or inference API in the browser runtime.

## Automatic hardware selection

The page runs a hardware pre-flight when it loads. The user is not asked to choose a backend.

The pre-flight detects and tests:

- WebGPU and the browser-exposed GPU adapter;
- WebGL2 as the older GPU compatibility tier;
- WebAssembly;
- WebAssembly SIMD;
- WebAssembly threads;
- `SharedArrayBuffer` / cross-origin isolation;
- logical CPU concurrency;
- browser-visible memory information and a conservative working-set budget;
- Web Workers; and
- AudioWorklet.

The neural execution preference is selected automatically:

```text
WebGPU -> WebGL -> WASM
```

ONNX Runtime is also allowed to fall back to WASM when a particular model cannot create a session on the selected GPU execution provider. The loaded-model display reports the provider actually selected for each model.

The footer reports the detected resources, chosen path and pre-flight timings. There is deliberately no backend selector in the UI.

## Models

The browser runtime accepts ONNX files from the user and caches them in IndexedDB:

- ContentVec / HuBERT;
- RMVPE, when required; and
- RVC.

The existing server exporter defines two supported RVC contracts:

F0 model:

```text
feats, p_len, pitch, pitchf, sid -> audio
```

F0-less model:

```text
feats, p_len, sid -> audio
```

The browser inspects the actual RVC inputs, so RMVPE is not loaded or executed for F0-less models.

RVC v1 commonly expects 256-channel projected ContentVec features, while RVC v2 commonly expects 768-channel hidden features. Static ONNX tensor metadata is checked before audio starts and an incompatible encoder/RVC pair is rejected with a clear error.

## TypeScript source policy

All handwritten browser runtime source is TypeScript. JavaScript exists only as generated build output.

Two strict compiler environments are used because AudioWorklet globals intentionally conflict with DOM globals:

- `tsconfig.json` for the window and Worker environments;
- `tsconfig.worklet.json` for `AudioWorkletGlobalScope`.

Both configurations require strict type checking with `skipLibCheck: false`. Regression tests reject explicit `any`, `@ts-ignore`, `@ts-nocheck`, handwritten JavaScript under `src/`, and weakening of the required strict compiler options.

## Serverless contract

`tests/serverless-contract.test.ts` rejects browser-source dependencies on:

- WebSocket;
- XMLHttpRequest;
- Socket.IO;
- localhost / 127.0.0.1;
- `/api/` inference routes;
- MMVCServer; and
- direct `fetch()` calls from the inference application source.

The cross-origin-isolation service worker is the only intentional fetch relay. It adds the headers required for `SharedArrayBuffer` on static hosting.

## Build

```bash
cd browser-serverless
bun install
bun run test:strict
bun run build
bun run check:dist
```

The resulting `dist/` directory is static and contains the generated JavaScript bundles, HTML/CSS and ONNX Runtime WASM assets.

## Current scope

Implemented on this branch:

- static browser application shell;
- automatic hardware pre-flight and footer;
- WebGPU, WebGL and WASM ONNX Runtime execution tiers;
- strict TypeScript-only source policy;
- IndexedDB model cache;
- AudioWorklet audio I/O;
- SharedArrayBuffer audio rings;
- dedicated inference Worker;
- ContentVec / HuBERT inference;
- optional RMVPE inference;
- RVC F0 and F0-less ONNX contracts;
- pitch shifting and speaker ID;
- local resampling; and
- per-chunk latency / real-time-factor reporting.

Still to be completed and validated before replacing the existing local-server path:

- browser smoke tests with real exported voice models;
- quality tuning for overlap/crossfade and chunk boundaries;
- real-model backend autotuning after models are loaded;
- message-based inference when cross-origin isolation is unavailable; and
- optional retrieval-index support to replace the Python FAISS stage.

The existing Python/local-server implementation remains untouched while this work is developed on `feat/serverless-browser-runtime`.
