import { getCachedModel } from './model-cache.js';
import { BrowserOrtRuntime } from './ort-runtime.js';
import { SharedFloatRingBuffer } from './ring-buffer.js';
import { BrowserRvcPipeline, resampleLinear } from './rvc-pipeline.js';
import type { ConversionSettings, ExecutionPlan, InferenceWorkerStatus, SharedRingDescriptor } from './types.js';

interface WorkerScopeLike {
  addEventListener(type: 'message', listener: (event: MessageEvent<unknown>) => void): void;
  postMessage(message: InferenceWorkerStatus): void;
}

interface ParsedInit {
  readonly plan: ExecutionPlan;
  readonly inputRing: SharedRingDescriptor;
  readonly outputRing: SharedRingDescriptor;
  readonly inputSampleRate: number;
  readonly settings: ConversionSettings;
}

const scope = globalThis as unknown as WorkerScopeLike;
let stopped = false;
let runtime: BrowserOrtRuntime | null = null;

function status(
  message: string,
  latencyMilliseconds: number | null = null,
  realtimeFactor: number | null = null,
): void {
  scope.postMessage({ kind: 'status', message, latencyMilliseconds, realtimeFactor });
}

function descriptor(value: unknown): SharedRingDescriptor | null {
  if (typeof value !== 'object' || value === null) return null;
  const record = value as Record<string, unknown>;
  const header = record['header'];
  const samples = record['samples'];
  const capacity = record['capacity'];
  if (
    !(header instanceof SharedArrayBuffer)
    || !(samples instanceof SharedArrayBuffer)
    || typeof capacity !== 'number'
    || !Number.isInteger(capacity)
    || capacity < 2
  ) return null;
  return { header, samples, capacity };
}

function plan(value: unknown): ExecutionPlan | null {
  if (typeof value !== 'object' || value === null) return null;
  const record = value as Record<string, unknown>;
  const backend = record['neuralBackend'];
  const wasmThreads = record['wasmThreads'];
  const workerCount = record['workerCount'];
  const sharedAudioBuffers = record['sharedAudioBuffers'];
  if (
    (backend !== 'webgpu' && backend !== 'webgl' && backend !== 'wasm')
    || typeof wasmThreads !== 'boolean'
    || typeof workerCount !== 'number'
    || !Number.isInteger(workerCount)
    || workerCount < 1
    || typeof sharedAudioBuffers !== 'boolean'
  ) return null;
  return { neuralBackend: backend, wasmThreads, workerCount, sharedAudioBuffers };
}

function settings(value: unknown): ConversionSettings | null {
  if (typeof value !== 'object' || value === null) return null;
  const record = value as Record<string, unknown>;
  const pitchShift = record['pitchShift'];
  const speakerId = record['speakerId'];
  const modelSampleRate = record['modelSampleRate'];
  const chunkMilliseconds = record['chunkMilliseconds'];
  if (
    typeof pitchShift !== 'number'
    || typeof speakerId !== 'number'
    || typeof modelSampleRate !== 'number'
    || typeof chunkMilliseconds !== 'number'
  ) return null;
  return { pitchShift, speakerId, modelSampleRate, chunkMilliseconds };
}

function parseInit(value: unknown): ParsedInit | null {
  if (typeof value !== 'object' || value === null) return null;
  const record = value as Record<string, unknown>;
  const inputSampleRate = record['inputSampleRate'];
  if (record['kind'] !== 'init' || typeof inputSampleRate !== 'number' || inputSampleRate <= 0) return null;
  const parsedPlan = plan(record['plan']);
  const parsedInput = descriptor(record['inputRing']);
  const parsedOutput = descriptor(record['outputRing']);
  const parsedSettings = settings(record['settings']);
  if (!parsedPlan || !parsedInput || !parsedOutput || !parsedSettings) return null;
  return {
    plan: parsedPlan,
    inputRing: parsedInput,
    outputRing: parsedOutput,
    inputSampleRate,
    settings: parsedSettings,
  };
}

async function loadRequiredModel(runtimeToLoad: BrowserOrtRuntime, kind: 'contentvec' | 'rmvpe' | 'rvc'): Promise<void> {
  const cached = await getCachedModel(kind);
  if (!cached) throw new Error(`${kind} model is not cached in this browser.`);
  await runtimeToLoad.load(kind, cached.name, cached.bytes);
}

async function loadStack(activePlan: ExecutionPlan): Promise<BrowserOrtRuntime> {
  const loaded = new BrowserOrtRuntime(activePlan);
  try {
    await loadRequiredModel(loaded, 'contentvec');
    await loadRequiredModel(loaded, 'rvc');
    if (loaded.rvcNeedsPitch()) await loadRequiredModel(loaded, 'rmvpe');
    return loaded;
  } catch (error: unknown) {
    loaded.close();
    throw error;
  }
}

function backendSummary(activeRuntime: BrowserOrtRuntime): string {
  return activeRuntime.metadata().map((model) => `${model.kind}:${model.backend}`).join(' · ');
}

async function run(init: ParsedInit): Promise<void> {
  stopped = false;
  runtime?.close();
  runtime = await loadStack(init.plan);
  const pipeline = new BrowserRvcPipeline(runtime);
  const inputRing = new SharedFloatRingBuffer(init.inputRing.capacity, init.inputRing);
  const outputRing = new SharedFloatRingBuffer(init.outputRing.capacity, init.outputRing);
  const chunkSamples = Math.max(256, Math.round(init.inputSampleRate * init.settings.chunkMilliseconds / 1000));
  const historySamples = Math.max(0, Math.round(init.inputSampleRate * 0.12));
  let history = new Float32Array(0);
  const chunk = new Float32Array(chunkSamples);
  status(`Inference worker ready · ${backendSummary(runtime)} · ${chunkSamples} input samples/chunk.`);

  while (!stopped) {
    if (inputRing.availableRead() < chunkSamples || outputRing.availableWrite() < chunkSamples * 2) {
      await new Promise<void>((resolve) => setTimeout(resolve, 2));
      continue;
    }
    const read = inputRing.read(chunk);
    if (read !== chunkSamples) continue;
    const combined = new Float32Array(history.length + chunk.length);
    combined.set(history);
    combined.set(chunk, history.length);
    const audio16k = resampleLinear(combined, init.inputSampleRate, 16000);
    const started = performance.now();
    try {
      const convertedAtModelRate = await pipeline.convert(
        audio16k,
        init.settings.pitchShift,
        init.settings.speakerId,
      );
      const converted = resampleLinear(
        convertedAtModelRate,
        init.settings.modelSampleRate,
        init.inputSampleRate,
      );
      const desired = Math.min(chunkSamples, converted.length);
      const newest = converted.subarray(converted.length - desired);
      outputRing.write(newest);
      const elapsed = performance.now() - started;
      const realtimeFactor = elapsed / init.settings.chunkMilliseconds;
      status(
        `RVC active · ${elapsed.toFixed(1)} ms/chunk · RTF ${realtimeFactor.toFixed(2)} · ${backendSummary(runtime)}`,
        elapsed,
        realtimeFactor,
      );
    } catch (error: unknown) {
      status(`Inference error: ${error instanceof Error ? error.message : String(error)}`);
      await new Promise<void>((resolve) => setTimeout(resolve, 25));
    }
    history = combined.slice(Math.max(0, combined.length - historySamples));
  }
}

scope.addEventListener('message', (event: MessageEvent<unknown>) => {
  if (
    typeof event.data === 'object'
    && event.data !== null
    && (event.data as Record<string, unknown>)['kind'] === 'stop'
  ) {
    stopped = true;
    runtime?.close();
    runtime = null;
    status('Inference worker stopped.');
    return;
  }
  const init = parseInit(event.data);
  if (!init) {
    status('Invalid inference-worker initialisation message.');
    return;
  }
  void run(init).catch((error: unknown) => {
    status(`Worker startup failed: ${error instanceof Error ? error.message : String(error)}`);
  });
});
