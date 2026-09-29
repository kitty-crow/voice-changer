import { getCachedModel } from './model-cache.js';
import { BrowserOrtRuntime } from './ort-runtime.js';
import { SharedFloatRingBuffer } from './ring-buffer.js';
import { BrowserRvcPipeline, resampleLinear } from './rvc-pipeline.js';
import type {
  ConversionSettings,
  ExecutionPlan,
  InferenceWorkerAudio,
  InferenceWorkerResponse,
  InferenceWorkerStatus,
  SharedRingDescriptor,
} from './types.js';

interface WorkerScopeLike {
  addEventListener(type: 'message', listener: (event: MessageEvent<unknown>) => void): void;
  postMessage(message: InferenceWorkerResponse, transfer?: Transferable[]): void;
}

interface ParsedBaseInit {
  readonly plan: ExecutionPlan;
  readonly inputSampleRate: number;
  readonly settings: ConversionSettings;
}

interface ParsedSharedInit extends ParsedBaseInit {
  readonly transport: 'shared';
  readonly inputRing: SharedRingDescriptor;
  readonly outputRing: SharedRingDescriptor;
}

interface ParsedMessageInit extends ParsedBaseInit {
  readonly transport: 'message';
}

type ParsedInit = ParsedSharedInit | ParsedMessageInit;

interface ParsedConvert {
  readonly id: number;
  readonly audio: Float32Array;
}

interface MessageRuntime {
  readonly pipeline: BrowserRvcPipeline;
  readonly inputSampleRate: number;
  readonly settings: ConversionSettings;
  history: Float32Array;
}

const scope = globalThis as unknown as WorkerScopeLike;
let stopped = false;
let runtime: BrowserOrtRuntime | null = null;
let messageRuntimePromise: Promise<MessageRuntime> | null = null;
let generation = 0;

function status(
  message: string,
  latencyMilliseconds: number | null = null,
  realtimeFactor: number | null = null,
): void {
  const response: InferenceWorkerStatus = {
    kind: 'status',
    message,
    latencyMilliseconds,
    realtimeFactor,
  };
  scope.postMessage(response);
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
  const transport = record['transport'];
  if (
    record['kind'] !== 'init'
    || (transport !== 'shared' && transport !== 'message')
    || typeof inputSampleRate !== 'number'
    || inputSampleRate <= 0
  ) return null;
  const parsedPlan = plan(record['plan']);
  const parsedSettings = settings(record['settings']);
  if (!parsedPlan || !parsedSettings) return null;
  if (transport === 'message') {
    return { transport, plan: parsedPlan, inputSampleRate, settings: parsedSettings };
  }
  const parsedInput = descriptor(record['inputRing']);
  const parsedOutput = descriptor(record['outputRing']);
  if (!parsedInput || !parsedOutput) return null;
  return {
    transport,
    plan: parsedPlan,
    inputRing: parsedInput,
    outputRing: parsedOutput,
    inputSampleRate,
    settings: parsedSettings,
  };
}

function parseConvert(value: unknown): ParsedConvert | null {
  if (typeof value !== 'object' || value === null) return null;
  const record = value as Record<string, unknown>;
  const id = record['id'];
  const audio = record['audio'];
  if (record['kind'] !== 'convert' || typeof id !== 'number' || !Number.isInteger(id) || !(audio instanceof Float32Array)) return null;
  return { id, audio };
}

async function loadRequiredModel(
  runtimeToLoad: BrowserOrtRuntime,
  kind: 'contentvec' | 'rmvpe' | 'rvc',
): Promise<void> {
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
    loaded.validateRvcStack();
    return loaded;
  } catch (error: unknown) {
    loaded.close();
    throw error;
  }
}

function backendSummary(activeRuntime: BrowserOrtRuntime): string {
  return activeRuntime.metadata().map((model) => `${model.kind}:${model.backend}`).join(' · ');
}

async function convertChunk(
  pipeline: BrowserRvcPipeline,
  input: Float32Array,
  history: Float32Array,
  inputSampleRate: number,
  activeSettings: ConversionSettings,
): Promise<{ readonly output: Float32Array; readonly history: Float32Array; readonly elapsed: number; readonly realtimeFactor: number }> {
  const combined = new Float32Array(history.length + input.length);
  combined.set(history);
  combined.set(input, history.length);
  const audio16k = resampleLinear(combined, inputSampleRate, 16000);
  const started = performance.now();
  const convertedAtModelRate = await pipeline.convert(audio16k, activeSettings.pitchShift, activeSettings.speakerId);
  const converted = resampleLinear(convertedAtModelRate, activeSettings.modelSampleRate, inputSampleRate);
  const desired = Math.min(input.length, converted.length);
  const output = converted.slice(converted.length - desired);
  const historySamples = Math.max(0, Math.round(inputSampleRate * 0.12));
  const nextHistory = combined.slice(Math.max(0, combined.length - historySamples));
  const elapsed = performance.now() - started;
  const chunkMilliseconds = input.length * 1000 / inputSampleRate;
  return {
    output,
    history: nextHistory,
    elapsed,
    realtimeFactor: elapsed / Math.max(chunkMilliseconds, 1),
  };
}

async function runShared(init: ParsedSharedInit, activeGeneration: number): Promise<void> {
  stopped = false;
  runtime?.close();
  const loaded = await loadStack(init.plan);
  if (activeGeneration !== generation) {
    loaded.close();
    return;
  }
  runtime = loaded;
  const pipeline = new BrowserRvcPipeline(loaded);
  const inputRing = new SharedFloatRingBuffer(init.inputRing.capacity, init.inputRing);
  const outputRing = new SharedFloatRingBuffer(init.outputRing.capacity, init.outputRing);
  const chunkSamples = Math.max(256, Math.round(init.inputSampleRate * init.settings.chunkMilliseconds / 1000));
  let history = new Float32Array(0);
  const chunk = new Float32Array(chunkSamples);
  status(`Inference worker ready · shared buffers · ${backendSummary(loaded)} · ${chunkSamples} samples/chunk.`);

  while (!stopped && activeGeneration === generation) {
    if (inputRing.availableRead() < chunkSamples || outputRing.availableWrite() < chunkSamples * 2) {
      await new Promise<void>((resolve) => setTimeout(resolve, 2));
      continue;
    }
    const read = inputRing.read(chunk);
    if (read !== chunkSamples) continue;
    try {
      const converted = await convertChunk(
        pipeline,
        chunk,
        history,
        init.inputSampleRate,
        init.settings,
      );
      history = converted.history;
      outputRing.write(converted.output);
      status(
        `RVC active · ${converted.elapsed.toFixed(1)} ms/chunk · RTF ${converted.realtimeFactor.toFixed(2)} · ${backendSummary(loaded)}`,
        converted.elapsed,
        converted.realtimeFactor,
      );
    } catch (error: unknown) {
      status(`Inference error: ${error instanceof Error ? error.message : String(error)}`);
      await new Promise<void>((resolve) => setTimeout(resolve, 25));
    }
  }
}

async function initialiseMessageRuntime(
  init: ParsedMessageInit,
  activeGeneration: number,
): Promise<MessageRuntime> {
  runtime?.close();
  const loaded = await loadStack(init.plan);
  if (activeGeneration !== generation) {
    loaded.close();
    throw new Error('Message inference initialisation was superseded.');
  }
  runtime = loaded;
  const messageRuntime: MessageRuntime = {
    pipeline: new BrowserRvcPipeline(loaded),
    inputSampleRate: init.inputSampleRate,
    settings: init.settings,
    history: new Float32Array(0),
  };
  status(`Inference worker ready · message buffers · ${backendSummary(loaded)}.`);
  return messageRuntime;
}

async function runMessageConvert(request: ParsedConvert): Promise<void> {
  const active = messageRuntimePromise;
  if (!active) {
    status('Message inference is not initialised.');
    return;
  }
  try {
    const state = await active;
    const converted = await convertChunk(
      state.pipeline,
      request.audio,
      state.history,
      state.inputSampleRate,
      state.settings,
    );
    state.history = converted.history;
    const response: InferenceWorkerAudio = {
      kind: 'audio',
      id: request.id,
      audio: converted.output,
      latencyMilliseconds: converted.elapsed,
      realtimeFactor: converted.realtimeFactor,
    };
    scope.postMessage(response, [converted.output.buffer]);
    const activeRuntime = runtime;
    if (activeRuntime) {
      status(
        `RVC active · ${converted.elapsed.toFixed(1)} ms/chunk · RTF ${converted.realtimeFactor.toFixed(2)} · ${backendSummary(activeRuntime)}`,
        converted.elapsed,
        converted.realtimeFactor,
      );
    }
  } catch (error: unknown) {
    status(`Inference error: ${error instanceof Error ? error.message : String(error)}`);
  }
}

function stop(): void {
  generation += 1;
  stopped = true;
  messageRuntimePromise = null;
  runtime?.close();
  runtime = null;
  status('Inference worker stopped.');
}

scope.addEventListener('message', (event: MessageEvent<unknown>) => {
  if (typeof event.data !== 'object' || event.data === null) {
    status('Invalid inference-worker message.');
    return;
  }
  const record = event.data as Record<string, unknown>;
  if (record['kind'] === 'stop') {
    stop();
    return;
  }
  if (record['kind'] === 'convert') {
    const request = parseConvert(event.data);
    if (!request) {
      status('Invalid message-buffer conversion request.');
      return;
    }
    void runMessageConvert(request);
    return;
  }

  const init = parseInit(event.data);
  if (!init) {
    status('Invalid inference-worker initialisation message.');
    return;
  }
  generation += 1;
  const activeGeneration = generation;
  if (init.transport === 'shared') {
    messageRuntimePromise = null;
    void runShared(init, activeGeneration).catch((error: unknown) => {
      status(`Worker startup failed: ${error instanceof Error ? error.message : String(error)}`);
    });
  } else {
    stopped = false;
    messageRuntimePromise = initialiseMessageRuntime(init, activeGeneration);
    void messageRuntimePromise.catch((error: unknown) => {
      status(`Worker startup failed: ${error instanceof Error ? error.message : String(error)}`);
    });
  }
});
