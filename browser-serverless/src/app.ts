import { LocalAudioPath } from './audio.js';
import { clearModelCache, getCachedModel, putCachedModel } from './model-cache.js';
import { BrowserOrtRuntime } from './ort-runtime.js';
import { runPreflight } from './preflight.js';
import type { ConversionSettings, ModelKind, ModelMetadata, PreflightResult } from './types.js';

function element<T extends HTMLElement>(id: string, constructor: { new (): T }): T {
  const found = document.getElementById(id);
  if (!(found instanceof constructor)) throw new Error(`Missing or invalid #${id}.`);
  return found;
}

function numberValue(id: string, fallback: number): number {
  const value = Number(element(id, HTMLInputElement).value);
  return Number.isFinite(value) ? value : fallback;
}

const preflightStatus = element('preflight-status', HTMLParagraphElement);
const resourceFooter = element('resource-footer', HTMLElement);
const loadModels = element('load-models', HTMLButtonElement);
const clearCache = element('clear-model-cache', HTMLButtonElement);
const modelStatus = element('model-status', HTMLPreElement);
const startAudio = element('start-audio', HTMLButtonElement);
const stopAudio = element('stop-audio', HTMLButtonElement);
const audioStatus = element('audio-status', HTMLParagraphElement);
const inputs: Readonly<Record<ModelKind, HTMLInputElement>> = {
  contentvec: element('contentvec-model', HTMLInputElement),
  rmvpe: element('rmvpe-model', HTMLInputElement),
  rvc: element('rvc-model', HTMLInputElement),
};

let preflight: PreflightResult | null = null;
let runtime: BrowserOrtRuntime | null = null;
let audioPath: LocalAudioPath | null = null;

function conversionSettings(): ConversionSettings {
  const modelRate = Number(element('model-sample-rate', HTMLSelectElement).value);
  const chunkMilliseconds = Number(element('chunk-ms', HTMLSelectElement).value);
  return {
    pitchShift: Math.max(-24, Math.min(24, Math.round(numberValue('pitch-shift', 0)))),
    speakerId: Math.max(0, Math.round(numberValue('speaker-id', 0))),
    modelSampleRate: [32000, 40000, 48000].includes(modelRate) ? modelRate : 40000,
    chunkMilliseconds: [160, 240, 320, 480].includes(chunkMilliseconds) ? chunkMilliseconds : 320,
  };
}

function benchmarkText(result: PreflightResult): string {
  return result.benchmarks
    .map((benchmark) => `${benchmark.backend}: ${benchmark.passed ? `${benchmark.milliseconds.toFixed(1)} ms` : 'unavailable'} (${benchmark.detail})`)
    .join(' · ');
}

function populateFooter(result: PreflightResult): void {
  const profile = result.profile;
  const gpuResources: string[] = [];
  if (profile.webgpu.available) gpuResources.push(`WebGPU ${profile.webgpu.description ?? 'adapter'}`);
  if (profile.webgl2.available) gpuResources.push(`WebGL2 ${profile.webgl2.renderer ?? 'adapter'}`);
  if (gpuResources.length === 0) gpuResources.push('no browser GPU');
  const memory = profile.deviceMemoryGiB !== null
    ? `~${profile.deviceMemoryGiB} GiB device memory`
    : profile.heapLimitMiB !== null
      ? `${profile.heapLimitMiB} MiB JS heap limit`
      : 'memory estimate unavailable';

  resourceFooter.replaceChildren();
  const selected = document.createElement('strong');
  selected.textContent = `Selected automatically: ${result.plan.neuralBackend.toUpperCase()} · ${result.plan.workerCount} CPU worker(s)`;
  const resources = document.createElement('span');
  resources.textContent = `${gpuResources.join(' · ')} · ${profile.logicalCores} logical CPU cores · ${memory} · ${profile.memoryTier} memory · WASM ${profile.wasmSimd ? 'SIMD' : 'scalar'}${profile.wasmThreads ? '+threads' : ''} · ${result.plan.sharedAudioBuffers ? 'SharedArrayBuffer audio' : 'message audio'}`;
  const benchmarks = document.createElement('span');
  benchmarks.textContent = `Pre-flight: ${benchmarkText(result)}`;
  resourceFooter.append(selected, resources, benchmarks);
}

async function registerIsolationWorker(): Promise<void> {
  if (!('serviceWorker' in navigator) || !location.protocol.startsWith('http')) return;
  try {
    await navigator.serviceWorker.register('./coi-serviceworker.js', { scope: './' });
    if (!crossOriginIsolated && !navigator.serviceWorker.controller && sessionStorage.getItem('vc-coi-reload') !== '1') {
      sessionStorage.setItem('vc-coi-reload', '1');
      await navigator.serviceWorker.ready;
      location.reload();
    } else if (crossOriginIsolated) {
      sessionStorage.removeItem('vc-coi-reload');
    }
  } catch (error: unknown) {
    console.warn('Cross-origin isolation service worker unavailable.', error);
  }
}

function metadataText(metadata: readonly ModelMetadata[], validation: readonly string[] = []): string {
  if (metadata.length === 0) return 'No browser models loaded.';
  const modelText = metadata.map((model) => [
    `${model.kind}: ${model.name}`,
    `  ${(model.sizeBytes / 1048576).toFixed(1)} MiB · ${model.backend.toUpperCase()}`,
    `  inputs: ${model.inputs.join(', ') || '(none)'}`,
    `  outputs: ${model.outputs.join(', ') || '(none)'}`,
  ].join('\n')).join('\n');
  return validation.length > 0 ? `${modelText}\n\nValidated stack: ${validation.join(' · ')}` : modelText;
}

async function modelBytes(kind: ModelKind): Promise<{ readonly name: string; readonly bytes: ArrayBuffer } | null> {
  const file = inputs[kind].files?.[0];
  if (file) return { name: file.name, bytes: await putCachedModel(kind, file) };
  const cached = await getCachedModel(kind);
  return cached ? { name: cached.name, bytes: cached.bytes } : null;
}

async function loadSelectedModels(): Promise<void> {
  const activePreflight = preflight;
  if (!activePreflight) return;
  runtime?.close();
  const nextRuntime = new BrowserOrtRuntime(activePreflight.plan);
  runtime = nextRuntime;
  const loaded: ModelMetadata[] = [];
  try {
    for (const kind of ['contentvec', 'rmvpe', 'rvc'] as const) {
      const model = await modelBytes(kind);
      if (!model) continue;
      modelStatus.textContent = `Loading ${kind} locally…`;
      loaded.push(await nextRuntime.load(kind, model.name, model.bytes));
    }
    const validation = nextRuntime.hasCompleteRvcStack() ? nextRuntime.validateRvcStack() : [];
    modelStatus.textContent = metadataText(loaded, validation);
  } catch (error: unknown) {
    modelStatus.textContent = `${metadataText(loaded)}\n\nCompatibility error: ${error instanceof Error ? error.message : String(error)}`;
    throw error;
  }
}

loadModels.addEventListener('click', () => {
  loadModels.disabled = true;
  void loadSelectedModels()
    .catch(() => undefined)
    .finally(() => { loadModels.disabled = false; });
});

clearCache.addEventListener('click', () => {
  clearCache.disabled = true;
  void clearModelCache()
    .then(() => {
      runtime?.close();
      runtime = preflight ? new BrowserOrtRuntime(preflight.plan) : null;
      modelStatus.textContent = 'Local model cache cleared.';
    })
    .catch((error: unknown) => { modelStatus.textContent = error instanceof Error ? error.message : String(error); })
    .finally(() => { clearCache.disabled = false; });
});

startAudio.addEventListener('click', () => {
  const result = preflight;
  const activeRuntime = runtime;
  if (!result) return;
  startAudio.disabled = true;

  let convert = false;
  try {
    if (activeRuntime?.hasCompleteRvcStack() === true) {
      activeRuntime.validateRvcStack();
      convert = true;
    }
  } catch (error: unknown) {
    audioStatus.textContent = `Model compatibility error: ${error instanceof Error ? error.message : String(error)}`;
    startAudio.disabled = false;
    return;
  }

  audioPath = new LocalAudioPath(result.plan, conversionSettings());
  void audioPath.start(convert, (workerStatus) => { audioStatus.textContent = workerStatus.message; })
    .then((detail) => {
      audioStatus.textContent = `${convert ? 'Local conversion requested.' : 'Models incomplete, using local pass-through.'} ${detail}`;
      stopAudio.disabled = false;
    })
    .catch((error: unknown) => {
      audioStatus.textContent = error instanceof Error ? error.message : String(error);
      startAudio.disabled = false;
    });
});

stopAudio.addEventListener('click', () => {
  stopAudio.disabled = true;
  const active = audioPath;
  audioPath = null;
  void (active?.stop() ?? Promise.resolve()).finally(() => {
    audioStatus.textContent = 'Audio stopped.';
    startAudio.disabled = false;
  });
});

void (async (): Promise<void> => {
  await registerIsolationWorker();
  preflightStatus.textContent = 'Benchmarking browser hardware automatically…';
  try {
    preflight = await runPreflight();
    populateFooter(preflight);
    preflightStatus.textContent = `Ready. Neural inference will use ${preflight.plan.neuralBackend.toUpperCase()} automatically.`;
    runtime = new BrowserOrtRuntime(preflight.plan);
  } catch (error: unknown) {
    preflightStatus.textContent = error instanceof Error ? error.message : String(error);
  }
})();
