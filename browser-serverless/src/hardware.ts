import type { HardwareProfile, MemoryTier, WebGlProfile, WebGpuProfile } from './types.js';

interface NavigatorWithMemory extends Navigator {
  readonly deviceMemory?: number;
}

interface PerformanceWithMemory extends Performance {
  readonly memory?: { readonly jsHeapSizeLimit?: number };
}

function featureValidate(bytes: readonly number[]): boolean {
  try {
    return typeof WebAssembly === 'object' && WebAssembly.validate(new Uint8Array(bytes));
  } catch {
    return false;
  }
}

function wasmSimdSupported(): boolean {
  return featureValidate([
    0, 97, 115, 109, 1, 0, 0, 0, 1, 4, 1, 96, 0, 0, 3,
    2, 1, 0, 10, 9, 1, 7, 0, 65, 0, 253, 15, 26, 11,
  ]);
}

function wasmThreadsSupported(sharedMemory: boolean): boolean {
  if (!sharedMemory) return false;
  try {
    new MessageChannel().port1.postMessage(new SharedArrayBuffer(1));
  } catch {
    return false;
  }
  return featureValidate([
    0, 97, 115, 109, 1, 0, 0, 0, 1, 4, 1, 96, 0, 0, 3, 2, 1, 0, 5,
    4, 1, 3, 1, 1, 10, 11, 1, 9, 0, 65, 0, 254, 16, 2, 0, 26, 11,
  ]);
}

function detectWebGl2(): WebGlProfile {
  try {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2', {
      alpha: false,
      antialias: false,
      powerPreference: 'high-performance',
    });
    if (!gl) return { available: false, renderer: null };
    const debug = gl.getExtension('WEBGL_debug_renderer_info');
    const renderer = debug
      ? String(gl.getParameter(debug.UNMASKED_RENDERER_WEBGL))
      : String(gl.getParameter(gl.RENDERER));
    return { available: true, renderer };
  } catch {
    return { available: false, renderer: null };
  }
}

async function detectWebGpu(): Promise<WebGpuProfile> {
  if (!navigator.gpu) {
    return { available: false, adapter: null, description: null, maxBufferSize: 0, maxStorageBufferBindingSize: 0 };
  }
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
      ?? await navigator.gpu.requestAdapter({ powerPreference: 'low-power' });
    if (!adapter) return { available: false, adapter: null, description: null, maxBufferSize: 0, maxStorageBufferBindingSize: 0 };
    const info = adapter.info;
    const description = [info.vendor, info.architecture, info.device, info.description]
      .map((part) => part.trim())
      .filter((part) => part.length > 0)
      .join(' · ') || 'browser GPU adapter';
    return {
      available: true,
      adapter,
      description,
      maxBufferSize: Number(adapter.limits.maxBufferSize),
      maxStorageBufferBindingSize: Number(adapter.limits.maxStorageBufferBindingSize),
    };
  } catch {
    return { available: false, adapter: null, description: null, maxBufferSize: 0, maxStorageBufferBindingSize: 0 };
  }
}

function memoryProfile(logicalCores: number): {
  readonly deviceMemoryGiB: number | null;
  readonly heapLimitMiB: number | null;
  readonly tier: MemoryTier;
  readonly budgetMiB: number;
} {
  const deviceMemory = Number((navigator as NavigatorWithMemory).deviceMemory ?? 0);
  const heapBytes = Number((performance as PerformanceWithMemory).memory?.jsHeapSizeLimit ?? 0);
  const heapLimitMiB = heapBytes > 0 ? Math.floor(heapBytes / 1048576) : null;
  const deviceMemoryGiB = deviceMemory > 0 ? deviceMemory : null;
  const mobile = /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent);
  let tier: MemoryTier = 'low';
  if (!mobile && ((deviceMemoryGiB ?? 0) >= 8 || (heapLimitMiB ?? 0) >= 3072)) tier = 'high';
  else if ((deviceMemoryGiB ?? 0) >= 4 || (heapLimitMiB ?? 0) >= 1536 || (!mobile && logicalCores >= 8)) tier = 'medium';
  const budgetMiB = deviceMemoryGiB !== null
    ? Math.floor(Math.min(2048, Math.max(256, deviceMemoryGiB * 1024 * 0.2)))
    : heapLimitMiB !== null
      ? Math.floor(Math.min(2048, Math.max(256, heapLimitMiB * 0.35)))
      : tier === 'high' ? 1280 : tier === 'medium' ? 768 : 320;
  return { deviceMemoryGiB, heapLimitMiB, tier, budgetMiB };
}

export async function detectHardware(): Promise<HardwareProfile> {
  const logicalCores = Math.max(1, navigator.hardwareConcurrency || 1);
  const memory = memoryProfile(logicalCores);
  const workerSupport = typeof Worker === 'function';
  const crossOriginIsolation = globalThis.crossOriginIsolated === true;
  const sharedMemory = crossOriginIsolation && typeof SharedArrayBuffer === 'function';
  const workerCap = memory.tier === 'high' ? 8 : memory.tier === 'medium' ? 4 : 2;
  const workerCount = workerSupport ? Math.max(1, Math.min(workerCap, logicalCores > 2 ? logicalCores - 1 : 1)) : 1;
  const webgpu = await detectWebGpu();
  return {
    webgpu,
    webgl2: detectWebGl2(),
    wasm: typeof WebAssembly === 'object',
    wasmSimd: wasmSimdSupported(),
    wasmThreads: wasmThreadsSupported(sharedMemory),
    sharedMemory,
    crossOriginIsolated: crossOriginIsolation,
    audioWorklet: typeof AudioWorkletNode === 'function' && 'audioWorklet' in AudioContext.prototype,
    workerSupport,
    logicalCores,
    deviceMemoryGiB: memory.deviceMemoryGiB,
    heapLimitMiB: memory.heapLimitMiB,
    memoryTier: memory.tier,
    memoryBudgetMiB: memory.budgetMiB,
    workerCount,
  };
}
