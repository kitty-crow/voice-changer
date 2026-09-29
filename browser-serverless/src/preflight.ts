import { detectHardware } from './hardware.js';
import type { BenchmarkRequest, BenchmarkResponse, ExecutionPlan, HardwareProfile, NeuralBackend, PreflightBenchmark, PreflightResult } from './types.js';

function benchmarkWorker(profile: HardwareProfile): Promise<PreflightBenchmark> {
  if (!profile.workerSupport) {
    return Promise.resolve({ backend: 'wasm', milliseconds: Number.POSITIVE_INFINITY, passed: false, detail: 'Web Workers unavailable.' });
  }

  const workerTotal = profile.workerCount;
  const started = performance.now();
  return new Promise((resolve) => {
    let completed = 0;
    let failed = false;
    let checksum = 0;
    const workers: Worker[] = [];
    const finish = (detail: string): void => {
      for (const worker of workers) worker.terminate();
      resolve({ backend: 'wasm', milliseconds: performance.now() - started, passed: !failed, detail });
    };

    for (let index = 0; index < workerTotal; index += 1) {
      const worker = new Worker(new URL('./benchmark-worker.js', import.meta.url), { type: 'module' });
      workers.push(worker);
      worker.addEventListener('message', (event: MessageEvent<unknown>) => {
        if (typeof event.data !== 'object' || event.data === null) {
          failed = true;
          finish('Worker returned an invalid response.');
          return;
        }
        const record = event.data as Record<string, unknown>;
        if (typeof record['id'] !== 'number' || typeof record['elapsedMilliseconds'] !== 'number' || typeof record['checksum'] !== 'number' || (record['error'] !== null && typeof record['error'] !== 'string')) {
          failed = true;
          finish('Worker returned an invalid response shape.');
          return;
        }
        const response: BenchmarkResponse = {
          id: record['id'],
          elapsedMilliseconds: record['elapsedMilliseconds'],
          checksum: record['checksum'],
          error: record['error'] as string | null,
        };
        if (response.error !== null) failed = true;
        checksum += response.checksum;
        completed += 1;
        if (completed === workerTotal) {
          finish(failed ? 'At least one CPU worker failed.' : `${workerTotal} worker(s), checksum ${checksum.toFixed(1)}`);
        }
      });
      const request: BenchmarkRequest = { id: index, iterations: 48, sampleCount: 32768 };
      worker.postMessage(request);
    }
  });
}

async function benchmarkWebGpu(profile: HardwareProfile): Promise<PreflightBenchmark> {
  const adapter = profile.webgpu.adapter;
  if (!profile.webgpu.available || adapter === null) {
    return { backend: 'webgpu', milliseconds: Number.POSITIVE_INFINITY, passed: false, detail: 'WebGPU unavailable.' };
  }

  let device: GPUDevice | null = null;
  try {
    device = await adapter.requestDevice();
    const elementCount = 262144;
    const byteLength = elementCount * Float32Array.BYTES_PER_ELEMENT;
    if (byteLength > profile.webgpu.maxStorageBufferBindingSize) {
      return { backend: 'webgpu', milliseconds: Number.POSITIVE_INFINITY, passed: false, detail: 'GPU storage-buffer limit is too small.' };
    }
    const buffer = device.createBuffer({
      size: byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({
          code: `
            @group(0) @binding(0) var<storage, read_write> values: array<f32>;
            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
              let index = id.x;
              if (index >= arrayLength(&values)) { return; }
              var value = values[index];
              for (var i = 0u; i < 128u; i = i + 1u) {
                value = value * 1.000061 + 0.000122;
              }
              values[index] = value;
            }
          `,
        }),
        entryPoint: 'main',
      },
    });
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer } }],
    });
    const seed = new Float32Array(elementCount);
    device.queue.writeBuffer(buffer, 0, seed);

    const encode = (): void => {
      const encoder = device?.createCommandEncoder();
      if (!encoder || !device) throw new Error('WebGPU device disappeared during benchmark.');
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(elementCount / 64));
      pass.end();
      device.queue.submit([encoder.finish()]);
    };

    encode();
    await device.queue.onSubmittedWorkDone();
    const started = performance.now();
    for (let pass = 0; pass < 4; pass += 1) encode();
    await device.queue.onSubmittedWorkDone();
    const elapsed = performance.now() - started;
    buffer.destroy();
    return { backend: 'webgpu', milliseconds: elapsed, passed: Number.isFinite(elapsed), detail: 'WebGPU compute kernel completed.' };
  } catch (error: unknown) {
    return {
      backend: 'webgpu',
      milliseconds: Number.POSITIVE_INFINITY,
      passed: false,
      detail: error instanceof Error ? error.message : String(error),
    };
  } finally {
    device?.destroy();
  }
}

function choosePlan(profile: HardwareProfile, benchmarks: readonly PreflightBenchmark[]): ExecutionPlan {
  const gpu = benchmarks.find((result) => result.backend === 'webgpu');
  const cpu = benchmarks.find((result) => result.backend === 'wasm');

  let neuralBackend: NeuralBackend = 'wasm';
  if (gpu?.passed === true) {
    const gpuHealthy = gpu.milliseconds < 750;
    const cpuUnhealthy = cpu?.passed !== true;
    if (gpuHealthy || cpuUnhealthy) neuralBackend = 'webgpu';
  }

  return {
    neuralBackend,
    wasmThreads: profile.sharedMemory && profile.workerCount > 1,
    workerCount: profile.workerCount,
    sharedAudioBuffers: profile.sharedMemory && profile.audioWorklet,
  };
}

export async function runPreflight(): Promise<PreflightResult> {
  const profile = await detectHardware();
  const benchmarks = await Promise.all([
    benchmarkWebGpu(profile),
    benchmarkWorker(profile),
  ]);
  return {
    profile,
    plan: choosePlan(profile, benchmarks),
    benchmarks,
    completedAt: Date.now(),
  };
}
