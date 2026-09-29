import { detectHardware } from './hardware.js';
import type { BenchmarkRequest, BenchmarkResponse, ExecutionPlan, HardwareProfile, NeuralBackend, PreflightBenchmark, PreflightResult } from './types.js';

function benchmarkWorker(profile: HardwareProfile): Promise<PreflightBenchmark> {
  if (!profile.workerSupport || !profile.wasm) return Promise.resolve({ backend: 'wasm', milliseconds: Number.POSITIVE_INFINITY, passed: false, detail: 'WASM/Workers unavailable.' });
  const workerTotal = profile.workerCount;
  const started = performance.now();
  return new Promise((resolve) => {
    let completed = 0;
    let failed = false;
    let checksum = 0;
    let settled = false;
    const workers: Worker[] = [];
    const finish = (detail: string): void => {
      if (settled) return;
      settled = true;
      for (const worker of workers) worker.terminate();
      resolve({ backend: 'wasm', milliseconds: performance.now() - started, passed: !failed, detail });
    };
    for (let index = 0; index < workerTotal; index += 1) {
      const worker = new Worker(new URL('./benchmark-worker.js', import.meta.url), { type: 'module' });
      workers.push(worker);
      worker.addEventListener('message', (event: MessageEvent<unknown>) => {
        if (settled) return;
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
        if (completed === workerTotal) finish(failed ? 'At least one CPU worker failed.' : `${workerTotal} worker(s), checksum ${checksum.toFixed(1)}`);
      });
      const request: BenchmarkRequest = { id: index, iterations: 48, sampleCount: 32768 };
      worker.postMessage(request);
    }
  });
}

async function benchmarkWebGpu(profile: HardwareProfile): Promise<PreflightBenchmark> {
  const adapter = profile.webgpu.adapter;
  if (!profile.webgpu.available || adapter === null) return { backend: 'webgpu', milliseconds: Number.POSITIVE_INFINITY, passed: false, detail: 'WebGPU unavailable.' };
  let device: GPUDevice | null = null;
  try {
    device = await adapter.requestDevice();
    const elementCount = 262144;
    const byteLength = elementCount * Float32Array.BYTES_PER_ELEMENT;
    if (byteLength > profile.webgpu.maxStorageBufferBindingSize) return { backend: 'webgpu', milliseconds: Number.POSITIVE_INFINITY, passed: false, detail: 'GPU storage-buffer limit is too small.' };
    const buffer = device.createBuffer({ size: byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({ code: `
          @group(0) @binding(0) var<storage, read_write> values: array<f32>;
          @compute @workgroup_size(64)
          fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            let index = id.x;
            if (index >= arrayLength(&values)) { return; }
            var value = values[index];
            for (var i = 0u; i < 128u; i = i + 1u) { value = value * 1.000061 + 0.000122; }
            values[index] = value;
          }
        ` }),
        entryPoint: 'main',
      },
    });
    const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer } }] });
    device.queue.writeBuffer(buffer, 0, new Float32Array(elementCount));
    const encode = (): void => {
      if (!device) throw new Error('WebGPU device disappeared during benchmark.');
      const encoder = device.createCommandEncoder();
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
    return { backend: 'webgpu', milliseconds: Number.POSITIVE_INFINITY, passed: false, detail: error instanceof Error ? error.message : String(error) };
  } finally {
    device?.destroy();
  }
}

function shader(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
  const result = gl.createShader(type);
  if (!result) throw new Error('WebGL2 could not allocate a shader.');
  gl.shaderSource(result, source);
  gl.compileShader(result);
  if (!gl.getShaderParameter(result, gl.COMPILE_STATUS)) {
    const message = gl.getShaderInfoLog(result) ?? 'WebGL2 shader compilation failed.';
    gl.deleteShader(result);
    throw new Error(message);
  }
  return result;
}

function benchmarkWebGl(profile: HardwareProfile): PreflightBenchmark {
  if (!profile.webgl2.available) return { backend: 'webgl', milliseconds: Number.POSITIVE_INFINITY, passed: false, detail: 'WebGL2 unavailable.' };
  try {
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 256;
    const gl = canvas.getContext('webgl2', { alpha: false, antialias: false, powerPreference: 'high-performance' });
    if (!gl) throw new Error('WebGL2 context creation failed.');
    const vertex = shader(gl, gl.VERTEX_SHADER, `#version 300 es
      precision highp float;
      void main() {
        vec2 p = gl_VertexID == 0 ? vec2(-1.0,-1.0) : gl_VertexID == 1 ? vec2(3.0,-1.0) : vec2(-1.0,3.0);
        gl_Position = vec4(p,0.0,1.0);
      }`);
    const fragment = shader(gl, gl.FRAGMENT_SHADER, `#version 300 es
      precision highp float;
      out vec4 colour;
      void main() {
        float v = gl_FragCoord.x * 0.001 + gl_FragCoord.y * 0.002;
        for (int i = 0; i < 96; ++i) { v = fract(v * 1.00061 + 0.000122); }
        colour = vec4(v, v * 0.7, v * 0.3, 1.0);
      }`);
    const program = gl.createProgram();
    if (!program) throw new Error('WebGL2 could not allocate a program.');
    gl.attachShader(program, vertex);
    gl.attachShader(program, fragment);
    gl.linkProgram(program);
    gl.deleteShader(vertex);
    gl.deleteShader(fragment);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program) ?? 'WebGL2 program link failed.');
    gl.useProgram(program);
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.finish();
    const started = performance.now();
    for (let pass = 0; pass < 32; pass += 1) gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.finish();
    const elapsed = performance.now() - started;
    gl.deleteProgram(program);
    return { backend: 'webgl', milliseconds: elapsed, passed: Number.isFinite(elapsed), detail: 'WebGL2 shader workload completed.' };
  } catch (error: unknown) {
    return { backend: 'webgl', milliseconds: Number.POSITIVE_INFINITY, passed: false, detail: error instanceof Error ? error.message : String(error) };
  }
}

function choosePlan(profile: HardwareProfile, benchmarks: readonly PreflightBenchmark[]): ExecutionPlan {
  const gpu = benchmarks.find((result) => result.backend === 'webgpu');
  const webgl = benchmarks.find((result) => result.backend === 'webgl');
  const cpu = benchmarks.find((result) => result.backend === 'wasm');
  let neuralBackend: NeuralBackend = 'wasm';
  if (gpu?.passed === true && (gpu.milliseconds < 750 || cpu?.passed !== true)) neuralBackend = 'webgpu';
  else if (webgl?.passed === true) neuralBackend = 'webgl';
  return {
    neuralBackend,
    wasmThreads: profile.wasmThreads && profile.workerCount > 1,
    workerCount: profile.workerCount,
    sharedAudioBuffers: profile.sharedMemory && profile.audioWorklet,
  };
}

export async function runPreflight(): Promise<PreflightResult> {
  const profile = await detectHardware();
  const benchmarks = await Promise.all([benchmarkWebGpu(profile), Promise.resolve(benchmarkWebGl(profile)), benchmarkWorker(profile)]);
  return { profile, plan: choosePlan(profile, benchmarks), benchmarks, completedAt: Date.now() };
}
