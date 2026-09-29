import type { BenchmarkRequest, BenchmarkResponse } from './types.js';

interface WorkerScope {
  addEventListener(type: 'message', listener: (event: MessageEvent<unknown>) => void): void;
  postMessage(message: BenchmarkResponse): void;
}

const scope = globalThis as unknown as WorkerScope;

function isRequest(value: unknown): value is BenchmarkRequest {
  if (typeof value !== 'object' || value === null) return false;
  const record = value as Record<string, unknown>;
  return typeof record['id'] === 'number'
    && typeof record['iterations'] === 'number'
    && typeof record['sampleCount'] === 'number';
}

scope.addEventListener('message', (event: MessageEvent<unknown>) => {
  if (!isRequest(event.data)) {
    scope.postMessage({ id: -1, elapsedMilliseconds: 0, checksum: 0, error: 'Invalid benchmark request.' });
    return;
  }
  const request = event.data;
  try {
    const values = new Float32Array(request.sampleCount);
    for (let index = 0; index < values.length; index += 1) values[index] = (index % 251) / 251;
    const started = performance.now();
    let checksum = 0;
    for (let pass = 0; pass < request.iterations; pass += 1) {
      for (let index = 0; index < values.length; index += 1) {
        const value = values[index];
        if (value === undefined) throw new Error('Benchmark buffer indexing failed.');
        const next = Math.fround(value * 1.000061 + 0.000122);
        values[index] = next;
        checksum += next;
      }
    }
    scope.postMessage({
      id: request.id,
      elapsedMilliseconds: performance.now() - started,
      checksum,
      error: null,
    });
  } catch (error: unknown) {
    scope.postMessage({
      id: request.id,
      elapsedMilliseconds: 0,
      checksum: 0,
      error: error instanceof Error ? error.message : String(error),
    });
  }
});
