import * as ort from 'onnxruntime-web/webgpu';
import type { ExecutionPlan, ModelKind, ModelMetadata } from './types.js';

interface LoadedSession {
  readonly session: ort.InferenceSession;
  readonly metadata: ModelMetadata;
}

function executionProviders(plan: ExecutionPlan): ('webgpu' | 'wasm')[] {
  return plan.neuralBackend === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm'];
}

export class BrowserOrtRuntime {
  private readonly plan: ExecutionPlan;
  private readonly sessions = new Map<ModelKind, LoadedSession>();

  constructor(plan: ExecutionPlan) {
    this.plan = plan;
    ort.env.wasm.numThreads = plan.wasmThreads ? plan.workerCount : 1;
    ort.env.wasm.wasmPaths = './ort/';
  }

  async load(kind: ModelKind, name: string, bytes: ArrayBuffer): Promise<ModelMetadata> {
    const existing = this.sessions.get(kind);
    if (existing) void existing.session.release();
    let session: ort.InferenceSession;
    let backend = this.plan.neuralBackend;
    try {
      session = await ort.InferenceSession.create(bytes, { executionProviders: executionProviders(this.plan) });
    } catch (error: unknown) {
      if (this.plan.neuralBackend !== 'webgpu') throw error;
      session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] });
      backend = 'wasm';
    }
    const metadata: ModelMetadata = {
      kind,
      name,
      sizeBytes: bytes.byteLength,
      inputs: [...session.inputNames],
      outputs: [...session.outputNames],
      backend,
    };
    this.sessions.set(kind, { session, metadata });
    return metadata;
  }

  getSession(kind: ModelKind): ort.InferenceSession | null {
    return this.sessions.get(kind)?.session ?? null;
  }

  metadata(): readonly ModelMetadata[] {
    return [...this.sessions.values()].map((value) => value.metadata);
  }

  hasCompleteRvcStack(): boolean {
    return this.sessions.has('contentvec') && this.sessions.has('rmvpe') && this.sessions.has('rvc');
  }

  close(): void {
    for (const loaded of this.sessions.values()) void loaded.session.release();
    this.sessions.clear();
  }
}

export { ort };
