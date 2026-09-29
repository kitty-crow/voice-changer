import * as ort from 'onnxruntime-web/all';
import type { ExecutionPlan, ModelKind, ModelMetadata, NeuralBackend } from './types.js';

interface LoadedSession {
  readonly session: ort.InferenceSession;
  readonly metadata: ModelMetadata;
}

function executionProviders(backend: NeuralBackend): ('webgpu' | 'webgl' | 'wasm')[] {
  if (backend === 'webgpu') return ['webgpu', 'wasm'];
  if (backend === 'webgl') return ['webgl', 'wasm'];
  return ['wasm'];
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
      session = await ort.InferenceSession.create(bytes, { executionProviders: executionProviders(this.plan.neuralBackend) });
    } catch (error: unknown) {
      if (this.plan.neuralBackend === 'wasm') throw error;
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

  rvcNeedsPitch(): boolean {
    const session = this.sessions.get('rvc')?.session;
    if (!session) return true;
    const names = new Set(session.inputNames.map((name) => name.toLowerCase()));
    return names.has('pitch') || names.has('pitchf');
  }

  hasCompleteRvcStack(): boolean {
    if (!this.sessions.has('contentvec') || !this.sessions.has('rvc')) return false;
    return !this.rvcNeedsPitch() || this.sessions.has('rmvpe');
  }

  close(): void {
    for (const loaded of this.sessions.values()) void loaded.session.release();
    this.sessions.clear();
  }
}

export { ort };
