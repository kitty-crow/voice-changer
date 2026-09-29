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

function lowerNames(session: ort.InferenceSession): Set<string> {
  return new Set(session.inputNames.map((name) => name.toLowerCase()));
}

function tensorShape(metadata: ort.InferenceSession.ValueMetadata | undefined): readonly (number | string)[] | null {
  return metadata?.isTensor === true ? metadata.shape : null;
}

function staticDimension(value: number | string | undefined): number | null {
  return typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : null;
}

export class BrowserOrtRuntime {
  private readonly plan: ExecutionPlan;
  private readonly sessions = new Map<ModelKind, LoadedSession>();

  constructor(plan: ExecutionPlan) {
    this.plan = plan;
    ort.env.wasm.numThreads = plan.wasmThreads ? plan.workerCount : 1;
    ort.env.wasm.proxy = false;
    ort.env.wasm.wasmPaths = './ort/';
  }

  async load(kind: ModelKind, name: string, bytes: ArrayBuffer): Promise<ModelMetadata> {
    const existing = this.sessions.get(kind);
    if (existing) void existing.session.release();
    let session: ort.InferenceSession;
    let backend = this.plan.neuralBackend;
    try {
      session = await ort.InferenceSession.create(bytes, {
        executionProviders: executionProviders(this.plan.neuralBackend),
      });
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
    const names = lowerNames(session);
    return names.has('pitch') || names.has('pitchf');
  }

  hasCompleteRvcStack(): boolean {
    if (!this.sessions.has('contentvec') || !this.sessions.has('rvc')) return false;
    return !this.rvcNeedsPitch() || this.sessions.has('rmvpe');
  }

  validateRvcStack(): readonly string[] {
    const content = this.sessions.get('contentvec')?.session;
    const rvc = this.sessions.get('rvc')?.session;
    if (!content || !rvc) throw new Error('ContentVec and RVC models must both be loaded.');

    const rvcNames = lowerNames(rvc);
    for (const required of ['feats', 'p_len', 'sid'] as const) {
      if (!rvcNames.has(required)) throw new Error(`RVC model is missing required '${required}' input.`);
    }
    const hasPitch = rvcNames.has('pitch');
    const hasPitchf = rvcNames.has('pitchf');
    if (hasPitch !== hasPitchf) {
      throw new Error('RVC model exposes only one of pitch/pitchf; both are required for an F0 model.');
    }
    if (hasPitch && !this.sessions.has('rmvpe')) {
      throw new Error('This RVC model requires F0 inputs, so an RMVPE ONNX model must also be loaded.');
    }

    const featsIndex = rvc.inputNames.findIndex((name) => name.toLowerCase() === 'feats');
    const rvcFeatureShape = tensorShape(rvc.inputMetadata[featsIndex]);
    const expectedChannels = staticDimension(rvcFeatureShape?.[2]);

    const contentShapes = content.outputMetadata
      .map((metadata) => tensorShape(metadata))
      .filter((shape): shape is readonly (number | string)[] => shape !== null && shape.length === 3);
    if (expectedChannels !== null && contentShapes.length > 0) {
      const compatible = contentShapes.some((shape) => {
        const second = staticDimension(shape[1]);
        const third = staticDimension(shape[2]);
        return second === expectedChannels || third === expectedChannels;
      });
      const fullyStatic = contentShapes.some((shape) => staticDimension(shape[1]) !== null && staticDimension(shape[2]) !== null);
      if (!compatible && fullyStatic) {
        throw new Error(
          `ContentVec/RVC feature-width mismatch: RVC expects ${expectedChannels} channels, but no ContentVec output has that static width.`,
        );
      }
    }

    const notes: string[] = [];
    notes.push(hasPitch ? 'F0 model · RMVPE required' : 'F0-less model · RMVPE not required');
    if (expectedChannels !== null) notes.push(`${expectedChannels}-channel RVC features`);
    else notes.push('dynamic RVC feature width');
    notes.push(`providers: ${this.metadata().map((model) => `${model.kind}=${model.backend}`).join(', ')}`);
    return notes;
  }

  close(): void {
    for (const loaded of this.sessions.values()) void loaded.session.release();
    this.sessions.clear();
  }
}

export { ort };
