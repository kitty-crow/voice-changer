export type NeuralBackend = 'webgpu' | 'wasm';
export type MemoryTier = 'low' | 'medium' | 'high';
export type ModelKind = 'contentvec' | 'rmvpe' | 'rvc';

export interface WebGpuProfile {
  readonly available: boolean;
  readonly adapter: GPUAdapter | null;
  readonly description: string | null;
  readonly maxBufferSize: number;
  readonly maxStorageBufferBindingSize: number;
}

export interface WebGlProfile {
  readonly available: boolean;
  readonly renderer: string | null;
}

export interface HardwareProfile {
  readonly webgpu: WebGpuProfile;
  readonly webgl2: WebGlProfile;
  readonly wasm: boolean;
  readonly wasmSimd: boolean;
  readonly wasmThreads: boolean;
  readonly sharedMemory: boolean;
  readonly crossOriginIsolated: boolean;
  readonly audioWorklet: boolean;
  readonly workerSupport: boolean;
  readonly logicalCores: number;
  readonly deviceMemoryGiB: number | null;
  readonly heapLimitMiB: number | null;
  readonly memoryTier: MemoryTier;
  readonly memoryBudgetMiB: number;
  readonly workerCount: number;
}

export interface PreflightBenchmark {
  readonly backend: NeuralBackend;
  readonly milliseconds: number;
  readonly passed: boolean;
  readonly detail: string;
}

export interface ExecutionPlan {
  readonly neuralBackend: NeuralBackend;
  readonly wasmThreads: boolean;
  readonly workerCount: number;
  readonly sharedAudioBuffers: boolean;
}

export interface PreflightResult {
  readonly profile: HardwareProfile;
  readonly plan: ExecutionPlan;
  readonly benchmarks: readonly PreflightBenchmark[];
  readonly completedAt: number;
}

export interface ModelMetadata {
  readonly kind: ModelKind;
  readonly name: string;
  readonly sizeBytes: number;
  readonly inputs: readonly string[];
  readonly outputs: readonly string[];
  readonly backend: NeuralBackend;
}

export interface SharedRingDescriptor {
  readonly header: SharedArrayBuffer;
  readonly samples: SharedArrayBuffer;
  readonly capacity: number;
}

export interface BenchmarkRequest {
  readonly id: number;
  readonly iterations: number;
  readonly sampleCount: number;
}

export interface BenchmarkResponse {
  readonly id: number;
  readonly elapsedMilliseconds: number;
  readonly checksum: number;
  readonly error: string | null;
}

export interface ConversionSettings {
  readonly pitchShift: number;
  readonly speakerId: number;
  readonly modelSampleRate: number;
  readonly chunkMilliseconds: number;
}

export interface InferenceWorkerInit {
  readonly kind: 'init';
  readonly plan: ExecutionPlan;
  readonly inputRing: SharedRingDescriptor;
  readonly outputRing: SharedRingDescriptor;
  readonly inputSampleRate: number;
  readonly settings: ConversionSettings;
}

export interface InferenceWorkerStop {
  readonly kind: 'stop';
}

export type InferenceWorkerRequest = InferenceWorkerInit | InferenceWorkerStop;

export interface InferenceWorkerStatus {
  readonly kind: 'status';
  readonly message: string;
  readonly latencyMilliseconds: number | null;
  readonly realtimeFactor: number | null;
}
