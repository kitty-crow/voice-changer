import { SharedFloatRingBuffer } from './ring-buffer.js';
import type { ConversionSettings, ExecutionPlan, InferenceWorkerInit, InferenceWorkerStatus } from './types.js';

export class LocalAudioPath {
  private readonly plan: ExecutionPlan;
  private readonly settings: ConversionSettings;
  private context: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private worklet: AudioWorkletNode | null = null;
  private inferenceWorker: Worker | null = null;

  constructor(plan: ExecutionPlan, settings: ConversionSettings) {
    this.plan = plan;
    this.settings = settings;
  }

  async start(convert: boolean, onStatus: (status: InferenceWorkerStatus) => void): Promise<string> {
    if (this.context) return 'Audio is already running.';
    const context = new AudioContext({ latencyHint: 'interactive' });
    await context.audioWorklet.addModule('./audio-worklet.js');
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
    const source = context.createMediaStreamSource(stream);
    let worklet: AudioWorkletNode;
    let transport: string;

    if (this.plan.sharedAudioBuffers) {
      const capacity = Math.max(32768, context.sampleRate * 8);
      const inputRing = new SharedFloatRingBuffer(capacity);
      const outputRing = new SharedFloatRingBuffer(capacity);
      worklet = new AudioWorkletNode(context, 'local-voice-worklet', {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        outputChannelCount: [1],
        processorOptions: { inputRing: inputRing.descriptor, outputRing: outputRing.descriptor, dryWhenEmpty: !convert },
      });
      if (convert) {
        const worker = new Worker(new URL('./inference-worker.js', import.meta.url), { type: 'module' });
        worker.addEventListener('message', (event: MessageEvent<unknown>) => {
          if (typeof event.data !== 'object' || event.data === null) return;
          const record = event.data as Record<string, unknown>;
          if (record['kind'] !== 'status' || typeof record['message'] !== 'string') return;
          const latency = typeof record['latencyMilliseconds'] === 'number' ? record['latencyMilliseconds'] : null;
          const rtf = typeof record['realtimeFactor'] === 'number' ? record['realtimeFactor'] : null;
          onStatus({ kind: 'status', message: record['message'], latencyMilliseconds: latency, realtimeFactor: rtf });
        });
        const init: InferenceWorkerInit = {
          kind: 'init',
          plan: this.plan,
          inputRing: inputRing.descriptor,
          outputRing: outputRing.descriptor,
          inputSampleRate: context.sampleRate,
          settings: this.settings,
        };
        worker.postMessage(init);
        this.inferenceWorker = worker;
        transport = 'AudioWorklet + SharedArrayBuffer + dedicated RVC inference worker';
      } else {
        const scratch = new Float32Array(2048);
        const pump = (): void => {
          if (!this.context) return;
          const count = inputRing.read(scratch);
          if (count > 0) outputRing.write(scratch.subarray(0, count));
          window.setTimeout(pump, 4);
        };
        pump();
        transport = 'AudioWorklet + SharedArrayBuffer local pass-through';
      }
    } else {
      worklet = new AudioWorkletNode(context, 'local-voice-worklet', { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1] });
      worklet.port.addEventListener('message', (event: MessageEvent<unknown>) => {
        if (event.data instanceof Float32Array) worklet.port.postMessage(event.data, [event.data.buffer]);
      });
      worklet.port.start();
      transport = convert ? 'Message-based pass-through; cross-origin isolation is required for the real-time inference worker.' : 'AudioWorklet + message transport';
    }

    source.connect(worklet).connect(context.destination);
    this.context = context;
    this.stream = stream;
    this.source = source;
    this.worklet = worklet;
    return `${transport} at ${context.sampleRate} Hz.`;
  }

  async stop(): Promise<void> {
    this.inferenceWorker?.postMessage({ kind: 'stop' });
    this.inferenceWorker?.terminate();
    this.inferenceWorker = null;
    this.source?.disconnect();
    this.worklet?.disconnect();
    for (const track of this.stream?.getTracks() ?? []) track.stop();
    if (this.context) await this.context.close();
    this.context = null;
    this.stream = null;
    this.source = null;
    this.worklet = null;
  }
}
