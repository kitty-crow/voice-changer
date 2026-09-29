import { SharedFloatRingBuffer } from './ring-buffer.js';
import type { ExecutionPlan } from './types.js';

export class LocalAudioPath {
  private readonly plan: ExecutionPlan;
  private context: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private worklet: AudioWorkletNode | null = null;
  private pump: number | null = null;

  constructor(plan: ExecutionPlan) {
    this.plan = plan;
  }

  async start(): Promise<string> {
    if (this.context) return 'Audio is already running.';
    const context = new AudioContext({ latencyHint: 'interactive' });
    await context.audioWorklet.addModule('./audio-worklet.js');
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });
    const source = context.createMediaStreamSource(stream);

    let worklet: AudioWorkletNode;
    let transport: string;
    if (this.plan.sharedAudioBuffers) {
      const capacity = Math.max(16384, context.sampleRate * 2);
      const inputRing = new SharedFloatRingBuffer(capacity);
      const outputRing = new SharedFloatRingBuffer(capacity);
      worklet = new AudioWorkletNode(context, 'local-voice-worklet', {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        outputChannelCount: [1],
        processorOptions: {
          inputRing: inputRing.descriptor,
          outputRing: outputRing.descriptor,
        },
      });
      const scratch = new Float32Array(2048);
      const pump = (): void => {
        const count = inputRing.read(scratch);
        if (count > 0) outputRing.write(scratch.subarray(0, count));
        this.pump = window.setTimeout(pump, 4);
      };
      pump();
      transport = 'AudioWorklet + SharedArrayBuffer ring buffers';
    } else {
      worklet = new AudioWorkletNode(context, 'local-voice-worklet', {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        outputChannelCount: [1],
      });
      worklet.port.addEventListener('message', (event: MessageEvent<unknown>) => {
        if (event.data instanceof Float32Array) worklet.port.postMessage(event.data, [event.data.buffer]);
      });
      worklet.port.start();
      transport = 'AudioWorklet + message transport';
    }

    source.connect(worklet).connect(context.destination);
    this.context = context;
    this.stream = stream;
    this.source = source;
    this.worklet = worklet;
    return `${transport} at ${context.sampleRate} Hz.`;
  }

  async stop(): Promise<void> {
    if (this.pump !== null) {
      window.clearTimeout(this.pump);
      this.pump = null;
    }
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
