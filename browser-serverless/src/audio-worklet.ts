interface SharedRingDescriptor {
  readonly header: SharedArrayBuffer;
  readonly samples: SharedArrayBuffer;
  readonly capacity: number;
}

interface ProcessorOptions {
  readonly inputRing?: SharedRingDescriptor;
  readonly outputRing?: SharedRingDescriptor;
  readonly dryWhenEmpty?: boolean;
}

interface WorkletOptionsLike {
  readonly processorOptions?: ProcessorOptions;
}

function validDescriptor(value: SharedRingDescriptor | undefined): value is SharedRingDescriptor {
  return value !== undefined
    && value.header instanceof SharedArrayBuffer
    && value.samples instanceof SharedArrayBuffer
    && Number.isInteger(value.capacity)
    && value.capacity > 1;
}

class RingView {
  private readonly descriptor: SharedRingDescriptor;
  private readonly header: Int32Array;
  private readonly samples: Float32Array;

  constructor(descriptor: SharedRingDescriptor) {
    this.descriptor = descriptor;
    this.header = new Int32Array(descriptor.header);
    this.samples = new Float32Array(descriptor.samples);
  }

  write(source: Float32Array): void {
    let write = Atomics.load(this.header, 0);
    const read = Atomics.load(this.header, 1);
    for (let index = 0; index < source.length; index += 1) {
      const next = (write + 1) % this.descriptor.capacity;
      if (next === read) break;
      const value = source[index];
      if (value === undefined) break;
      this.samples[write] = value;
      write = next;
    }
    Atomics.store(this.header, 0, write);
  }

  read(target: Float32Array): number {
    const write = Atomics.load(this.header, 0);
    let read = Atomics.load(this.header, 1);
    let count = 0;
    while (read !== write && count < target.length) {
      const value = this.samples[read];
      if (value === undefined) break;
      target[count] = value;
      read = (read + 1) % this.descriptor.capacity;
      count += 1;
    }
    Atomics.store(this.header, 1, read);
    return count;
  }
}

class LocalVoiceWorklet extends AudioWorkletProcessor {
  private readonly inputRing: RingView | null;
  private readonly outputRing: RingView | null;
  private readonly dryWhenEmpty: boolean;
  private readonly queuedOutput: Float32Array[] = [];
  private queuedOffset = 0;
  private queuedSamples = 0;

  constructor(options?: WorkletOptionsLike) {
    super();
    const processorOptions = options?.processorOptions;
    this.inputRing = validDescriptor(processorOptions?.inputRing) ? new RingView(processorOptions.inputRing) : null;
    this.outputRing = validDescriptor(processorOptions?.outputRing) ? new RingView(processorOptions.outputRing) : null;
    this.dryWhenEmpty = processorOptions?.dryWhenEmpty ?? true;
    this.port.addEventListener('message', (event: MessageEvent<unknown>) => {
      if (!(event.data instanceof Float32Array) || event.data.length === 0) return;
      this.queuedOutput.push(event.data);
      this.queuedSamples += event.data.length;
      this.trimQueuedOutput(Math.max(128, Math.round(sampleRate * 2)));
    });
    this.port.start();
  }

  private trimQueuedOutput(maxSamples: number): void {
    while (this.queuedSamples > maxSamples && this.queuedOutput.length > 0) {
      const first = this.queuedOutput[0];
      if (!first) break;
      const available = first.length - this.queuedOffset;
      const excess = this.queuedSamples - maxSamples;
      const drop = Math.min(available, excess);
      this.queuedOffset += drop;
      this.queuedSamples -= drop;
      if (this.queuedOffset >= first.length) {
        this.queuedOutput.shift();
        this.queuedOffset = 0;
      }
    }
  }

  private readQueuedOutput(target: Float32Array): number {
    let written = 0;
    while (written < target.length && this.queuedOutput.length > 0) {
      const first = this.queuedOutput[0];
      if (!first) break;
      const available = first.length - this.queuedOffset;
      const count = Math.min(available, target.length - written);
      target.set(first.subarray(this.queuedOffset, this.queuedOffset + count), written);
      written += count;
      this.queuedOffset += count;
      this.queuedSamples -= count;
      if (this.queuedOffset >= first.length) {
        this.queuedOutput.shift();
        this.queuedOffset = 0;
      }
    }
    return written;
  }

  process(inputs: Float32Array[][], outputs: Float32Array[][]): boolean {
    const input = inputs[0]?.[0];
    const output = outputs[0]?.[0];
    if (!output) return true;
    output.fill(0);

    if (this.inputRing && this.outputRing) {
      if (input) this.inputRing.write(input);
      const converted = this.outputRing.read(output);
      if (converted === 0 && this.dryWhenEmpty && input) output.set(input.subarray(0, output.length));
      return true;
    }

    if (input) {
      const copy = input.slice();
      this.port.postMessage(copy, [copy.buffer]);
    }
    const converted = this.readQueuedOutput(output);
    if (converted === 0 && this.dryWhenEmpty && input) output.set(input.subarray(0, output.length));
    return true;
  }
}

registerProcessor('local-voice-worklet', LocalVoiceWorklet);
