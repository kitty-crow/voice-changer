import type { SharedRingDescriptor } from './types.js';

const WRITE_INDEX = 0;
const READ_INDEX = 1;

export class SharedFloatRingBuffer {
  readonly descriptor: SharedRingDescriptor;
  private readonly header: Int32Array;
  private readonly samples: Float32Array;

  constructor(capacity: number, descriptor?: SharedRingDescriptor) {
    if (!Number.isInteger(capacity) || capacity < 2) throw new Error('Ring-buffer capacity must be at least two samples.');
    const activeDescriptor = descriptor ?? {
      header: new SharedArrayBuffer(Int32Array.BYTES_PER_ELEMENT * 2),
      samples: new SharedArrayBuffer(Float32Array.BYTES_PER_ELEMENT * capacity),
      capacity,
    };
    if (activeDescriptor.capacity !== capacity) throw new Error('Shared ring descriptor capacity mismatch.');
    this.descriptor = activeDescriptor;
    this.header = new Int32Array(activeDescriptor.header);
    this.samples = new Float32Array(activeDescriptor.samples);
  }

  availableRead(): number {
    const write = Atomics.load(this.header, WRITE_INDEX);
    const read = Atomics.load(this.header, READ_INDEX);
    return write >= read ? write - read : this.descriptor.capacity - read + write;
  }

  availableWrite(): number {
    return this.descriptor.capacity - 1 - this.availableRead();
  }

  write(source: Float32Array): number {
    let write = Atomics.load(this.header, WRITE_INDEX);
    const count = Math.min(source.length, this.availableWrite());
    for (let index = 0; index < count; index += 1) {
      const value = source[index];
      if (value === undefined) throw new Error('Source audio indexing failed.');
      this.samples[write] = value;
      write = (write + 1) % this.descriptor.capacity;
    }
    Atomics.store(this.header, WRITE_INDEX, write);
    return count;
  }

  read(target: Float32Array): number {
    let read = Atomics.load(this.header, READ_INDEX);
    const count = Math.min(target.length, this.availableRead());
    for (let index = 0; index < count; index += 1) {
      const value = this.samples[read];
      if (value === undefined) throw new Error('Shared audio indexing failed.');
      target[index] = value;
      read = (read + 1) % this.descriptor.capacity;
    }
    Atomics.store(this.header, READ_INDEX, read);
    return count;
  }
}
