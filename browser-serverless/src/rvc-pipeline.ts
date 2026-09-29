import { BrowserOrtRuntime, ort } from './ort-runtime.js';

const INPUT_SAMPLE_RATE = 16000;
const PITCH_WINDOW = 160;
const F0_MIN = 50;
const F0_MAX = 1100;
const F0_MEL_MIN = 1127 * Math.log(1 + F0_MIN / 700);
const F0_MEL_MAX = 1127 * Math.log(1 + F0_MAX / 700);

interface FeatureBlock {
  readonly data: Float32Array;
  readonly frames: number;
  readonly channels: number;
}

interface PitchBlock {
  readonly coarse: bigint[];
  readonly continuous: Float32Array;
}

function floatToHalf(value: number): number {
  const float = new Float32Array(1);
  const bits = new Uint32Array(float.buffer);
  float[0] = value;
  const raw = bits[0];
  if (raw === undefined) return 0;
  const sign = (raw >>> 16) & 0x8000;
  const exponent = ((raw >>> 23) & 0xff) - 127 + 15;
  const mantissa = raw & 0x7fffff;
  if (exponent <= 0) {
    if (exponent < -10) return sign;
    const shifted = (mantissa | 0x800000) >>> (1 - exponent);
    return sign | ((shifted + 0x1000) >>> 13);
  }
  if (exponent >= 31) return sign | 0x7c00;
  return sign | (exponent << 10) | ((mantissa + 0x1000) >>> 13);
}

function halfToFloat(raw: number): number {
  const sign = (raw & 0x8000) ? -1 : 1;
  const exponent = (raw >>> 10) & 0x1f;
  const fraction = raw & 0x03ff;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  if (exponent === 31) return fraction === 0 ? sign * Number.POSITIVE_INFINITY : Number.NaN;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

async function tensorFloats(tensor: ort.Tensor): Promise<Float32Array> {
  const data = await tensor.getData();
  if (data instanceof Float32Array) return data.slice();
  if (
    data instanceof Float64Array
    || data instanceof Int32Array
    || data instanceof Int16Array
    || data instanceof Int8Array
    || data instanceof Uint32Array
    || data instanceof Uint8Array
  ) return Float32Array.from(data);
  if (data instanceof Uint16Array) {
    if (tensor.type === 'float16') return Float32Array.from(data, halfToFloat);
    return Float32Array.from(data);
  }
  throw new Error(`Unsupported numeric tensor output type: ${tensor.type}.`);
}

function tensorOutput(outputs: ort.InferenceSession.ReturnType, preferred: string): ort.Tensor {
  const direct = outputs[preferred];
  if (direct instanceof ort.Tensor) return direct;
  for (const value of Object.values(outputs)) {
    if (value instanceof ort.Tensor) return value;
  }
  throw new Error('ONNX session returned no tensor outputs.');
}

function inputTensor(
  session: ort.InferenceSession,
  inputIndex: number,
  data: Float32Array,
  dims: readonly number[],
): ort.Tensor {
  const metadata = session.inputMetadata[inputIndex];
  if (metadata?.isTensor === true && metadata.type === 'float16') {
    return new ort.Tensor('float16', Uint16Array.from(data, floatToHalf), dims);
  }
  return new ort.Tensor('float32', data, dims);
}

function bigintTensor(values: readonly number[], dims: readonly number[]): ort.Tensor {
  return new ort.Tensor('int64', values.map((value) => BigInt(Math.trunc(value))), dims);
}

function inputName(session: ort.InferenceSession, expected: string): string | null {
  return session.inputNames.find((name) => name.toLowerCase() === expected.toLowerCase()) ?? null;
}

function requiredInputName(session: ort.InferenceSession, expected: string, fallbackIndex: number): string {
  return inputName(session, expected) ?? session.inputNames[fallbackIndex] ?? expected;
}

function expectedFeatureChannels(session: ort.InferenceSession): number | null {
  const featsName = requiredInputName(session, 'feats', 0);
  const index = session.inputNames.indexOf(featsName);
  const metadata = session.inputMetadata[index];
  if (metadata?.isTensor !== true || metadata.shape.length !== 3) return null;
  const channelDimension = metadata.shape[2];
  return typeof channelDimension === 'number' && channelDimension > 0 ? channelDimension : null;
}

function duplicateFeatures(
  source: Float32Array,
  dims: readonly number[],
  expectedChannels: number | null,
): FeatureBlock {
  if (dims.length !== 3 || dims[0] !== 1) {
    throw new Error(`ContentVec output must be rank-3 [1,T,C] or [1,C,T], got [${dims.join(',')}].`);
  }
  const second = dims[1];
  const third = dims[2];
  if (second === undefined || third === undefined) throw new Error('ContentVec output dimensions are incomplete.');

  const commonChannels = new Set([256, 512, 768, 1024]);
  let channelFirst: boolean;
  if (expectedChannels !== null && second === expectedChannels && third !== expectedChannels) channelFirst = true;
  else if (expectedChannels !== null && third === expectedChannels) channelFirst = false;
  else channelFirst = commonChannels.has(second) && !commonChannels.has(third);

  const frames = channelFirst ? third : second;
  const channels = channelFirst ? second : third;
  if (frames <= 0 || channels <= 0 || source.length < frames * channels) {
    throw new Error('ContentVec output size does not match its shape.');
  }
  if (expectedChannels !== null && channels !== expectedChannels) {
    throw new Error(
      `ContentVec/RVC feature width mismatch: encoder produced ${channels} channels but RVC expects ${expectedChannels}. `
      + 'Use the final-projection ContentVec model for 256-channel RVC v1 models and the hidden-feature model for 768-channel RVC v2 models.',
    );
  }

  const doubled = new Float32Array(frames * 2 * channels);
  for (let frame = 0; frame < frames; frame += 1) {
    for (let channel = 0; channel < channels; channel += 1) {
      const sourceIndex = channelFirst ? channel * frames + frame : frame * channels + channel;
      const value = source[sourceIndex];
      if (value === undefined) throw new Error('ContentVec feature indexing failed.');
      const base = frame * 2 * channels + channel;
      doubled[base] = value;
      doubled[base + channels] = value;
    }
  }
  return { data: doubled, frames: frames * 2, channels };
}

function alignTail(source: Float32Array, length: number): Float32Array {
  const result = new Float32Array(length);
  if (source.length >= length) result.set(source.subarray(source.length - length));
  else result.set(source, length - source.length);
  return result;
}

function coarsePitch(f0: Float32Array, pitchShift: number): PitchBlock {
  const ratio = 2 ** (pitchShift / 12);
  const continuous = new Float32Array(f0.length);
  const coarse: bigint[] = new Array<bigint>(f0.length);
  for (let index = 0; index < f0.length; index += 1) {
    const shifted = (f0[index] ?? 0) * ratio;
    continuous[index] = shifted;
    let mel = 1127 * Math.log(1 + shifted / 700);
    if (mel > 0) mel = (mel - F0_MEL_MIN) * 254 / (F0_MEL_MAX - F0_MEL_MIN) + 1;
    mel = Math.max(1, Math.min(255, mel));
    coarse[index] = BigInt(Math.round(mel));
  }
  return { coarse, continuous };
}

export function resampleLinear(input: Float32Array, sourceRate: number, targetRate: number): Float32Array {
  if (sourceRate <= 0 || targetRate <= 0) throw new Error('Sample rates must be positive.');
  if (input.length === 0 || sourceRate === targetRate) return input.slice();
  const outputLength = Math.max(1, Math.round(input.length * targetRate / sourceRate));
  const output = new Float32Array(outputLength);
  const scale = sourceRate / targetRate;
  for (let index = 0; index < outputLength; index += 1) {
    const position = index * scale;
    const left = Math.min(input.length - 1, Math.floor(position));
    const right = Math.min(input.length - 1, left + 1);
    const fraction = position - left;
    const a = input[left] ?? 0;
    const b = input[right] ?? a;
    output[index] = a + (b - a) * fraction;
  }
  return output;
}

export class BrowserRvcPipeline {
  private readonly content: ort.InferenceSession;
  private readonly rmvpe: ort.InferenceSession | null;
  private readonly rvc: ort.InferenceSession;
  private readonly needsPitch: boolean;
  private readonly featureChannels: number | null;

  constructor(runtime: BrowserOrtRuntime) {
    const content = runtime.getSession('contentvec');
    const rvc = runtime.getSession('rvc');
    if (!content || !rvc) throw new Error('ContentVec and RVC sessions must be loaded.');
    this.content = content;
    this.rvc = rvc;
    this.needsPitch = runtime.rvcNeedsPitch();
    this.rmvpe = this.needsPitch ? runtime.getSession('rmvpe') : null;
    if (this.needsPitch && !this.rmvpe) throw new Error('This RVC model requires pitch, but RMVPE is not loaded.');
    this.featureChannels = expectedFeatureChannels(rvc);
  }

  private async extractFeatures(audio16k: Float32Array): Promise<FeatureBlock> {
    if (this.content.inputNames.length === 0) throw new Error('ContentVec model has no inputs.');
    const feeds: Record<string, ort.Tensor> = {};
    for (let index = 0; index < this.content.inputNames.length; index += 1) {
      const name = this.content.inputNames[index];
      if (!name) continue;
      const lower = name.toLowerCase();
      if (index === 0 || lower.includes('source') || lower.includes('waveform') || lower.includes('input_values')) {
        feeds[name] = inputTensor(this.content, index, audio16k, [1, audio16k.length]);
      } else if (lower.includes('padding_mask')) {
        feeds[name] = new ort.Tensor('bool', new Uint8Array(audio16k.length), [1, audio16k.length]);
      } else if (lower.includes('output_layer')) {
        feeds[name] = bigintTensor([this.featureChannels === 768 ? 12 : 9], [1]);
      } else {
        throw new Error(`Unsupported ContentVec input '${name}'.`);
      }
    }
    const outputs = await this.content.run(feeds);
    const tensor = tensorOutput(outputs, this.content.outputNames[0] ?? 'output');
    return duplicateFeatures(await tensorFloats(tensor), tensor.dims, this.featureChannels);
  }

  private async extractPitch(audio16k: Float32Array, pitchShift: number, frames: number): Promise<PitchBlock> {
    const rmvpe = this.rmvpe;
    if (!rmvpe || rmvpe.inputNames.length === 0) throw new Error('RMVPE model has no inputs.');
    const feeds: Record<string, ort.Tensor> = {};
    for (let index = 0; index < rmvpe.inputNames.length; index += 1) {
      const name = rmvpe.inputNames[index];
      if (!name) continue;
      const lower = name.toLowerCase();
      if (index === 0 || lower.includes('waveform') || lower.includes('audio')) {
        feeds[name] = inputTensor(rmvpe, index, audio16k, [1, audio16k.length]);
      } else if (lower.includes('threshold')) {
        feeds[name] = new ort.Tensor('float32', new Float32Array([0.3]), [1]);
      } else {
        throw new Error(`Unsupported RMVPE input '${name}'.`);
      }
    }
    const outputs = await rmvpe.run(feeds);
    const f0Tensor = tensorOutput(outputs, rmvpe.outputNames.find((name) => name.toLowerCase() === 'f0') ?? 'f0');
    return coarsePitch(alignTail(await tensorFloats(f0Tensor), frames), pitchShift);
  }

  async convert(audio16k: Float32Array, pitchShift: number, speakerId: number): Promise<Float32Array> {
    if (audio16k.length < INPUT_SAMPLE_RATE / 20) throw new Error('Audio chunk is too short for RVC inference.');
    const features = await this.extractFeatures(audio16k);
    const frames = this.needsPitch
      ? Math.max(1, Math.min(features.frames, Math.floor(audio16k.length / PITCH_WINDOW)))
      : features.frames;
    const channels = features.channels;
    const featureStartFrame = features.frames - frames;
    const featureData = features.data.subarray(featureStartFrame * channels);

    const feeds: Record<string, ort.Tensor> = {};
    const featsName = requiredInputName(this.rvc, 'feats', 0);
    const lengthName = requiredInputName(this.rvc, 'p_len', 1);
    const sidName = requiredInputName(this.rvc, 'sid', this.needsPitch ? 4 : 2);
    const featsIndex = this.rvc.inputNames.indexOf(featsName);
    feeds[featsName] = inputTensor(this.rvc, Math.max(0, featsIndex), featureData, [1, frames, channels]);
    feeds[lengthName] = bigintTensor([frames], [1]);

    if (this.needsPitch) {
      const pitch = await this.extractPitch(audio16k, pitchShift, frames);
      const pitchName = inputName(this.rvc, 'pitch');
      const pitchfName = inputName(this.rvc, 'pitchf');
      if (!pitchName || !pitchfName) throw new Error('RVC pitch inputs are incomplete.');
      feeds[pitchName] = new ort.Tensor('int64', pitch.coarse, [1, frames]);
      feeds[pitchfName] = new ort.Tensor('float32', pitch.continuous, [1, frames]);
    }
    feeds[sidName] = bigintTensor([speakerId], [1]);

    const outputs = await this.rvc.run(feeds);
    const audioTensor = tensorOutput(outputs, this.rvc.outputNames.find((name) => name.toLowerCase() === 'audio') ?? 'audio');
    const audio = await tensorFloats(audioTensor);
    for (let index = 0; index < audio.length; index += 1) {
      audio[index] = Math.max(-1, Math.min(1, audio[index] ?? 0));
    }
    return audio;
  }
}
