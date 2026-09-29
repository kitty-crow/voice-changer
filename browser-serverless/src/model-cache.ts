import type { ModelKind } from './types.js';

const DATABASE = 'voice-changer-serverless';
const STORE = 'models';
const VERSION = 1;

interface CachedModel {
  readonly kind: ModelKind;
  readonly name: string;
  readonly bytes: ArrayBuffer;
  readonly savedAt: number;
}

function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DATABASE, VERSION);
    request.addEventListener('upgradeneeded', () => {
      const database = request.result;
      if (!database.objectStoreNames.contains(STORE)) database.createObjectStore(STORE, { keyPath: 'kind' });
    });
    request.addEventListener('success', () => resolve(request.result));
    request.addEventListener('error', () => reject(request.error ?? new Error('IndexedDB open failed.')));
  });
}

function complete(transaction: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    transaction.addEventListener('complete', () => resolve());
    transaction.addEventListener('abort', () => reject(transaction.error ?? new Error('IndexedDB transaction aborted.')));
    transaction.addEventListener('error', () => reject(transaction.error ?? new Error('IndexedDB transaction failed.')));
  });
}

export async function putCachedModel(kind: ModelKind, file: File): Promise<ArrayBuffer> {
  const bytes = await file.arrayBuffer();
  const database = await openDatabase();
  try {
    const transaction = database.transaction(STORE, 'readwrite');
    const record: CachedModel = { kind, name: file.name, bytes, savedAt: Date.now() };
    transaction.objectStore(STORE).put(record);
    await complete(transaction);
    return bytes;
  } finally {
    database.close();
  }
}

export async function getCachedModel(kind: ModelKind): Promise<CachedModel | null> {
  const database = await openDatabase();
  try {
    const transaction = database.transaction(STORE, 'readonly');
    const request = transaction.objectStore(STORE).get(kind);
    const result = await new Promise<unknown>((resolve, reject) => {
      request.addEventListener('success', () => resolve(request.result));
      request.addEventListener('error', () => reject(request.error ?? new Error('IndexedDB read failed.')));
    });
    await complete(transaction);
    if (typeof result !== 'object' || result === null) return null;
    const record = result as Record<string, unknown>;
    if (record['kind'] !== kind || typeof record['name'] !== 'string' || !(record['bytes'] instanceof ArrayBuffer) || typeof record['savedAt'] !== 'number') return null;
    return { kind, name: record['name'], bytes: record['bytes'], savedAt: record['savedAt'] };
  } finally {
    database.close();
  }
}

export async function clearModelCache(): Promise<void> {
  const database = await openDatabase();
  try {
    const transaction = database.transaction(STORE, 'readwrite');
    transaction.objectStore(STORE).clear();
    await complete(transaction);
  } finally {
    database.close();
  }
}
