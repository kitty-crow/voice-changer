import { expect, test } from 'bun:test';
import { readdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';

const root = new URL('../', import.meta.url).pathname;
const sourceRoot = join(root, 'src');

async function sourceFiles(directory: string): Promise<string[]> {
  const entries = await readdir(directory, { withFileTypes: true });
  const files: string[] = [];
  for (const entry of entries) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) files.push(...await sourceFiles(path));
    else if (entry.isFile() && entry.name.endsWith('.ts')) files.push(path);
  }
  return files;
}

test('inference source contains no server transport or local-server dependency', async () => {
  const forbidden = [
    'WebSocket',
    'XMLHttpRequest',
    'socket.io',
    '127.0.0.1',
    'localhost',
    '/api/',
    'MMVCServer',
  ] as const;

  for (const file of await sourceFiles(sourceRoot)) {
    if (file.endsWith('coi-serviceworker.ts')) continue;
    const source = await readFile(file, 'utf8');
    for (const token of forbidden) expect(source.includes(token)).toBe(false);
    expect(/\bfetch\s*\(/u.test(source)).toBe(false);
  }
});

test('page exposes no backend selector', async () => {
  const html = await readFile(join(root, 'index.html'), 'utf8');
  expect(/<(?:select|input)[^>]+(?:backend|provider|webgpu|webgl|wasm)/iu.test(html)).toBe(false);
});
