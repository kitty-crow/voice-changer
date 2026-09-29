import { expect, test } from 'bun:test';
import { readdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import ts from 'typescript';

const root = new URL('../', import.meta.url).pathname;

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

test('strict compiler options cannot be loosened', async () => {
  const raw: unknown = JSON.parse(await readFile(join(root, 'tsconfig.json'), 'utf8'));
  expect(typeof raw).toBe('object');
  expect(raw).not.toBeNull();
  const config = raw as Record<string, unknown>;
  const compilerOptions = config['compilerOptions'];
  expect(typeof compilerOptions).toBe('object');
  expect(compilerOptions).not.toBeNull();
  const options = compilerOptions as Record<string, unknown>;
  for (const key of [
    'strict',
    'noImplicitAny',
    'strictNullChecks',
    'useUnknownInCatchVariables',
    'noUncheckedIndexedAccess',
    'exactOptionalPropertyTypes',
    'noPropertyAccessFromIndexSignature',
  ]) expect(options[key]).toBe(true);
  expect(options['skipLibCheck']).toBe(false);
});

test('browser TypeScript contains no explicit any or suppression comments', async () => {
  const files = await sourceFiles(join(root, 'src'));
  expect(files.length).toBeGreaterThan(0);
  for (const file of files) {
    const text = await readFile(file, 'utf8');
    expect(text.includes('@ts-ignore')).toBe(false);
    expect(text.includes('@ts-nocheck')).toBe(false);
    const source = ts.createSourceFile(file, text, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);
    const explicitAny: number[] = [];
    const visit = (node: ts.Node): void => {
      if (node.kind === ts.SyntaxKind.AnyKeyword) explicitAny.push(node.getStart(source));
      ts.forEachChild(node, visit);
    };
    visit(source);
    expect(explicitAny).toEqual([]);
  }
});

test('canonical browser source contains no handwritten JavaScript', async () => {
  const entries = await readdir(join(root, 'src'), { recursive: true });
  const javascript = entries.filter((name) => typeof name === 'string' && (name.endsWith('.js') || name.endsWith('.mjs') || name.endsWith('.cjs')));
  expect(javascript).toEqual([]);
});
