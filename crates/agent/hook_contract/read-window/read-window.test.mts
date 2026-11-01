// Tests of the read window hook: `node --test read-window.test.mts`.
// Each test runs the real script as Zed does, with its input on stdin.
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const script = path.join(here, 'read-window.mts');

function run(input: object | string): any {
  const result = spawnSync(process.execPath, [script], {
    input: typeof input === 'string' ? input : JSON.stringify(input),
    encoding: 'utf8',
  });
  assert.equal(result.status, 0, result.stderr);
  return result.stdout.trim() ? JSON.parse(result.stdout) : {};
}

/** A project folder with a file of `lines` lines at each relative path. */
function project(files: Record<string, number>): string {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'read-window-'));
  for (const [file, lines] of Object.entries(files)) {
    const full = path.join(root, file);
    fs.mkdirSync(path.dirname(full), { recursive: true });
    fs.writeFileSync(full, Array.from({ length: lines }, (_, i) => `line ${i + 1}`).join('\n') + '\n');
  }
  return root;
}

/** A PreToolUse input for read_file, from a project rooted at `cwd`. */
function readFile(cwd: string, toolInput: object): object {
  return {
    session_id: 'root-1',
    transcript_path: null,
    cwd,
    hook_event_name: 'PreToolUse',
    zed: { contract_version: '0.1.0', thread_id: 'thread-1', is_subagent: true },
    tool_name: 'read_file',
    tool_input: toolInput,
    tool_use_id: 'r1',
  };
}

/** The Zed form of a project path: the root folder name, then the path. */
function zedPath(cwd: string, file: string): string {
  return `${path.basename(cwd)}/${file}`;
}

function denial(output: any): string | undefined {
  const specific = output.hookSpecificOutput;
  if (specific?.permissionDecision !== 'deny') {
    return undefined;
  }
  assert.equal(specific.hookEventName, 'PreToolUse');
  return specific.permissionDecisionReason;
}

test('a whole read of a file of more than 120 lines is denied, with what to do instead', () => {
  const cwd = project({ 'src/big.ts': 200 });
  const reason = denial(run(readFile(cwd, { path: zedPath(cwd, 'src/big.ts') })));
  assert.ok(reason, 'expected a denial');
  // In the words of Zed's own answer for a large file (crates/agent/src/tools/read_file_tool.rs:522-530).
  for (const phrase of [
    'This file is too large to read all at once',
    'it has 200 lines',
    'at most 120 lines',
    'IMPORTANT: Do NOT retry this call without line numbers',
    'find the lines you need with grep first',
    'calling this tool again with start_line and end_line parameters',
    'NEXT STEPS:',
    'call read_file with the same path plus start_line and end_line',
    'use start_line: 100 and end_line: 150.',
  ]) {
    assert.ok(reason.includes(phrase), `missing « ${phrase} » in:\n${reason}`);
  }
});

test('a window too large is denied in the same words', () => {
  const cwd = project({ 'src/big.ts': 300 });
  const reason = denial(run(readFile(cwd, { path: zedPath(cwd, 'src/big.ts'), start_line: 1, end_line: 200 })));
  assert.ok(reason, 'expected a denial');
  for (const phrase of [
    'This window is too large to read at once',
    'it covers 200 lines',
    'at most 120 lines',
    'IMPORTANT: Do NOT retry this call with the same window',
    'find the lines you need with grep first',
    'NEXT STEPS:',
    'call read_file with the same path plus start_line and end_line',
    'use start_line: 100 and end_line: 150.',
  ]) {
    assert.ok(reason.includes(phrase), `missing « ${phrase} » in:\n${reason}`);
  }
});

test('a file of 120 lines or fewer is read whole', () => {
  const cwd = project({ 'docs/story.md': 120, 'plan.md': 3 });
  assert.deepEqual(run(readFile(cwd, { path: zedPath(cwd, 'docs/story.md') })), {});
  assert.deepEqual(run(readFile(cwd, { path: zedPath(cwd, 'plan.md') })), {});
});

test('a skill is read whole, whatever its length', () => {
  const cwd = project({ '.agents/skills/tdd/SKILL.md': 400, '.claude/skills/tdd/reference.md': 400 });
  assert.deepEqual(run(readFile(cwd, { path: zedPath(cwd, '.agents/skills/tdd/SKILL.md') })), {});
  assert.deepEqual(run(readFile(cwd, { path: path.join(cwd, '.claude/skills/tdd/reference.md') })), {});
});

test('a window of more than 120 lines is denied, one of 120 lines passes', () => {
  const cwd = project({ 'src/big.ts': 300 });
  const file = zedPath(cwd, 'src/big.ts');
  const reason = denial(run(readFile(cwd, { path: file, start_line: 1, end_line: 200 })));
  assert.ok(reason, 'expected a denial');
  assert.match(reason, /200 lines/);
  assert.deepEqual(run(readFile(cwd, { path: file, start_line: 50, end_line: 169 })), {});
});

test('a window with one bound only is measured against the file', () => {
  const cwd = project({ 'src/big.ts': 200 });
  const file = zedPath(cwd, 'src/big.ts');
  // From line 150 to the end: 51 lines.
  assert.deepEqual(run(readFile(cwd, { path: file, start_line: 150 })), {});
  // From line 10 to the end: 191 lines.
  assert.ok(denial(run(readFile(cwd, { path: file, start_line: 10 }))));
  // From the start to line 130: 130 lines.
  assert.ok(denial(run(readFile(cwd, { path: file, end_line: 130 }))));
});

test('a path prefixed with the root folder name, or absolute, is found', () => {
  const cwd = project({ 'epics/e1/notes.md': 200 });
  assert.ok(denial(run(readFile(cwd, { path: zedPath(cwd, 'epics/e1/notes.md') }))));
  assert.ok(denial(run(readFile(cwd, { path: path.join(cwd, 'epics/e1/notes.md') }))));
  assert.ok(denial(run(readFile(cwd, { path: 'epics/e1/notes.md' }))));
});

test('a file it cannot find decides nothing', () => {
  const cwd = project({});
  assert.deepEqual(run(readFile(cwd, { path: zedPath(cwd, 'missing.ts') })), {});
});

test('other tools are left alone', () => {
  const cwd = project({ 'src/big.ts': 200 });
  const input = { ...readFile(cwd, { path: zedPath(cwd, 'src/big.ts') }), tool_name: 'grep' };
  assert.deepEqual(run(input), {});
});

test('a window with both bounds is decided on its size, as the model asked for it', () => {
  const cwd = project({ 'src/short.ts': 50 });
  const reason = denial(run(readFile(cwd, { path: zedPath(cwd, 'src/short.ts'), start_line: 1, end_line: 200 })));
  assert.match(reason ?? '', /covers 200 lines/);
});

test('the hook adds less than 50 ms to node start-up on a file of 10 MB', () => {
  const cwd = project({});
  const line = 'const value = compute(1, 2, 3); // one line of a big file\n';
  fs.writeFileSync(path.join(cwd, 'big.js'), line.repeat(Math.ceil(10_000_000 / line.length)));
  const median = (input: object | string) => {
    const times = Array.from({ length: 5 }, () => {
      const start = performance.now();
      run(input);
      return performance.now() - start;
    }).sort((a, b) => a - b);
    return times[2];
  };
  // An unreadable input measures node start-up alone.
  const startUp = median('not json');
  // A window with one bound makes the hook count every line of the file.
  const call = readFile(cwd, { path: zedPath(cwd, 'big.js'), start_line: 10 });
  const work = median(call) - startUp;
  assert.ok(denial(run(call)));
  assert.ok(work < 50, `${Math.round(work)} ms over a start-up of ${Math.round(startUp)} ms`);
});

/** A file of `bytes` bytes on disk, in lines of 64 characters, with this line ending. */
function sized(cwd: string, file: string, bytes: number, ending = '\n'): void {
  const line = `${'x'.repeat(64 - ending.length)}${ending}`;
  const text = line.repeat(Math.floor(bytes / line.length));
  fs.writeFileSync(path.join(cwd, file), text + 'y'.repeat(bytes - text.length));
}

test('a file over 16384 characters is read whole: Zed answers with its outline', () => {
  // crates/agent/src/outline.rs:10, AUTO_OUTLINE_SIZE = 16384, compared with `>` at line 35.
  const cwd = project({});
  sized(cwd, 'at-limit.ts', 16384);
  sized(cwd, 'over-limit.ts', 16385);
  assert.ok(denial(run(readFile(cwd, { path: zedPath(cwd, 'at-limit.ts') }))));
  assert.deepEqual(run(readFile(cwd, { path: zedPath(cwd, 'over-limit.ts') })), {});
  // A window still counts: the outline is only for a read without one.
  assert.ok(denial(run(readFile(cwd, { path: zedPath(cwd, 'over-limit.ts'), start_line: 1, end_line: 200 }))));
  assert.ok(denial(run(readFile(cwd, { path: zedPath(cwd, 'over-limit.ts'), start_line: 10 }))));
});

test('the outline limit counts characters as Zed holds them, with \\r\\n read as \\n', () => {
  // 16 500 bytes on disk, but 16 243 in the buffer once Zed has turned each \r\n into \n.
  const cwd = project({});
  sized(cwd, 'windows.ts', 16500, '\r\n');
  assert.ok(denial(run(readFile(cwd, { path: zedPath(cwd, 'windows.ts') }))));
});

test('an input it cannot read changes nothing', () => {
  assert.deepEqual(run('not json'), {});
});
