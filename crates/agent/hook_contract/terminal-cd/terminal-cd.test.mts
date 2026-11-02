// Tests of the terminal working directory hook: `node --test terminal-cd.test.mts`.
// Each test runs the real script as Zed does, with its input on stdin.
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const script = path.join(here, 'terminal-cd.mts');

function run(input: object | string): any {
  const result = spawnSync(process.execPath, [script], {
    input: typeof input === 'string' ? input : JSON.stringify(input),
    encoding: 'utf8',
  });
  assert.equal(result.status, 0, result.stderr);
  return result.stdout.trim() ? JSON.parse(result.stdout) : {};
}

/** A PreToolUse input for a tool, from the project rooted at C:/work/clone. */
function call(toolName: string, toolInput: object): object {
  return {
    session_id: 'root-1',
    transcript_path: null,
    cwd: 'C:/work/clone',
    hook_event_name: 'PreToolUse',
    zed: { contract_version: '0.1.0', thread_id: 'thread-1', is_subagent: false },
    tool_name: toolName,
    tool_input: toolInput,
    tool_use_id: 't1',
  };
}

function updatedInput(output: any): unknown {
  assert.equal(output.hookSpecificOutput?.hookEventName, 'PreToolUse');
  assert.equal(output.hookSpecificOutput?.permissionDecision, undefined, 'the hook decides nothing');
  return output.hookSpecificOutput.updatedInput;
}

// Measured 02/11 (tour 26b): Qwen wrote `cwd` seven times, Zed answered « missing field `cd` », and the
// agent concluded its terminal was broken. On 01/11 (tour 19) it wrote `c`.
test('a working directory written as `cwd` becomes `cd`', () => {
  const output = run(call('terminal', { command: 'pwd', cwd: 'C:\\work\\clone' }));
  assert.deepEqual(updatedInput(output), { command: 'pwd', cd: 'C:\\work\\clone' });
});

test('a working directory written as `c` becomes `cd`', () => {
  const output = run(call('terminal', { command: 'git status', c: '.' }));
  assert.deepEqual(updatedInput(output), { command: 'git status', cd: '.' });
});

test('a call without any working directory runs at the project root', () => {
  const output = run(call('terminal', { command: 'echo hello' }));
  assert.deepEqual(updatedInput(output), { command: 'echo hello', cd: '.' });
});

test('the other fields of the call are kept', () => {
  const output = run(call('terminal', { command: 'npm test', cwd: '.', timeout_ms: 60000, tail_lines: 40 }));
  assert.deepEqual(updatedInput(output), { command: 'npm test', cd: '.', timeout_ms: 60000, tail_lines: 40 });
});

test('the sandboxed terminal is corrected the same way', () => {
  const output = run(call('sandboxed_terminal', { command: 'ls', cwd: '.', reason: 'list' }));
  assert.deepEqual(updatedInput(output), { command: 'ls', cd: '.', reason: 'list' });
});

test('a call that already has `cd` is left alone', () => {
  assert.deepEqual(run(call('terminal', { command: 'pwd', cd: '.' })), {});
});

test('other tools are left alone', () => {
  assert.deepEqual(run(call('read_file', { path: 'clone/a.ts', cwd: '.' })), {});
});

test('an input it cannot read changes nothing', () => {
  assert.deepEqual(run('not json'), {});
  assert.deepEqual(run(call('terminal', 'not an object' as unknown as object)), {});
});
