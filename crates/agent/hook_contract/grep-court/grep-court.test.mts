// Tests of the short grep hook: `node --test grep-court.test.mts`.
// Each test runs the real script as Zed does, with its input on stdin.
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const script = path.join(here, 'grep-court.mts');

function dataDir(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'grep-court-'));
}

function run(input: object | string, data: string): any {
  const result = spawnSync(process.execPath, [script], {
    input: typeof input === 'string' ? input : JSON.stringify(input),
    env: { ...process.env, ZED_HOOK_DATA_DIR: data },
    encoding: 'utf8',
  });
  assert.equal(result.status, 0, result.stderr);
  return result.stdout.trim() ? JSON.parse(result.stdout) : {};
}

function replacements(output: any): { tool_use_id: string; text: string }[] {
  return output.hookSpecificOutput?.tool_result_replacements ?? [];
}

/** A BeforeModel input whose new messages are `messages`, after `known` known ones. */
function beforeModel(messages: object[], known = 0, threadId = 'thread-1'): object {
  return {
    session_id: 'root-1',
    transcript_path: null,
    cwd: here,
    hook_event_name: 'BeforeModel',
    zed: { contract_version: '0.1.0', thread_id: threadId, is_subagent: false },
    request_id: '00000000-0000-4000-8000-000000000000',
    intent: 'thread',
    model: 'vllm/qwen3.8-27b',
    is_compaction: false,
    new_messages: messages,
    known_message_count: known,
    total_message_count: known + messages.length + 1,
    max_output_tokens: 8192,
    tool_names: ['grep', 'read_file'],
    skill_names: [],
  };
}

function toolCall(id: string, name: string): object {
  return { role: 'assistant', content: [{ type: 'tool_use', id, name, input: { regex: 'needle' } }] };
}

function toolResult(id: string, name: string, text: string, isError = false): object {
  return { role: 'user', content: [{ type: 'tool_result', tool_use_id: id, tool_name: name, is_error: isError, text }] };
}

/** A grep output in the shape of crates/agent/src/tools/grep_tool.rs, `lines` lines per match. */
function grepOutput(matches: [string, number][], lines: number, header?: string): string {
  let output = '';
  let written = '';
  for (const [file, line] of matches) {
    if (file !== written) {
      output += `\n## Matches in ${file}\n`;
      written = file;
    }
    output += `\n### fn handler › L${line}-${line + lines - 1}\n\`\`\`\n`;
    output += Array.from({ length: lines }, (_, i) => `    let needle_${i} = compute(${line + i});`).join('\n');
    output += '\n```\n';
  }
  return `${header ?? `Found ${matches.length} matches:`}\n${output}`;
}

const BIG_MATCHES: [string, number][] = [
  ['proxy/src/a.ts', 10], ['proxy/src/a.ts', 40], ['proxy/src/b.ts', 5],
  ['proxy/src/c.ts', 7], ['proxy/src/c.ts', 70], ['proxy/src/c.ts', 120],
];

test('a small grep result is left as it is', () => {
  const small = grepOutput([['proxy/src/a.ts', 10]], 3);
  const output = run(beforeModel([toolCall('g1', 'grep'), toolResult('g1', 'grep', small)]), dataDir());
  assert.deepEqual(replacements(output), []);
});

test('a big grep result becomes a sorted summary with every match line, the first lines and a way to narrow it', () => {
  const big = grepOutput(BIG_MATCHES, 40);
  assert.ok(big.length > 6000);
  const [replacement, ...others] = replacements(
    run(beforeModel([toolCall('g1', 'grep'), toolResult('g1', 'grep', big)]), dataDir()),
  );
  assert.equal(others.length, 0);
  assert.equal(replacement.tool_use_id, 'g1');
  const text = replacement.text;
  assert.ok(text.length < big.length / 2, `${text.length} of ${big.length}`);
  assert.ok(text.startsWith('Found 6 matches:'), text);
  // Sorted by number of matches, then path; each match keeps its lines.
  const summary = text.indexOf('proxy/src/c.ts (3): L7-46, L70-109, L120-159');
  const a = text.indexOf('proxy/src/a.ts (2): L10-49, L40-79');
  const b = text.indexOf('proxy/src/b.ts (1): L5-44');
  assert.ok(summary >= 0 && summary < a && a < b, text);
  // The first lines of the original result.
  assert.ok(text.includes(big.split('\n').slice(0, 12).join('\n')), text);
  // The way to get the rest.
  assert.match(text, /read_file/);
  assert.match(text, /include_pattern/);
});

test('the pagination line of a big result is kept', () => {
  const header = 'Showing matches 1-6 (there were more matches found; use offset: 20 to see next page):';
  const big = grepOutput(BIG_MATCHES, 40, header);
  const [replacement] = replacements(
    run(beforeModel([toolCall('g1', 'grep'), toolResult('g1', 'grep', big)]), dataDir()),
  );
  assert.ok(replacement.text.startsWith(header), replacement.text);
});

test('other tools and failed greps are left as they are', () => {
  const big = grepOutput(BIG_MATCHES, 40);
  const output = run(beforeModel([
    toolCall('r1', 'read_file'), toolResult('r1', 'read_file', big),
    toolCall('g1', 'grep'), toolResult('g1', 'grep', big, true),
  ]), dataDir());
  assert.deepEqual(replacements(output), []);
});

test('the next requests replace the same result with the same text', () => {
  const data = dataDir();
  const big = grepOutput(BIG_MATCHES, 40);
  const first = replacements(run(beforeModel([toolCall('g1', 'grep'), toolResult('g1', 'grep', big)]), data));
  const second = replacements(run(beforeModel([{ role: 'assistant', content: [{ type: 'text', text: 'ok' }] }], 2), data));
  const third = replacements(run(beforeModel([], 3), data));
  assert.equal(first.length, 1);
  assert.deepEqual(second, first);
  assert.deepEqual(third, first);
});

test('a result no longer in the known messages is not replaced again', () => {
  const data = dataDir();
  const big = grepOutput(BIG_MATCHES, 40);
  run(beforeModel([{ role: 'user', content: [{ type: 'text', text: 'go' }] }, toolCall('g1', 'grep'), toolResult('g1', 'grep', big)]), data);
  // The history changed from its second message on (a compaction, an edited message).
  const output = run(beforeModel([{ role: 'user', content: [{ type: 'text', text: 'summary' }] }], 1), data);
  assert.deepEqual(replacements(output), []);
});

test('each thread keeps its own results', () => {
  const data = dataDir();
  const big = grepOutput(BIG_MATCHES, 40);
  run(beforeModel([toolCall('g1', 'grep'), toolResult('g1', 'grep', big)], 0, 'thread-1'), data);
  const other = run(beforeModel([], 2, 'thread-2'), data);
  assert.deepEqual(replacements(other), []);
});

test('a result made of a few very long lines is shortened too', () => {
  const minified = `Found 1 matches:\n\n## Matches in web/app.min.js\n\n### L1\n\`\`\`\n${'var needle=1;'.repeat(600)}\n\`\`\`\n`;
  const [replacement] = replacements(
    run(beforeModel([toolCall('g1', 'grep'), toolResult('g1', 'grep', minified)]), dataDir()),
  );
  assert.ok(replacement.text.length < minified.length / 2, `${replacement.text.length} of ${minified.length}`);
  assert.match(replacement.text, /web\/app\.min\.js \(1\): L1/);
});

test('the kept lines never leave a code block open', () => {
  const big = grepOutput(BIG_MATCHES, 40);
  const [replacement] = replacements(
    run(beforeModel([toolCall('g1', 'grep'), toolResult('g1', 'grep', big)]), dataDir()),
  );
  const fences = replacement.text.split('\n').filter((line) => line.startsWith('```')).length;
  assert.equal(fences % 2, 0, replacement.text);
});

test('lines inside a snippet are not taken for matches', () => {
  const docs = grepOutput(BIG_MATCHES, 40).replace('    let needle_0 = compute(10);', '### L999');
  const [replacement] = replacements(
    run(beforeModel([toolCall('g1', 'grep'), toolResult('g1', 'grep', docs)]), dataDir()),
  );
  const list = replacement.text.slice(replacement.text.indexOf('Matches by file'));
  assert.match(list, /proxy\/src\/a\.ts \(2\): L10-49, L40-79$/m);
  assert.doesNotMatch(list, /L999/);
});

test('a state it cannot save or read back still replaces', () => {
  const big = grepOutput(BIG_MATCHES, 40);
  const notADir = path.join(dataDir(), 'file');
  fs.writeFileSync(notADir, '');
  assert.equal(replacements(run(beforeModel([toolCall('g1', 'grep'), toolResult('g1', 'grep', big)]), notADir)).length, 1);

  const data = dataDir();
  run(beforeModel([], 0), data);
  for (const name of fs.readdirSync(data)) fs.writeFileSync(path.join(data, name), '{}');
  assert.equal(replacements(run(beforeModel([toolCall('g1', 'grep'), toolResult('g1', 'grep', big)]), data)).length, 1);
});

test('an input it cannot read changes nothing', () => {
  assert.deepEqual(run('not json', dataDir()), {});
});
