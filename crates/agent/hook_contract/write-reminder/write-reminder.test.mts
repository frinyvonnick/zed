// Tests of the write reminder hook: `node --test write-reminder.test.mts`.
// Each test runs the real script as Zed does, with its input on stdin.
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const script = path.join(here, 'write-reminder.mts');

const GOAL = 'Write the failing test for story S1 in v2/src/goal.test.ts, then stop.';
const AFTER_COMPACTION = 'The facts you read are in your summary above: do not read those files again; write the next file now.';

function run(input: object | string, data: string): any {
  const result = spawnSync(process.execPath, [script], {
    input: typeof input === 'string' ? input : JSON.stringify(input),
    env: { ...process.env, ZED_HOOK_DATA_DIR: data },
    encoding: 'utf8',
  });
  assert.equal(result.status, 0, result.stderr);
  return result.stdout.trim() ? JSON.parse(result.stdout) : {};
}

function userText(text: string): object {
  return { role: 'user', content: [{ type: 'text', text }] };
}

let callId = 0;

/** The new messages of one model request that follows a call to `tool`. */
function afterTool(tool: string, isError = false): object[] {
  const id = `call-${++callId}`;
  return [
    { role: 'assistant', content: [{ type: 'tool_use', id, name: tool, input: {} }] },
    { role: 'user', content: [{ type: 'tool_result', tool_use_id: id, tool_name: tool, is_error: isError, text: 'ok' }] },
  ];
}

interface Usage {
  input_tokens: number;
  output_tokens: number;
  cache_creation_input_tokens: number;
  cache_read_input_tokens: number;
}

/** One thread as Zed shows it to the hook. */
class Thread {
  readonly id: string;
  readonly data: string;

  constructor(id = 'thread-1', data?: string) {
    this.id = id;
    this.data = data ?? fs.mkdtempSync(path.join(os.tmpdir(), 'write-reminder-'));
  }

  private common(event: string): object {
    return {
      session_id: 'root-1',
      transcript_path: null,
      cwd: here,
      hook_event_name: event,
      zed: { contract_version: '0.1.0', thread_id: this.id, is_subagent: true },
    };
  }

  /** A BeforeModel with these new messages; returns its ephemeral reminders. */
  request(newMessages: object[], extra: object = {}): string[] {
    const output = run({
      ...this.common('BeforeModel'),
      request_id: '00000000-0000-4000-8000-000000000000',
      intent: 'tool_results',
      model: { provider: 'openai_compatible', id: 'qwen3.8-27b' },
      is_compaction: false,
      new_messages: newMessages,
      known_message_count: 0,
      total_message_count: newMessages.length + 1,
      max_output_tokens: 8192,
      tool_names: ['edit_file', 'grep', 'read_file', 'write_file'],
      skill_names: [],
      ...extra,
    }, this.data);
    return output.hookSpecificOutput?.ephemeral_messages ?? [];
  }

  /** `count` requests that each follow a read, with the reminders of each. */
  reads(count: number): string[][] {
    return Array.from({ length: count }, () => this.request(afterTool('read_file')));
  }

  /** An AfterModel with this usage, as the provider reported it. */
  respond(usage: Usage, extra: object = {}): any {
    return run({
      ...this.common('AfterModel'),
      request_id: '00000000-0000-4000-8000-000000000000',
      intent: 'tool_results',
      model: { provider: 'openai_compatible', id: 'qwen3.8-27b' },
      is_compaction: false,
      stop_reason: 'tool_use',
      usage,
      duration_ms: 1200,
      ...extra,
    }, this.data);
  }

  compact(): string | undefined {
    const output = run({ ...this.common('PostCompact'), trigger: 'auto', compact_summary: 'Summary: read a.ts and b.ts.' }, this.data);
    assert.ok(!output.hookSpecificOutput || output.hookSpecificOutput.hookEventName === 'PostCompact');
    return output.hookSpecificOutput?.zed?.append_to_summary;
  }
}

function usage(input: number, output: number, cacheRead = 0, cacheCreation = 0): Usage {
  return { input_tokens: input, output_tokens: output, cache_read_input_tokens: cacheRead, cache_creation_input_tokens: cacheCreation };
}

const nudge = (count: number) => `You have made ${count} model requests since your last write. `
  + 'Do you already have what you need to write the next file? If yes, write it now; '
  + 'read again only the few lines the next edit needs.';

const windowLevel = (thousands: number, percent: number) => `Context window: ${thousands}k of 123k tokens (${percent} %). `
  + 'Around 111k, Zed compacts the conversation and the files you read are dropped from it. '
  + 'Write your next file before that; read again only the lines it needs.';

test('the eighth model request since the start reminds the model to write', () => {
  const thread = new Thread();
  const first = thread.request([userText(GOAL)]);
  const next = thread.reads(7);
  assert.deepEqual([first, ...next.slice(0, 6)], Array(7).fill([]));
  // No usage measured yet: nothing is said of the window.
  assert.deepEqual(next[6], [nudge(8)]);
});

test('the reminder comes back every eight requests, not in between', () => {
  const thread = new Thread();
  thread.reads(24).forEach((texts, index) => {
    const count = index + 1;
    assert.deepEqual(texts, count % 8 === 0 ? [nudge(count)] : [], `request ${count}`);
  });
});

test('the reminder tells the context size of the last response, from its usage', () => {
  const thread = new Thread();
  thread.reads(7);
  // 40 000 prompt tokens and 1 000 generated: 41k, 33 % of 122 880.
  thread.respond(usage(40000, 1000));
  assert.deepEqual(thread.reads(1), [[`${nudge(8)} ${windowLevel(41, 33)}`]]);
});

test('cached prompt tokens are counted once', () => {
  // Zed's OpenAI mapping takes the cached tokens out of input_tokens (crates/open_ai/src/completion.rs:1485-1488).
  const thread = new Thread();
  thread.reads(7);
  thread.respond(usage(5000, 1000, 35000));
  assert.deepEqual(thread.reads(1), [[`${nudge(8)} ${windowLevel(41, 33)}`]]);
  // A provider that writes to its cache reports those tokens apart too.
  const other = new Thread('thread-2');
  other.reads(7);
  other.respond(usage(5000, 1000, 30000, 5000));
  assert.deepEqual(other.reads(1), [[`${nudge(8)} ${windowLevel(41, 33)}`]]);
});

test('a successful write starts the count again', () => {
  const thread = new Thread();
  thread.reads(7);
  for (const tool of ['edit_file', 'write_file', 'create_directory', 'copy_path', 'move_path', 'delete_path']) {
    assert.deepEqual(thread.request(afterTool(tool)), [], tool);
    // The request after the write is the first since it.
    const next = thread.reads(7);
    assert.deepEqual(next.slice(0, 6), Array(6).fill([]), tool);
    assert.deepEqual(next[6], [nudge(8)], tool);
  }
});

test('a failed write does not start the count again', () => {
  const thread = new Thread();
  thread.reads(6);
  assert.deepEqual(thread.request(afterTool('edit_file', true)), []);
  assert.deepEqual(thread.reads(1), [[nudge(8)]]);
});

test('beyond 70 % of the window, every request without a write tells the level', () => {
  const thread = new Thread();
  thread.reads(2);
  // 90 000 tokens: 73 % of 122 880.
  thread.respond(usage(89000, 1000));
  assert.deepEqual(thread.reads(1), [[windowLevel(90, 73)]]);
  assert.deepEqual(thread.reads(1), [[windowLevel(90, 73)]]);
  // The request that follows a write gets nothing.
  assert.deepEqual(thread.request(afterTool('write_file')), []);
  assert.deepEqual(thread.reads(1), [[windowLevel(90, 73)]]);
});

test('below 70 %, requests between two nudges get nothing', () => {
  const thread = new Thread();
  thread.reads(2);
  thread.respond(usage(80000, 1000));
  assert.deepEqual(thread.reads(3), [[], [], []]);
});

test('after a compaction, the level waits for the next measured response', () => {
  const thread = new Thread();
  thread.reads(2);
  thread.respond(usage(100000, 1000));
  thread.compact();
  // The compaction request itself reports the old, full context: it is not a measure of the new one.
  thread.respond(usage(108000, 2000), { is_compaction: true });
  const next = thread.reads(6);
  assert.deepEqual(next.slice(0, 5), Array(5).fill([]));
  assert.deepEqual(next[5], [nudge(8)]);
  thread.respond(usage(24000, 600));
  assert.deepEqual(thread.reads(8)[7], [`${nudge(16)} ${windowLevel(25, 20)}`]);
});

test('a retry of the same request is not counted and keeps its reminder', () => {
  const thread = new Thread();
  thread.reads(7);
  const [reminder] = thread.reads(1);
  assert.deepEqual(reminder, [nudge(8)]);
  // Zed calls BeforeModel again on a retry, with no message it has not sent yet.
  assert.deepEqual(thread.request([]), reminder);
  assert.deepEqual(thread.reads(7).slice(0, 6), Array(6).fill([]));
});

test('compaction and summary requests are not counted', () => {
  const thread = new Thread();
  thread.reads(7);
  thread.request(afterTool('read_file'), { is_compaction: true });
  thread.request(afterTool('read_file'), { intent: 'thread_summarization' });
  assert.deepEqual(thread.reads(1), [[nudge(8)]]);
});

test('each compaction appends the initial goal word for word, then the instruction to write', () => {
  const thread = new Thread();
  thread.request([userText(GOAL), ...afterTool('read_file')]);
  // Later user messages, the summary among them, do not change the goal.
  thread.request([userText('Summary: read a.ts and b.ts.')]);
  for (const compaction of [1, 2]) {
    const text = thread.compact();
    assert.ok(text, `compaction ${compaction}`);
    assert.ok(text.includes(GOAL), text);
    assert.ok(text.indexOf(GOAL) < text.indexOf(AFTER_COMPACTION), text);
    assert.ok(text.endsWith(AFTER_COMPACTION), text);
    assert.doesNotMatch(text, /Summary: read a\.ts/);
  }
});

test('a compaction without a known goal still appends the instruction to write', () => {
  assert.equal(new Thread('unknown-thread').compact(), AFTER_COMPACTION);
});

test('each thread keeps its own count, goal and level', () => {
  const first = new Thread('thread-1');
  const second = new Thread('thread-2', first.data);
  first.request([userText(GOAL)]);
  first.reads(6);
  first.respond(usage(40000, 1000));
  second.request([userText('Another goal.')]);
  assert.deepEqual(second.reads(1), [[]]);
  assert.deepEqual(second.reads(6)[5], [nudge(8)]);
  assert.deepEqual(first.reads(1), [[`${nudge(8)} ${windowLevel(41, 33)}`]]);
  assert.match(second.compact() ?? '', /Another goal\./);
});

test('the state file stays a few numbers and the goal, however long the thread', () => {
  const thread = new Thread();
  thread.request([userText(GOAL)]);
  thread.request(Array.from({ length: 50 }, () => afterTool('read_file')).flat());
  thread.respond(usage(40000, 1000));
  const [file] = fs.readdirSync(thread.data);
  assert.ok(fs.statSync(path.join(thread.data, file)).size < GOAL.length + 400);
});

test('the hook adds less than 50 ms to node start-up on a request of 200 messages', () => {
  const data = new Thread().data;
  const messages = Array.from({ length: 100 }, () => afterTool('read_file')).flat()
    .map((message: any) => ({ ...message, content: message.content.map((part: any) => part.type === 'tool_result' ? { ...part, text: 'x'.repeat(4000) } : part) }));
  const input = JSON.stringify({
    session_id: 'root-1', transcript_path: null, cwd: here, hook_event_name: 'BeforeModel',
    zed: { contract_version: '0.1.0', thread_id: 'timed', is_subagent: true },
    request_id: '00000000-0000-4000-8000-000000000000', intent: 'tool_results',
    model: { provider: 'openai_compatible', id: 'qwen3.8-27b' }, is_compaction: false,
    new_messages: [userText(GOAL), ...messages], known_message_count: 0, total_message_count: 202,
    max_output_tokens: 8192, tool_names: ['read_file'], skill_names: [],
  });
  const median = (input: string) => {
    const times = Array.from({ length: 5 }, () => {
      const start = performance.now();
      run(input, data);
      return performance.now() - start;
    }).sort((a, b) => a - b);
    return times[2];
  };
  // An unreadable input measures node start-up alone.
  const startUp = median('not json');
  const work = median(input) - startUp;
  assert.ok(work < 50, `${Math.round(work)} ms over a start-up of ${Math.round(startUp)} ms`);
});

test('an input it cannot read changes nothing', () => {
  assert.deepEqual(run('not json', new Thread().data), {});
});
