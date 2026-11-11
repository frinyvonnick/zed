// Tests of the test-command-alone hook: `node --test test-command-alone.test.mts`.
// Each test runs the real script as Zed does, with its input on stdin and the list of prefixes
// allowed to leave the sandbox in ZED_EVAL_MXC_UNSANDBOXED.
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const script = path.join(here, 'test-command-alone.mts');
const PREFIX = 'node --experimental-strip-types scripts/run-scripts-tests.ts';
const LIST = `git commit;git merge;npm run test:scripts:hooks;${PREFIX}`;

function run(input: object | string, list: string | null = LIST): any {
  const env = { ...process.env };
  delete env.ZED_EVAL_MXC_UNSANDBOXED;
  if (list !== null) env.ZED_EVAL_MXC_UNSANDBOXED = list;
  const result = spawnSync(process.execPath, [script], {
    input: typeof input === 'string' ? input : JSON.stringify(input),
    env,
    encoding: 'utf8',
  });
  assert.equal(result.status, 0, result.stderr);
  return result.stdout.trim() ? JSON.parse(result.stdout) : {};
}

function terminal(command: string, toolName = 'terminal'): object {
  return {
    session_id: 'root-1',
    transcript_path: null,
    cwd: 'C:/work/clone',
    hook_event_name: 'PreToolUse',
    zed: { contract_version: '0.1.0', thread_id: 'thread-1', is_subagent: false },
    tool_name: toolName,
    tool_input: { command, cd: '.' },
    tool_use_id: 't1',
  };
}

function reason(output: any): string {
  assert.equal(output.hookSpecificOutput?.hookEventName, 'PreToolUse');
  assert.equal(output.hookSpecificOutput?.permissionDecision, 'deny');
  return output.hookSpecificOutput.permissionDecisionReason;
}

// Measured 11/11 (run N11111529-RDPD-a, requests 36 and 44): composed, the command stayed in the
// sandbox, node could not start (0xC0000142) and the agent hunted an antivirus for 18 requests.
test('the bare prefix runs as it is', () => {
  assert.deepEqual(run(terminal(PREFIX)), {});
});

test('the prefix with its arguments runs as it is', () => {
  assert.deepEqual(run(terminal(`${PREFIX} --only hooks`)), {});
});

test('a chained command that holds the prefix is refused, with the exact message', () => {
  const text = reason(run(terminal(`git add -A && ${PREFIX} --only hooks`)));
  assert.equal(text, `This command must run alone to leave the sandbox: run exactly \`${PREFIX} …\` with no \`&&\`, \`;\`, pipe or redirection. Run any other step (git add, reading the output) as a separate command. Inside the sandbox node cannot start (error 0xC0000142).`);
});

test('a redirected command that holds the prefix is refused', () => {
  reason(run(terminal(`${PREFIX} > tmp.log 2>&1`)));
});

test('each chaining, substitution, redirection and line break form is refused', () => {
  for (const tail of [' || true', '; echo done', ' | tail -5', ' < in.txt', ' $(echo x)', '\necho x']) {
    reason(run(terminal(`${PREFIX}${tail}`)));
  }
});

test('a command that holds a prefix without starting with it is refused', () => {
  reason(run(terminal(`cd scripts && git commit -m x`)));
  reason(run(terminal(`echo go ${PREFIX}`)));
});

test('another command runs as it is', () => {
  assert.deepEqual(run(terminal('git status')), {});
  assert.deepEqual(run(terminal('git add -A && git diff --stat')), {});
});

test('a prefix that is only the start of a longer word is not the prefix', () => {
  assert.deepEqual(run(terminal('git committed-files')), {});
});

test('another tool runs as it is', () => {
  assert.deepEqual(run(terminal(`git add -A && ${PREFIX}`, 'read_file')), {});
});

test('without the list nothing is refused', () => {
  assert.deepEqual(run(terminal(`git add -A && ${PREFIX}`), null), {});
  assert.deepEqual(run(terminal(`git add -A && ${PREFIX}`), ''), {});
});

test('an input it cannot read gives no output', () => {
  assert.deepEqual(run('not json'), {});
});
