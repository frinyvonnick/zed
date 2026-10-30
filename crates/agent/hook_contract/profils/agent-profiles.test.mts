// Tests of the agent profiles hook: `node --test agent-profiles.test.mts`.
// Each test runs the real script as Zed does, with its input on stdin.
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const script = path.join(here, 'agent-profiles.mts');
const agents = path.join(here, 'test-agents');

const ROOT_TOOLS = [
  'diagnostics', 'edit_file', 'fetch', 'find_path', 'grep', 'list_directory',
  'read_file', 'skill', 'spawn_agent', 'terminal', 'write_file',
];

function dataDir(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'agent-profiles-'));
}

function run(
  input: object,
  data: string,
  agentDirs: string[] = [agents],
): { output: any; stderr: string } {
  const result = spawnSync(
    process.execPath,
    [script, ...agentDirs.flatMap((dir) => ['--agents', dir])],
    { input: JSON.stringify(input), env: { ...process.env, ZED_HOOK_DATA_DIR: data }, encoding: 'utf8' },
  );
  assert.equal(result.status, 0, result.stderr);
  return { output: result.stdout.trim() ? JSON.parse(result.stdout) : {}, stderr: result.stderr };
}

/** A temporary directory holding `files`, by relative path. */
function tempDir(files: Record<string, string>): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'agent-profiles-files-'));
  for (const [file, text] of Object.entries(files)) {
    fs.mkdirSync(path.dirname(path.join(dir, file)), { recursive: true });
    fs.writeFileSync(path.join(dir, file), text);
  }
  return dir;
}

function testAgent(name: string): string {
  return fs.readFileSync(path.join(agents, `${name}.md`), 'utf8');
}

function common(event: string, threadId: string, isSubagent: boolean, cwd = here): object {
  return {
    session_id: 'root-1',
    transcript_path: null,
    cwd,
    hook_event_name: event,
    zed: {
      contract_version: '0.1.0',
      thread_id: threadId,
      parent_thread_id: isSubagent ? 'root-1' : null,
      is_subagent: isSubagent,
      depth: isSubagent ? 1 : 0,
      timestamp: '2026-10-30T10:12:03.418Z',
    },
  };
}

function beforeModel(threadId: string, isSubagent: boolean, extra: object = {}): object {
  return {
    ...common('BeforeModel', threadId, isSubagent),
    request_id: 'request-1',
    intent: 'user_prompt',
    model: { provider: 'fake', id: 'fake' },
    is_compaction: false,
    new_messages: [],
    known_message_count: 0,
    total_message_count: 2,
    max_output_tokens: 1000,
    tool_names: ROOT_TOOLS,
    skill_names: ['create-skill', 'exploring-code', 'tdd'],
    ...extra,
  };
}

function spawnAgent(label: string): object {
  return {
    ...common('PreToolUse', 'root-1', false),
    tool_name: 'spawn_agent',
    tool_input: { label, message: 'Find where the token is read.' },
    tool_use_id: 'tool-1',
  };
}

function subagentStart(agentId: string, label: string): object {
  return { ...common('SubagentStart', 'root-1', false), agent_id: agentId, agent_type: label };
}

/**
 * Starts a subagent as Zed would after an allowed spawn_agent: the token of
 * its label is recorded as issued first, as the PreToolUse that allowed it does.
 */
function start(agentId: string, label: string, data: string, agentDirs: string[] = [agents]) {
  issue(label, data);
  return run(subagentStart(agentId, label), data, agentDirs);
}

/** Records the `[name#token]` of a label as issued. */
function issue(label: string, data: string): void {
  const token = /^\[([^\]]+)\]/.exec(label)![1];
  fs.mkdirSync(path.join(data, 'tokens'), { recursive: true });
  fs.writeFileSync(path.join(data, 'tokens', token), '');
}

function toolCall(threadId: string, tool: string, input: object): object {
  return { ...common('PreToolUse', threadId, true), tool_name: tool, tool_input: input, tool_use_id: 'tool-9' };
}

test('the root conversation gets the pilote profile and its subagent list', () => {
  const { output } = run(beforeModel('root-1', false), dataDir());
  const specific = output.hookSpecificOutput;
  assert.equal(specific.hookEventName, 'BeforeModel');
  const insert: string = specific.system_prompt_insert;
  assert.ok(insert.startsWith('## Agent Profile\n'), insert);
  assert.ok(insert.includes('Tu es le pilote.'), insert);
  assert.ok(
    insert.includes('\nSubagent profiles you may launch (start the spawn_agent label with [name]):\n'),
    insert,
  );
  assert.ok(insert.includes('\n- explorateur: Cherche dans le dépôt'), insert);
  assert.ok(insert.includes('\n- planificateur: '), insert);
  assert.ok(insert.includes('\n- general: '), insert);
  assert.deepEqual(specific.remove_tools, ['diagnostics', 'fetch']);
  // pilote lists no skills: it may use them all.
  assert.equal(specific.remove_skills, undefined);
});

test('the output does not depend on the order of the names Zed sends', () => {
  const data = dataDir();
  start('sub-6', '[explorateur#1a2b] cherche', data);
  const names = { skill_names: ['create-skill', 'exploring-code', 'tdd'] };
  const first = run(beforeModel('sub-6', true, names), data).output;
  const second = run(beforeModel('sub-6', true, {
    tool_names: [...ROOT_TOOLS].reverse(),
    skill_names: [...names.skill_names].reverse(),
  }), data).output;
  assert.deepEqual(first, second);
});

test('the compaction request is left alone', () => {
  const { output } = run(beforeModel('root-1', false, { is_compaction: true }), dataDir());
  assert.deepEqual(output, {});
});

test('a spawn without a profile is denied with the allowed list', () => {
  const { output } = run(spawnAgent('où est lu le jeton ?'), dataDir());
  assert.equal(output.hookSpecificOutput.permissionDecision, 'deny');
  assert.equal(
    output.hookSpecificOutput.permissionDecisionReason,
    'Start the label with an allowed profile: [explorateur], [planificateur], [general].',
  );
});

test('a spawn with a profile outside the list is denied', () => {
  const { output } = run(spawnAgent('[pirate] où est lu le jeton ?'), dataDir());
  assert.equal(output.hookSpecificOutput.permissionDecision, 'deny');
});

test('an allowed spawn gets a unique token and the profile model', () => {
  const { output } = run(spawnAgent('[explorateur] où est lu le jeton ?'), dataDir());
  const specific = output.hookSpecificOutput;
  assert.equal(specific.permissionDecision, 'allow');
  assert.match(specific.updatedInput.label, /^\[explorateur#[0-9a-f]{4}\] où est lu le jeton \?$/);
  assert.equal(specific.updatedInput.message, 'Find where the token is read.');
  assert.equal(specific.updatedInput.model, 'fake/fake');
});

test('a subagent keeps its profile: tools outside it are removed and refused', () => {
  const data = dataDir();
  // The skills of a profile are allowed, not loaded: nothing joins the
  // subagent's first message.
  const started = start('sub-1', '[explorateur#a1b2] où est lu le jeton ?', data).output;
  assert.deepEqual(started, {});

  const request = run(beforeModel('sub-1', true), data).output.hookSpecificOutput;
  assert.ok(request.system_prompt_insert.includes('Tu es un explorateur en lecture seule.'));
  assert.ok(!request.system_prompt_insert.includes('Subagent profiles you may launch'));
  assert.deepEqual(request.remove_tools, [
    'diagnostics', 'edit_file', 'fetch', 'spawn_agent', 'terminal', 'write_file',
  ]);
  assert.deepEqual(request.remove_skills, ['create-skill', 'tdd']);

  const write = run(toolCall('sub-1', 'write_file', { path: 'notes.md', content: 'x' }), data).output;
  assert.equal(write.hookSpecificOutput.permissionDecision, 'deny');
  assert.equal(
    write.hookSpecificOutput.permissionDecisionReason,
    'The write_file tool is not allowed for the explorateur profile.',
  );

  assert.deepEqual(run(toolCall('sub-1', 'read_file', { path: 'notes.md' }), data).output, {});
});

test('a profile allows only the skills it lists', () => {
  const data = dataDir();
  start('sub-3', '[explorateur#e5f6] où est lu le jeton ?', data);

  assert.deepEqual(run(toolCall('sub-3', 'skill', { name: 'exploring-code' }), data).output, {});
  const skill = run(toolCall('sub-3', 'skill', { name: 'tdd' }), data).output;
  assert.equal(skill.hookSpecificOutput.permissionDecision, 'deny');
  assert.equal(
    skill.hookSpecificOutput.permissionDecisionReason,
    'The tdd skill is not allowed for the explorateur profile.',
  );

  const allowedRead = { path: '/home/me/.agents/skills/exploring-code/SKILL.md' };
  assert.deepEqual(run(toolCall('sub-3', 'read_file', allowedRead), data).output, {});
  const deniedRead = { path: 'C:\\Users\\me\\.agents\\skills\\tdd\\SKILL.md' };
  const read = run(toolCall('sub-3', 'read_file', deniedRead), data).output;
  assert.equal(read.hookSpecificOutput.permissionDecision, 'deny');
  assert.equal(
    read.hookSpecificOutput.permissionDecisionReason,
    'The tdd skill is not allowed for the explorateur profile.',
  );
});

test('an empty skill list allows none; no list allows all', () => {
  const data = dataDir();
  start('sub-4', '[lecteur#0a0b] lis', data);
  const hidden = run(beforeModel('sub-4', true), data).output.hookSpecificOutput;
  assert.deepEqual(hidden.remove_skills, ['create-skill', 'exploring-code', 'tdd']);
  // Without any skill, the skill tool goes too, although the profile lists it.
  assert.ok(hidden.remove_tools.includes('skill'), JSON.stringify(hidden.remove_tools));
  const none = run(toolCall('sub-4', 'skill', { name: 'exploring-code' }), data).output;
  assert.equal(none.hookSpecificOutput.permissionDecision, 'deny');

  start('sub-5', '[general#0c0d] cherche', data);
  assert.deepEqual(run(toolCall('sub-5', 'skill', { name: 'tdd' }), data).output, {});
});

test('a subagent whose profile is unknown has every tool refused', () => {
  const { output } = run(toolCall('sub-unknown', 'read_file', { path: 'notes.md' }), dataDir());
  assert.equal(output.hookSpecificOutput.permissionDecision, 'deny');
  assert.match(output.hookSpecificOutput.permissionDecisionReason, /profile is unknown/);
  const request = run(beforeModel('sub-unknown', true), dataDir()).output;
  assert.match(request.hookSpecificOutput.ephemeral_messages[0], /profile is unknown/);
});

test('fields Zed does not apply yet are named on stderr', () => {
  const { stderr } = start('sub-2', '[planificateur#c3d4] plan', dataDir());
  assert.match(stderr, /planificateur/);
  assert.match(stderr, /effort/);
  assert.match(stderr, /hooks/);
});

const SKILLS = { skill_names: ['create-skill', 'exploring-code', 'tdd'] };

test('folded and literal descriptions do not hide the rest of the header', () => {
  const dir = tempDir({
    'plieur.md': '---\nname: plieur\ndescription: >\n  A profile whose description\n  spans two lines.\nskills: []\nzed:\n  tools: [read_file]\n---\n\nBody.\n',
    'litteral.md': '---\nname: litteral\ndescription: |\n  First line.\n  Second line.\nzed:\n  tools: [grep]\n---\n\nBody.\n',
  });
  const data = dataDir();
  start('sub-7', '[plieur#2b3c] lis', data, [dir]);
  start('sub-8', '[litteral#3c4d] lis', data, [dir]);

  const folded = run(beforeModel('sub-7', true, SKILLS), data, [dir]).output.hookSpecificOutput;
  assert.ok(folded.remove_tools.includes('edit_file'), JSON.stringify(folded));
  assert.ok(!folded.remove_tools.includes('read_file'), JSON.stringify(folded));
  assert.deepEqual(folded.remove_skills, SKILLS.skill_names);
  const literal = run(beforeModel('sub-8', true, SKILLS), data, [dir]).output.hookSpecificOutput;
  assert.ok(literal.remove_tools.includes('read_file'), JSON.stringify(literal));
  assert.ok(!literal.remove_tools.includes('grep'), JSON.stringify(literal));
});

test('a header with a line the reader cannot place is rejected, and said so', () => {
  const dir = tempDir({
    'bancal.md': '---\nname: bancal\ndescription: Broken.\n  stray: line\nzed:\n  tools: [read_file]\n---\n\nBody.\n',
  });
  const data = dataDir();
  const { stderr } = start('sub-9', '[bancal#4d5e] lis', data, [dir]);
  assert.match(stderr, /bancal\.md/);
  assert.match(stderr, /stray: line/);
  const read = run(toolCall('sub-9', 'read_file', { path: 'notes.md' }), data, [dir]).output;
  assert.equal(read.hookSpecificOutput.permissionDecision, 'deny');
});

test('a pilote profile written with a byte order mark still applies', () => {
  const dir = tempDir({
    'pilote.md': '﻿' + testAgent('pilote'),
    'explorateur.md': testAgent('explorateur'),
  });
  const { output } = run(beforeModel('root-1', false), dataDir(), [dir]);
  assert.deepEqual(output.hookSpecificOutput.remove_tools, ['diagnostics', 'fetch']);
});

test('the subagent list shows only where spawn_agent is offered', () => {
  const tools = { tool_names: ROOT_TOOLS.filter((tool) => tool !== 'spawn_agent') };
  const root = run(beforeModel('root-1', false, tools), dataDir()).output.hookSpecificOutput;
  assert.ok(!root.system_prompt_insert.includes('Subagent profiles you may launch'));
  const data = dataDir();
  start('sub-10', '[general#5e6f] cherche', data);
  const general = run(beforeModel('sub-10', true, tools), data).output.hookSpecificOutput;
  assert.ok(!general.system_prompt_insert.includes('Subagent profiles you may launch'));
});

test('a comment at the end of a line is not part of the value', () => {
  const dir = tempDir({
    'commente.md': '---\nname: commente\ndescription: Reads. # short\nskills: [] # none\nzed:\n  tools: [read_file, grep] # read only\n---\n\nBody.\n',
  });
  const data = dataDir();
  start('sub-11', '[commente#6f7a] lis', data, [dir]);
  const request = run(beforeModel('sub-11', true, SKILLS), data, [dir]).output.hookSpecificOutput;
  assert.ok(!request.remove_tools.includes('grep'), JSON.stringify(request));
  assert.ok(request.remove_tools.includes('edit_file'), JSON.stringify(request));
  assert.deepEqual(request.remove_skills, SKILLS.skill_names);
});

test('only the SKILL.md of a skill root names a skill, by its header', () => {
  const home = tempDir({
    '.agents/skills/dossier-x/SKILL.md': '---\nname: tdd\ndescription: Red then green.\n---\n',
    '.agents/skills/tdd/SKILL.md': '---\nname: exploring-code\ndescription: Locate.\n---\n',
  });
  const data = dataDir();
  start('sub-12', '[explorateur#7a8b] cherche', data);
  const read = (file: string) => run(toolCall('sub-12', 'read_file', { path: file }), data).output;

  assert.deepEqual(read('docs/tdd/SKILL.md'), {});
  assert.deepEqual(read(path.join(home, 'other', 'skills', 'tdd', 'SKILL.md')), {});
  const renamed = read(path.join(home, '.agents', 'skills', 'dossier-x', 'SKILL.md'));
  assert.match(renamed.hookSpecificOutput.permissionDecisionReason, /The tdd skill/);
  assert.deepEqual(read(path.join(home, '.agents', 'skills', 'tdd', 'SKILL.md')), {});
  const lowercase = read(path.join(home, '.agents', 'skills', 'autre', 'skill.md'));
  assert.equal(lowercase.hookSpecificOutput.permissionDecision, 'deny');
});

test('a profile without the skill tool hides every skill', () => {
  const dir = tempDir({
    'sansoutil.md': '---\nname: sansoutil\ndescription: No skill tool.\nzed:\n  tools: [read_file]\n---\n\nBody.\n',
  });
  const data = dataDir();
  start('sub-13', '[sansoutil#8b9c] lis', data, [dir]);
  const request = run(beforeModel('sub-13', true, SKILLS), data, [dir]).output.hookSpecificOutput;
  assert.deepEqual(request.remove_skills, SKILLS.skill_names);
});

test('no subagent may open a sibling conversation', () => {
  const data = dataDir();
  start('sub-14', '[general#9c0d] cherche', data);
  const create = run(toolCall('sub-14', 'create_thread', { title: 'x' }), data).output;
  assert.equal(create.hookSpecificOutput.permissionDecision, 'deny');
  const request = run(beforeModel('sub-14', true, { tool_names: ['create_thread', 'read_file'] }), data)
    .output.hookSpecificOutput;
  assert.deepEqual(request.remove_tools, ['create_thread']);
});

test('the fields pilote does not apply are named once per content', () => {
  const data = dataDir();
  const first = run(beforeModel('root-1', false), data).stderr;
  assert.match(first, /pilote/);
  assert.match(first, /effort/);
  assert.equal(run(beforeModel('root-1', false), data).stderr, '');
});

test('a project profile replacing a personal one is said so', () => {
  const project = tempDir({ '.zed/agents/explorateur.md': testAgent('explorateur') });
  const { stderr } = run({ ...beforeModel('root-1', false), cwd: project }, dataDir());
  assert.match(stderr, /explorateur/);
  assert.match(stderr, /replaces/);
});

const READ_TOOLS = ['find_path', 'grep', 'list_directory', 'read_file'];
const NOT_READ = ROOT_TOOLS.filter((tool) => !READ_TOOLS.includes(tool));

/** The test pilote with its `zed:` block replaced by `zed`. */
function piloteWith(zed: string): string {
  return testAgent('pilote').replace(/\r\n/g, '\n').replace(/^zed:\n(?:  .*\n)+/m, zed);
}

/** A pilote that cannot be read leaves the root agent with reading only. */
function assertReadOnlyRoot(dir: string): string {
  const { output, stderr } = run(beforeModel('root-1', false), dataDir(), [dir]);
  const specific = output.hookSpecificOutput;
  assert.deepEqual(specific.remove_tools, NOT_READ, JSON.stringify(output));
  assert.match(specific.ephemeral_messages[0], /pilote\.md/);
  assert.match(stderr, /pilote\.md/);
  return stderr;
}

for (const [form, zed] of [
  ['a table on one line', 'zed: {tools: [read_file, grep]}\n'],
  ['a quoted key', '"zed":\n  tools: [read_file, grep]\n'],
  ['a quoted nested key', "zed:\n  'tools': [read_file, grep]\n"],
  ['a key written twice', 'zed:\n  tools: [read_file]\nzed:\n  tools: [grep]\n'],
  ['a list where a table belongs', 'zed:\n  - read_file\n  - grep\n'],
  ['a table where a list belongs', 'zed:\n  tools: [read_file]\n  disallowed_tools: {terminal: true}\n'],
  ['a model that is not a table', 'zed:\n  tools: [read_file]\n  model: fake/fake\n'],
  ['an empty zed', 'zed:\n'],
] as const) {
  test(`a pilote with ${form} is rejected, and the root agent keeps reading only`, () => {
    assertReadOnlyRoot(tempDir({ 'pilote.md': piloteWith(zed) }));
  });
}

test('without a pilote, the root agent keeps reading only and is told which file to fix', () => {
  const dir = tempDir({ 'explorateur.md': testAgent('explorateur') });
  assertReadOnlyRoot(dir);
  const data = dataDir();
  const rootCall = (tool: string) => ({
    ...common('PreToolUse', 'root-1', false),
    tool_name: tool,
    tool_input: { path: 'a.md' },
    tool_use_id: 'tool-8',
  });
  const edit = run(rootCall('edit_file'), data, [dir]).output;
  assert.equal(edit.hookSpecificOutput.permissionDecision, 'deny');
  assert.match(edit.hookSpecificOutput.permissionDecisionReason, /pilote\.md/);
  const spawn = run(spawnAgent('[explorateur] cherche'), data, [dir]).output;
  assert.equal(spawn.hookSpecificOutput.permissionDecision, 'deny');
  assert.deepEqual(run(rootCall('read_file'), data, [dir]).output, {});

  const noDescription = tempDir({
    'pilote.md': testAgent('pilote').replace(/^description:.*\r?\n/m, ''),
  });
  assert.match(assertReadOnlyRoot(noDescription), /description/);
});

test('a blank session_id is a new launch, checked like any other', () => {
  const blank = { ...spawnAgent('[pilote] anything') } as any;
  blank.tool_input = { ...blank.tool_input, session_id: '   ' };
  const { output } = run(blank, dataDir());
  assert.equal(output.hookSpecificOutput.permissionDecision, 'deny');
});

test('only a token the hook issued starts a profile, once', () => {
  const data = dataDir();
  // A label made up without an allowed spawn_agent.
  run(subagentStart('sub-20', '[general#abcd] anything'), data);
  const forged = run(toolCall('sub-20', 'read_file', { path: 'a.md' }), data).output;
  assert.match(forged.hookSpecificOutput.permissionDecisionReason, /profile is unknown/);

  const spawn = run(spawnAgent('[explorateur] cherche'), data).output;
  const label = spawn.hookSpecificOutput.updatedInput.label;
  run(subagentStart('sub-21', label), data);
  const issued = run(beforeModel('sub-21', true), data).output.hookSpecificOutput;
  assert.ok(issued.remove_tools.includes('write_file'), JSON.stringify(issued));
  // The token is spent: reusing the label starts nothing.
  run(subagentStart('sub-22', label), data);
  const reused = run(toolCall('sub-22', 'read_file', { path: 'a.md' }), data).output;
  assert.match(reused.hookSpecificOutput.permissionDecisionReason, /profile is unknown/);
});

test('when the hook fails, it closes: calls refused, reading only', () => {
  const broken = tempDir({ 'x.md/placeholder.txt': '' });
  const data = dataDir();
  const call = run(toolCall('sub-1', 'read_file', { path: 'a.md' }), data, [broken, agents]).output;
  assert.equal(call.hookSpecificOutput.permissionDecision, 'deny');
  assert.match(call.hookSpecificOutput.permissionDecisionReason, /agent profiles hook failed/);
  const request = run(beforeModel('root-1', false), data, [broken, agents]).output;
  assert.deepEqual(request.hookSpecificOutput.remove_tools, NOT_READ);
  assert.match(request.hookSpecificOutput.ephemeral_messages[0], /agent profiles hook failed/);

  const notAList = run(beforeModel('root-1', false, { tool_names: 'read_file' }), dataDir()).output;
  assert.match(notAList.hookSpecificOutput.ephemeral_messages[0], /agent profiles hook failed/);
});

test('a profile written with CRLF line ends gives no carriage return', () => {
  const dir = tempDir({ 'pilote.md': testAgent('pilote').replace(/\r?\n/g, '\r\n') });
  const { output } = run(beforeModel('root-1', false), dataDir(), [dir]);
  assert.ok(!output.hookSpecificOutput.system_prompt_insert.includes('\r'));
});

test('a top-level key written zed.tools is rejected, and the root agent keeps reading only', () => {
  assertReadOnlyRoot(tempDir({ 'pilote.md': piloteWith('zed.tools: [read_file]\n') }));
});

test('a top-level key the schema does not know is named once', () => {
  const dir = tempDir({
    'pilote.md': testAgent('pilote').replace(/\r\n/g, '\n').replace(/^effort: high\n/m, 'effort: high\ncolour: red\n'),
  });
  const data = dataDir();
  const first = run(beforeModel('root-1', false), data, [dir]);
  assert.match(first.stderr, /colour/);
  assert.ok(first.output.hookSpecificOutput.remove_tools.includes('fetch'));
  assert.doesNotMatch(run(beforeModel('root-1', false), data, [dir]).stderr, /colour/);
});
