// Agent profiles hook for Zed. It applies the profile of each agent, written
// in the format of Claude Code's agents (see README.md next to this file):
// its prompt, its tools, its model and the skills it allows, through the
// generic hook events. Zed knows nothing of profiles.
//
// Declare it as a command hook on BeforeModel, PreToolUse and SubagentStart:
//   node agent-profiles.mts --agents <dir> [--agents <dir>]
//
// - Personal profiles come from the `--agents` directories; project profiles
//   from `<cwd>/.zed/agents`, which may not declare hooks.
// - The root conversation runs the `pilote` profile. Its `zed.tools` entry
//   `spawn_agent(a, b)` lists the profiles it may launch, named at the start
//   of the spawn_agent label: `[explorateur] where is the token read?`.
// - The profile each subagent runs is kept in ZED_HOOK_DATA_DIR, one file per
//   subagent thread, so that it survives a restart of Zed.
// - A profile that cannot be read restricts more, never less: without
//   `pilote`, spawning is refused; a subagent of unknown profile gets no tool.
import { createHash, randomBytes } from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

type Yaml = string | number | boolean | null | Yaml[] | { [key: string]: Yaml };
type YamlMap = { [key: string]: Yaml };

interface Profile {
  name: string;
  description: string;
  body: string;
  header: YamlMap;
  file: string;
  /** The text of its file, so that each of its warnings shows once. */
  text: string;
  isProject: boolean;
  /** Tools the profile allows; undefined allows every tool. */
  tools?: string[];
  disallowedTools: string[];
  /** Profiles it may launch: a list, every profile, or none. */
  subagents: string[] | 'all' | 'none';
  /** The model of its subagents, as `provider/id`. */
  model?: string;
  /** Skills it allows; undefined allows every skill. */
  skills?: string[];
}

const ROOT_PROFILE = 'pilote';

/** The tools an agent keeps when its profile cannot be applied. */
const READ_TOOLS = ['find_path', 'grep', 'list_directory', 'read_file'];

const PROFILE_UNKNOWN =
  "This subagent's profile is unknown (renamed or removed), so its tools are refused.";

function piloteMissing(fix: string): string {
  return `No readable \`pilote\` agent profile (fix ${fix}), so this agent keeps its reading tools only.`;
}

function hookFailed(error: unknown): string {
  return `The agent profiles hook failed (${(error as Error)?.message ?? error}), so this agent keeps its reading tools only.`;
}

/** Profile files rejected while loading, to name the one to fix. */
const rejectedFiles: string[] = [];

// Fields Zed does not apply yet, named on stderr when a profile is used.
const NOT_APPLIED_YET = ['effort', 'omitClaudeMd', 'autoCompactWindow', 'hooks'];
const NOT_APPLIED = [
  'permissionMode', 'memory', 'background', 'isolation', 'color',
  'initialPrompt', 'experimental', 'mcpServers', 'maxTurns',
];

// Warnings. Each one shows once per content, or the log would repeat it on
// every request.

let dataDir = '';

/** Shows a warning unless it was shown for the same content: one file per fingerprint. */
function warnOnce(message: string, content = ''): void {
  const key = createHash('sha256').update(content).update('\0').update(message).digest('hex');
  const dir = path.join(dataDir, 'warnings');
  try {
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(path.join(dir, key), '', { flag: 'wx' });
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'EEXIST') return;
    // Unrecorded: the warning shows again next time.
  }
  process.stderr.write(`agent-profiles: ${message}\n`);
}

// A small YAML reader for profile headers: maps, block and flow lists, lists
// of maps, plain or quoted scalars, `|` and `>` blocks, and comments. A line
// it cannot place rejects the header, rather than cutting it short.

interface Line {
  indent: number;
  /** The text without indentation; empty for a blank or comment line. */
  text: string;
  raw: string;
  number: number;
}

function parseYaml(text: string): YamlMap {
  const lines: Line[] = text.split(/\r?\n/).map((raw, i) => {
    const trimmed = raw.trim();
    return {
      indent: raw.length - raw.trimStart().length,
      text: trimmed.startsWith('#') ? '' : trimmed,
      raw,
      number: i + 1,
    };
  });
  const start = skipBlank(lines, 0);
  if (start === lines.length) return {};
  const [value, end] = parseBlock(lines, start, lines[start].indent);
  const rest = skipBlank(lines, end);
  if (rest < lines.length) {
    throw new Error(`cannot place line ${lines[rest].number}: \`${lines[rest].text}\``);
  }
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error('the header is not a map');
  }
  return value;
}

function skipBlank(lines: Line[], start: number): number {
  let i = start;
  while (i < lines.length && lines[i].text === '') i += 1;
  return i;
}

function isListItem(text: string): boolean {
  return text === '-' || text.startsWith('- ');
}

function parseBlock(lines: Line[], start: number, indent: number): [Yaml, number] {
  return isListItem(lines[start].text)
    ? parseList(lines, start, indent)
    : parseMap(lines, start, indent);
}

function parseMap(lines: Line[], start: number, indent: number): [YamlMap, number] {
  const map: YamlMap = {};
  let i = skipBlank(lines, start);
  while (i < lines.length && lines[i].indent === indent && !isListItem(lines[i].text)) {
    const match = /^([^:]+):(?:\s+(.*))?$/.exec(stripComment(lines[i].text));
    if (!match) throw new Error(`cannot read line ${lines[i].number}: \`${lines[i].text}\``);
    const key = match[1].trim();
    const value = match[2];
    // Forms this reader does not handle reject the header, never shorten it.
    if (/^["']/.test(key)) {
      throw new Error(`quoted keys are not read, line ${lines[i].number}: \`${lines[i].text}\``);
    }
    if (Object.hasOwn(map, key)) {
      throw new Error(`the key \`${key}\` appears twice, line ${lines[i].number}`);
    }
    i += 1;
    if (value !== undefined && /^[|>][+-]?$/.test(value)) {
      [map[key], i] = parseBlockScalar(lines, i, indent, value);
    } else if (value !== undefined) {
      map[key] = parseScalar(value);
    } else {
      const next = skipBlank(lines, i);
      if (
        next < lines.length &&
        (lines[next].indent > indent || (lines[next].indent === indent && isListItem(lines[next].text)))
      ) {
        [map[key], i] = parseBlock(lines, next, lines[next].indent);
      } else {
        map[key] = null;
      }
    }
    i = skipBlank(lines, i);
  }
  return [map, i];
}

function parseList(lines: Line[], start: number, indent: number): [Yaml[], number] {
  const list: Yaml[] = [];
  let i = skipBlank(lines, start);
  while (i < lines.length && lines[i].indent === indent && isListItem(lines[i].text)) {
    const item = stripComment(lines[i].text.slice(1));
    if (item === '') {
      const next = skipBlank(lines, i + 1);
      if (next === lines.length || lines[next].indent <= indent) {
        list.push(null);
        i = next;
      } else {
        let value: Yaml;
        [value, i] = parseBlock(lines, next, lines[next].indent);
        list.push(value);
      }
    } else if (/^[^\s:[\]{}"']+:(\s|$)/.test(item)) {
      // A map item: its first key sits after the dash.
      const itemIndent = lines[i].indent + (lines[i].text.length - lines[i].text.slice(1).trimStart().length);
      const itemLines = [...lines];
      itemLines[i] = { ...lines[i], indent: itemIndent, text: lines[i].text.slice(1).trim() };
      const [map, next] = parseMap(itemLines, i, itemIndent);
      list.push(map);
      i = next;
    } else {
      list.push(parseScalar(item));
      i += 1;
    }
    i = skipBlank(lines, i);
  }
  return [list, i];
}

/** A `|` (literal) or `>` (folded) block: the lines indented under its key. */
function parseBlockScalar(
  lines: Line[],
  start: number,
  keyIndent: number,
  indicator: string,
): [string, number] {
  let end = start;
  while (end < lines.length && (lines[end].raw.trim() === '' || lines[end].indent > keyIndent)) {
    end += 1;
  }
  // Trailing blank lines belong to what follows.
  while (end > start && lines[end - 1].raw.trim() === '') end -= 1;
  const block = lines.slice(start, end);
  const contentIndent = Math.min(
    ...block.filter((line) => line.raw.trim() !== '').map((line) => line.indent),
  );
  const texts = block.map((line) => (line.raw.trim() === '' ? '' : line.raw.slice(contentIndent).trimEnd()));
  let value: string;
  if (indicator.startsWith('|')) {
    value = texts.join('\n');
  } else {
    // Folded: lines join with a space, a blank line keeps a line break.
    value = texts.reduce((folded, text) => {
      if (text === '') return `${folded}\n`;
      return folded === '' || folded.endsWith('\n') ? folded + text : `${folded} ${text}`;
    }, '');
  }
  return [indicator.endsWith('-') ? value : `${value}\n`, end];
}

/** Removes a ` # comment` at the end of a line, outside quotes. */
function stripComment(text: string): string {
  let quote = '';
  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    if (quote) {
      if (char === quote) quote = '';
    } else if (char === '"' || char === "'") {
      quote = char;
    } else if (char === '#' && (i === 0 || /\s/.test(text[i - 1]))) {
      return text.slice(0, i).trim();
    }
  }
  return text.trim();
}

function parseScalar(text: string): Yaml {
  const value = stripComment(text);
  if (value.startsWith('{')) throw new Error(`tables written \`{...}\` are not read: \`${value}\``);
  if (value.startsWith('[') && value.endsWith(']')) {
    return splitTopLevel(value.slice(1, -1)).map(parseScalar);
  }
  if (/^".*"$/.test(value) || /^'.*'$/.test(value)) return value.slice(1, -1);
  if (value === 'true' || value === 'false') return value === 'true';
  if (value === 'null' || value === '~') return null;
  if (/^-?\d+(\.\d+)?$/.test(value)) return Number(value);
  return value;
}

/** Splits on commas outside parentheses and brackets. */
function splitTopLevel(text: string): string[] {
  const parts: string[] = [];
  let depth = 0;
  let current = '';
  for (const char of text) {
    if (char === '(' || char === '[') depth += 1;
    if (char === ')' || char === ']') depth -= 1;
    if (char === ',' && depth === 0) {
      parts.push(current);
      current = '';
    } else {
      current += char;
    }
  }
  if (current.trim() !== '') parts.push(current);
  return parts.map((part) => part.trim()).filter((part) => part !== '');
}

// Profiles.

function stringList(value: Yaml | undefined): string[] {
  if (Array.isArray(value)) return value.map(String);
  if (typeof value === 'string') return splitTopLevel(value);
  return [];
}

/** Splits a Markdown file into its YAML header and its body. */
function frontMatter(text: string): { header: string; body: string } | undefined {
  const match = /^---\r?\n([\s\S]*?)\r?\n---(?:\r?\n|$)([\s\S]*)$/.exec(text.replace(/^﻿/, ''));
  return match ? { header: match[1], body: match[2] } : undefined;
}

function isMap(value: Yaml | undefined): value is YamlMap {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function isList(value: Yaml | undefined): boolean {
  return Array.isArray(value) || typeof value === 'string';
}

/** The top-level keys of `agent.schema.json`. */
const KNOWN_KEYS = [
  'autoCompactWindow', 'background', 'color', 'description', 'disallowedTools', 'effort',
  'experimental', 'hooks', 'initialPrompt', 'isolation', 'maxTurns', 'mcpServers', 'memory',
  'model', 'name', 'omitClaudeMd', 'permissionMode', 'skills', 'tools', 'zed',
];

/** Why the keys this hook uses have the wrong type, if they do. */
function headerTypeError(header: YamlMap): string | undefined {
  const dotted = Object.keys(header).find((key) => key.startsWith('zed.'));
  if (dotted) return `\`${dotted}\` must be written under a \`zed:\` table`;
  if (header.zed !== undefined && !isMap(header.zed)) return '`zed` must be a table';
  const zed = (header.zed ?? {}) as YamlMap;
  for (const key of ['tools', 'disallowed_tools']) {
    if (zed[key] !== undefined && !isList(zed[key])) return `\`zed.${key}\` must be a list`;
  }
  if (zed.model !== undefined && !isMap(zed.model)) return '`zed.model` must be a table';
  if (header.skills !== undefined && !isList(header.skills)) return '`skills` must be a list';
  return undefined;
}

function readProfile(file: string, isProject: boolean): Profile | undefined {
  const text = fs.readFileSync(file, 'utf8').replace(/\r\n/g, '\n');
  const reject = (reason: string) => {
    warnOnce(`${file}: profile ignored: ${reason}`, text);
    rejectedFiles.push(file);
    return undefined;
  };
  const parts = frontMatter(text);
  if (!parts) return reject('it does not start with a YAML header between `---` lines');
  let header: YamlMap;
  try {
    header = parseYaml(parts.header);
  } catch (error) {
    return reject(`its header cannot be read: ${(error as Error).message}`);
  }
  const name = typeof header.name === 'string' ? header.name.trim() : undefined;
  if (!name || name.startsWith('-') || name.includes(':') || name.length > 256) {
    return reject('its `name` is missing or invalid');
  }
  if (typeof header.description !== 'string' || header.description.trim() === '') {
    return reject('its `description` is missing');
  }
  if (isProject && header.hooks !== undefined) {
    return reject('a project profile may not declare hooks');
  }
  const typeError = headerTypeError(header);
  if (typeError) return reject(typeError);
  // Claude Code ignores unknown keys, so they stay accepted, but named.
  const unknown = Object.keys(header).filter((key) => !KNOWN_KEYS.includes(key));
  if (unknown.length > 0) warnOnce(`${file}: unknown keys ignored: ${unknown.join(', ')}`, text);
  const zed = (header.zed ?? {}) as YamlMap;
  let tools: string[] | undefined;
  let subagents: Profile['subagents'] = 'none';
  if (zed.tools !== undefined) {
    tools = [];
    for (const entry of stringList(zed.tools)) {
      const spawn = /^spawn_agent\s*\((.*)\)$/.exec(entry);
      if (spawn) {
        tools.push('spawn_agent');
        subagents = splitTopLevel(spawn[1]);
      } else {
        tools.push(entry);
        if (entry === 'spawn_agent') subagents = 'all';
      }
    }
  } else {
    // Without `zed.tools`, Claude Code's tool names do not apply to Zed's
    // tools: every tool is allowed, spawning included.
    subagents = 'all';
  }
  const model = zed.model as YamlMap | undefined;
  return {
    name,
    description: header.description.trim(),
    body: parts.body.trim(),
    header,
    file,
    text,
    isProject,
    tools,
    disallowedTools: stringList(zed.disallowed_tools),
    subagents,
    model: model?.provider && model?.id ? `${model.provider}/${model.id}` : undefined,
    skills: header.skills === undefined ? undefined : stringList(header.skills),
  };
}

function markdownFiles(dir: string): string[] {
  if (!fs.existsSync(dir)) return [];
  return fs.readdirSync(dir, { recursive: true, encoding: 'utf8' })
    .filter((file) => file.endsWith('.md'))
    .sort()
    .map((file) => path.join(dir, file));
}

/** Personal profiles first; a project profile of the same name wins. */
function loadProfiles(personalDirs: string[], cwd: string): Map<string, Profile> {
  const profiles = new Map<string, Profile>();
  const sources: [string, boolean][] = [
    ...personalDirs.map((dir): [string, boolean] => [dir, false]),
    [path.join(cwd, '.zed', 'agents'), true],
  ];
  for (const [dir, isProject] of sources) {
    for (const file of markdownFiles(dir)) {
      const profile = readProfile(file, isProject);
      if (!profile) continue;
      const replaced = profiles.get(profile.name);
      if (replaced) {
        warnOnce(
          `profile ${profile.name}: ${profile.file} replaces ${replaced.file}`,
          replaced.text + profile.text,
        );
      }
      profiles.set(profile.name, profile);
    }
  }
  return profiles;
}

function warnUnapplied(profile: Profile): void {
  const notYet = NOT_APPLIED_YET.filter((field) => profile.header[field] !== undefined);
  if (notYet.length > 0) {
    warnOnce(`profile ${profile.name}: not applied yet by Zed: ${notYet.join(', ')}`, profile.text);
  }
  const zed = (profile.header.zed ?? {}) as YamlMap;
  const notApplied = [
    ...NOT_APPLIED.filter((field) => profile.header[field] !== undefined),
    ...(zed.max_spawn_depth !== undefined ? ['zed.max_spawn_depth'] : []),
  ];
  if (notApplied.length > 0) {
    warnOnce(`profile ${profile.name}: accepted but not applied: ${notApplied.join(', ')}`, profile.text);
  }
}

function isAllowed(profile: Profile, tool: string): boolean {
  if (profile.disallowedTools.includes(tool)) return false;
  return profile.tools === undefined || profile.tools.includes(tool);
}

function isSkillAllowed(profile: Profile, skill: string): boolean {
  if (!isAllowed(profile, 'skill')) return false;
  return profile.skills === undefined || profile.skills.includes(skill);
}

/**
 * The skill whose `SKILL.md` a read targets, when it sits in a skill root
 * (`.agents/skills/<folder>/SKILL.md`). Zed names a skill by its header, not
 * its folder (crates/agent_skills/agent_skills.rs:276).
 */
function skillOfRead(file: string, cwd: string): string | undefined {
  const absolute = path.resolve(cwd, file);
  const parts = absolute.split(/[\\/]/);
  const last = parts.length - 1;
  if (
    last < 3 ||
    parts[last].toLowerCase() !== 'skill.md' ||
    parts[last - 2].toLowerCase() !== 'skills' ||
    parts[last - 3].toLowerCase() !== '.agents'
  ) {
    return undefined;
  }
  try {
    const parts = frontMatter(fs.readFileSync(absolute, 'utf8'));
    const name = parts ? parseYaml(parts.header).name : undefined;
    if (typeof name === 'string' && name.trim() !== '') return name.trim();
  } catch {
    // Unreadable: the folder names it.
  }
  return parts[last - 1];
}

/** The profiles `profile` may launch, in the order of its list. */
function launchable(profile: Profile, profiles: Map<string, Profile>): Profile[] {
  if (profile.subagents === 'none') return [];
  const names = profile.subagents === 'all'
    ? [...profiles.keys()].filter((name) => name !== ROOT_PROFILE).sort()
    : profile.subagents;
  return names.flatMap((name) => profiles.get(name) ?? []);
}

// Which profile a thread runs.

type Agent =
  | { kind: 'profile'; profile: Profile; isSubagent: boolean }
  | { kind: 'no-pilote'; message: string }
  | { kind: 'unknown' };

function recordFile(threadId: string): string {
  return path.join(dataDir, `${threadId.replace(/[^A-Za-z0-9_-]/g, '_')}.json`);
}

/** The file of a token the hook issued for an allowed spawn_agent. */
function tokenFile(token: string): string {
  return path.join(dataDir, 'tokens', token);
}

function agentOf(input: any, profiles: Map<string, Profile>, agentDirs: string[]): Agent {
  if (!input.zed?.is_subagent) {
    const pilote = profiles.get(ROOT_PROFILE);
    if (!pilote) {
      const rejected = rejectedFiles.find((file) => path.basename(file) === `${ROOT_PROFILE}.md`);
      const fix = rejected ?? agentDirs.map((dir) => path.join(dir, `${ROOT_PROFILE}.md`)).join(' or ');
      const message = piloteMissing(fix || `${ROOT_PROFILE}.md`);
      warnOnce(message);
      return { kind: 'no-pilote', message };
    }
    return { kind: 'profile', profile: pilote, isSubagent: false };
  }
  try {
    const name = JSON.parse(fs.readFileSync(recordFile(input.zed.thread_id), 'utf8')).profile;
    const profile = profiles.get(name);
    if (profile) return { kind: 'profile', profile, isSubagent: true };
  } catch {
    // Not recorded.
  }
  return { kind: 'unknown' };
}

// Events.

function reminder(text: string): object {
  return { hookSpecificOutput: { hookEventName: 'BeforeModel', ephemeral_messages: [text] } };
}

/** Sorted, so that the same profile gives the same text and lists. */
function names(value: unknown, field: string): string[] {
  if (value === undefined) return [];
  if (!Array.isArray(value)) throw new Error(`\`${field}\` is not a list`);
  return value.map(String).sort();
}

/** Keeps the reading tools only, saying why. */
function readingOnly(toolNames: string[], message: string): object {
  const removed = toolNames.filter((tool) => !READ_TOOLS.includes(tool));
  return {
    hookSpecificOutput: {
      hookEventName: 'BeforeModel',
      ephemeral_messages: [message],
      ...(removed.length > 0 ? { remove_tools: removed } : {}),
    },
  };
}

function beforeModel(input: any, agent: Agent, profiles: Map<string, Profile>): object {
  if (input.is_compaction) return {};
  const toolNames = names(input.tool_names, 'tool_names');
  const skillNames = names(input.skill_names, 'skill_names');
  if (agent.kind === 'no-pilote') return readingOnly(toolNames, agent.message);
  if (agent.kind === 'unknown') return reminder(PROFILE_UNKNOWN);
  const { profile, isSubagent } = agent;
  const parts = profile.body ? [profile.body] : [];
  const subagents = launchable(profile, profiles);
  if (subagents.length > 0 && toolNames.includes('spawn_agent') && isAllowed(profile, 'spawn_agent')) {
    parts.push([
      'Subagent profiles you may launch (start the spawn_agent label with [name]):',
      ...subagents.map((subagent) => `- ${subagent.name}: ${subagent.description}`),
    ].join('\n'));
  }
  const removed = toolNames.filter((tool) =>
    !isAllowed(profile, tool) ||
    // A profile without any skill has no use for the skill tool either.
    (tool === 'skill' && profile.skills?.length === 0) ||
    // A sibling conversation would start over as pilote.
    (tool === 'create_thread' && isSubagent));
  // Skills outside the profile leave the catalogue; PreToolUse refuses them.
  const hidden = skillNames.filter((skill) => !isSkillAllowed(profile, skill));
  return {
    hookSpecificOutput: {
      hookEventName: 'BeforeModel',
      system_prompt_insert: ['## Agent Profile', parts.join('\n\n')].join('\n'),
      ...(removed.length > 0 ? { remove_tools: removed } : {}),
      ...(hidden.length > 0 ? { remove_skills: hidden } : {}),
    },
  };
}

function deny(reason: string): object {
  return {
    hookSpecificOutput: {
      hookEventName: 'PreToolUse',
      permissionDecision: 'deny',
      permissionDecisionReason: reason,
    },
  };
}

function preToolUse(input: any, agent: Agent, profiles: Map<string, Profile>, cwd: string): object {
  const tool = input.tool_name;
  if (agent.kind === 'no-pilote') return READ_TOOLS.includes(tool) ? {} : deny(agent.message);
  if (agent.kind === 'unknown') return deny(PROFILE_UNKNOWN);
  const { profile, isSubagent } = agent;
  if (!isAllowed(profile, tool)) {
    return deny(`The ${tool} tool is not allowed for the ${profile.name} profile.`);
  }
  if (tool === 'create_thread' && isSubagent) {
    return deny('A subagent may not open a sibling conversation.');
  }
  // Skills outside the profile: loaded by the skill tool, or read directly.
  const skill = tool === 'skill'
    ? String(input.tool_input?.name ?? '')
    : tool === 'read_file'
      ? skillOfRead(String(input.tool_input?.path ?? ''), cwd)
      : undefined;
  if (skill !== undefined && !isSkillAllowed(profile, skill)) {
    return deny(`The ${skill} skill is not allowed for the ${profile.name} profile.`);
  }
  if (tool !== 'spawn_agent') return {};
  // A follow-up to an existing subagent keeps its profile. Zed takes a blank
  // session id for none (spawn_agent_tool.rs:63-75): that is a new launch.
  const sessionId = input.tool_input?.session_id;
  if (typeof sessionId === 'string' && sessionId.trim() !== '') return {};
  const permitted = launchable(profile, profiles);
  const label = String(input.tool_input?.label ?? '');
  const match = /^\[([^\]#\s]+)\]\s*([\s\S]*)$/.exec(label);
  const target = match ? permitted.find((candidate) => candidate.name === match[1]) : undefined;
  if (!match || !target) {
    const names = permitted.map((candidate) => `[${candidate.name}]`).join(', ');
    return deny(`Start the label with an allowed profile: ${names}.`);
  }
  // A unique token tells two launches of the same profile apart. It is
  // recorded as issued: SubagentStart accepts no other.
  const token = `${target.name}#${randomBytes(2).toString('hex')}`;
  fs.mkdirSync(path.dirname(tokenFile(token)), { recursive: true });
  fs.writeFileSync(tokenFile(token), '');
  const updatedInput = { ...input.tool_input, label: `[${token}] ${match[2]}` };
  if (target.model) updatedInput.model = target.model;
  return {
    hookSpecificOutput: {
      hookEventName: 'PreToolUse',
      permissionDecision: 'allow',
      updatedInput,
    },
  };
}

function subagentStart(input: any, profiles: Map<string, Profile>): object {
  const match = /^\[(([^\]#\s]+)#[0-9a-f]{4})\]/.exec(String(input.agent_type ?? ''));
  const profile = match ? profiles.get(match[2]) : undefined;
  if (!match || !profile) return {};
  // Only a token the hook issued starts a profile, and only once: a label
  // made up by the model leaves the subagent with an unknown profile.
  try {
    fs.unlinkSync(tokenFile(match[1]));
  } catch {
    warnOnce(`subagent ${input.agent_id}: the token ${match[1]} was not issued, so its tools are refused`);
    return {};
  }
  fs.mkdirSync(dataDir, { recursive: true });
  fs.writeFileSync(recordFile(input.agent_id), JSON.stringify({ profile: profile.name }));
  warnUnapplied(profile);
  return {};
}

function optionValues(args: string[], option: string): string[] {
  return args.flatMap((arg, i) => (arg === option && i + 1 < args.length ? [args[i + 1]] : []));
}

let text = '';
for await (const chunk of process.stdin) text += chunk;
const input = JSON.parse(text);
const args = process.argv.slice(2);
const cwd = typeof input.cwd === 'string' ? input.cwd : process.cwd();
dataDir = process.env.ZED_HOOK_DATA_DIR ?? path.join(os.tmpdir(), 'zed-agent-profiles');

let output: object = {};
try {
  const agentDirs = optionValues(args, '--agents');
  const profiles = loadProfiles(agentDirs, cwd);
  if (input.hook_event_name === 'SubagentStart') {
    output = subagentStart(input, profiles);
  } else if (input.hook_event_name === 'BeforeModel' || input.hook_event_name === 'PreToolUse') {
    const agent = agentOf(input, profiles, agentDirs);
    if (agent.kind === 'profile') warnUnapplied(agent.profile);
    output = input.hook_event_name === 'BeforeModel'
      ? beforeModel(input, agent, profiles)
      : preToolUse(input, agent, profiles, cwd);
  }
} catch (error) {
  // A failure closes: refuse the call, or keep the reading tools only. A
  // subagent the hook could not start has an unknown profile.
  process.stderr.write(`agent-profiles: ${(error as Error)?.stack ?? error}\n`);
  if (input.hook_event_name === 'PreToolUse') {
    output = deny(`The agent profiles hook failed (${(error as Error)?.message ?? error}), so this call is refused.`);
  } else if (input.hook_event_name === 'BeforeModel' && !input.is_compaction) {
    const toolNames = Array.isArray(input.tool_names) ? input.tool_names.map(String).sort() : [];
    output = readingOnly(toolNames, hookFailed(error));
  }
}
process.stdout.write(JSON.stringify(output));
