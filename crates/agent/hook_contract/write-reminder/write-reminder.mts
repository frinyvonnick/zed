// Write reminder hook for Zed. An agent can keep reading for a long time
// without writing anything, and read the same files again after each
// compaction. This hook nudges it to write; it never stops it.
//
// Declare it as a command hook on BeforeModel, AfterModel and PostCompact:
//   node write-reminder.mts
//
// - BeforeModel: it counts the model requests since the last successful write
//   (a result of one of WRITE_TOOLS in new_messages). At every REMINDER_EVERY
//   requests, it adds an ephemeral reminder to write the next file, with the
//   context size once one is measured. Beyond LEVEL_EVERY_REQUEST of the
//   window, every request without a write gets the context size.
// - AfterModel: it keeps the context size of the response, from the usage the
//   provider reported. No count of its own, no call to the model server.
// - PostCompact: at every compaction, it appends to the summary the first user
//   message of the thread, word for word (its initial goal, kept from the first
//   BeforeModel), then an instruction not to read the same files again. The
//   context size is forgotten until the next measured response.
// - It keeps one small file per thread (`zed.thread_id`) in ZED_HOOK_DATA_DIR:
//   the goal and a few numbers, never the history. It reads only new_messages.
// - Zed calls BeforeModel again on a retry, with no new message: that request
//   is not counted again, and gets the same reminder as before.
// - An input it cannot read changes nothing.
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

/** The reminder comes every this many model requests without a write. */
const REMINDER_EVERY = 8;

/** Context window of the model, in tokens. */
const CONTEXT_WINDOW = 122880;

/** Zed compacts the conversation at this size (90 %), measured in its logs: "threshold is 110592 of 122880". */
const COMPACTION_THRESHOLD = 110592;

/** Beyond this share of the window, every request without a write tells the context size. */
const LEVEL_EVERY_REQUEST = 0.7;

/** Tools that change files, from crates/agent/src/tools/*.rs. */
const WRITE_TOOLS = ['edit_file', 'write_file', 'create_directory', 'copy_path', 'move_path', 'delete_path', 'rename_symbol', 'apply_code_action'];

/** Requests that are not a step of the agent: they are not counted. */
const SUMMARY_INTENTS = ['thread_summarization', 'thread_context_summarization'];

const AFTER_COMPACTION = 'The facts you read are in your summary above: do not read those files again; write the next file now.';

function reminder(count: number): string {
  return `You have made ${count} model requests since your last write. `
    + 'Do you already have what you need to write the next file? If yes, write it now; '
    + 'read again only the few lines the next edit needs.';
}

function level(tokens: number): string {
  return `Context window: ${Math.round(tokens / 1000)}k of ${Math.round(CONTEXT_WINDOW / 1000)}k tokens `
    + `(${Math.round((tokens / CONTEXT_WINDOW) * 100)} %). `
    + `Around ${Math.round(COMPACTION_THRESHOLD / 1000)}k, Zed compacts the conversation and the files you read are dropped from it. `
    + 'Write your next file before that; read again only the lines it needs.';
}

/** Tokens of the context, from the usage of one response. Zed reports cached prompt tokens apart from input_tokens. */
function contextTokens(usage: any): number | undefined {
  const fields = ['input_tokens', 'output_tokens', 'cache_creation_input_tokens', 'cache_read_input_tokens'];
  const values = fields.map((field) => usage?.[field]);
  return values.every((value) => Number.isInteger(value) && value >= 0) ? values.reduce((sum, value) => sum + value, 0) : undefined;
}

interface State {
  /** The first user message of the thread, word for word. */
  goal?: string;
  /** Model requests since the last write. */
  requests: number;
  /** Context size of the last measured response, in tokens. */
  contextTokens?: number;
  /** The reminders of the last counted request, sent again on its retry. */
  lastReminders: string[];
}

function readState(file: string): State {
  try {
    const stored = JSON.parse(fs.readFileSync(file, 'utf8'));
    return {
      goal: typeof stored?.goal === 'string' ? stored.goal : undefined,
      requests: Number.isInteger(stored?.requests) ? stored.requests : 0,
      contextTokens: Number.isInteger(stored?.contextTokens) ? stored.contextTokens : undefined,
      lastReminders: Array.isArray(stored?.lastReminders) ? stored.lastReminders.filter((text: unknown) => typeof text === 'string') : [],
    };
  } catch {
    return { requests: 0, lastReminders: [] };
  }
}

/** Written then renamed, so that a hook run of the same thread never reads half a file. */
function saveState(dir: string, file: string, state: State): void {
  try {
    fs.mkdirSync(dir, { recursive: true });
    const temporary = `${file}.${process.pid}.tmp`;
    fs.writeFileSync(temporary, JSON.stringify(state));
    fs.renameSync(temporary, file);
  } catch (error) {
    process.stderr.write(`write-reminder: could not save its state: ${error instanceof Error ? error.message : String(error)}\n`);
  }
}

/**
 * Blocks Zed adds to a user message: a slash command puts the skill it loads before what the user
 * typed (crates/agent/src/agent.rs:2318-2324), in the envelope of skill_tool.rs:63-76, whose body
 * cannot hold the closing tag (skill_tool.rs:30-31). System reminders are dropped too.
 */
const ENVELOPES = /<(skill_content|system-reminder)\b[^>]*>[\s\S]*?<\/\1\s*>/g;

/** The first user message that has some text, without the blocks Zed adds; the whole text if nothing else is left. */
function firstUserText(messages: any[]): string | undefined {
  for (const message of messages) {
    if (message?.role !== 'user') {
      continue;
    }
    const texts = (message.content ?? []).filter((part: any) => part?.type === 'text' && typeof part.text === 'string');
    if (texts.length > 0) {
      const whole = texts.map((part: any) => part.text).join('\n\n');
      const typed = whole.replace(ENVELOPES, '').trim();
      return typed === '' ? whole : typed;
    }
  }
  return undefined;
}

function hasWrite(messages: any[]): boolean {
  return messages.some((message) => (message?.content ?? []).some((part: any) =>
    part?.type === 'tool_result' && WRITE_TOOLS.includes(part.tool_name) && !part.is_error));
}

function beforeModel(input: any, state: State): string[] {
  const messages: any[] = Array.isArray(input.new_messages) ? input.new_messages : [];
  state.goal ??= firstUserText(messages);
  if (input.is_compaction === true || SUMMARY_INTENTS.includes(input.intent)) {
    return [];
  }
  if (messages.length === 0) {
    return state.lastReminders;
  }
  const wrote = hasWrite(messages);
  state.requests = wrote ? 1 : state.requests + 1;
  const measured = state.contextTokens === undefined ? undefined : level(state.contextTokens);
  if (state.requests % REMINDER_EVERY === 0) {
    state.lastReminders = [measured === undefined ? reminder(state.requests) : `${reminder(state.requests)} ${measured}`];
  } else if (!wrote && measured !== undefined && state.contextTokens! > LEVEL_EVERY_REQUEST * CONTEXT_WINDOW) {
    state.lastReminders = [measured];
  } else {
    state.lastReminders = [];
  }
  return state.lastReminders;
}

function postCompact(state: State): string {
  if (state.goal === undefined) {
    return AFTER_COMPACTION;
  }
  return `The initial goal of this conversation, its first user message word for word:\n\n${state.goal}\n\n${AFTER_COMPACTION}`;
}

function main(): void {
  const input = JSON.parse(fs.readFileSync(0, 'utf8'));
  const thread: string = input.zed?.thread_id ?? input.session_id;
  const event = input.hook_event_name;
  if (!['BeforeModel', 'AfterModel', 'PostCompact'].includes(event) || typeof thread !== 'string' || !thread) {
    return;
  }
  const dir = process.env.ZED_HOOK_DATA_DIR ?? path.join(os.tmpdir(), 'zed-write-reminder');
  const file = path.join(dir, `${createHash('sha256').update(thread).digest('hex').slice(0, 32)}.json`);
  const state = readState(file);
  if (event === 'AfterModel') {
    // The compaction request reports the old context, and a summary request is not the agent's.
    const tokens = contextTokens(input.usage);
    if (tokens !== undefined && input.is_compaction !== true && !SUMMARY_INTENTS.includes(input.intent)) {
      state.contextTokens = tokens;
      saveState(dir, file, state);
    }
    return;
  }
  if (event === 'PostCompact') {
    state.contextTokens = undefined;
    saveState(dir, file, state);
    process.stdout.write(JSON.stringify({
      hookSpecificOutput: { hookEventName: 'PostCompact', zed: { append_to_summary: postCompact(state) } },
    }));
    return;
  }
  const reminders = beforeModel(input, state);
  if (reminders.length > 0) {
    process.stdout.write(JSON.stringify({
      hookSpecificOutput: { hookEventName: 'BeforeModel', ephemeral_messages: reminders },
    }));
  }
  // Saved after the output, so that a failed save loses only the count, never this reminder.
  saveState(dir, file, state);
}

try {
  main();
} catch (error) {
  process.stderr.write(`write-reminder: ${error instanceof Error ? error.message : String(error)}; nothing added\n`);
}
