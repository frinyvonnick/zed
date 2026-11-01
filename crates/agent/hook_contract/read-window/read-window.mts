// Read window hook for Zed. A whole read_file of a long file, sent again with
// every request, fills the context with lines the model does not need. On
// PreToolUse, this hook denies a read_file without a window on a file of more
// than MAX_LINES lines, and any window of more than MAX_LINES lines. The reason
// tells the model to search first, then to read a window around what it found.
//
// Declare it as a command hook on PreToolUse, matcher read_file:
//   node read-window.mts
//
// - Skills are read whole: a path under .agents/skills or .claude/skills passes.
// - Zed paths often start with the name of the project root folder (the last
//   segment of `cwd`), or are absolute; the hook resolves both against `cwd`.
// - A file it cannot find, an input it cannot read, or another tool gives no
//   output at all: Zed runs the call as the model wrote it.
import fs from 'node:fs';
import path from 'node:path';

/** Most lines one read_file may return. */
const MAX_LINES = 120;

/** Folders whose files are read whole. */
const SKILL_FOLDERS = ['.agents/skills', '.claude/skills'];

function isSkill(file: string): boolean {
  const normalized = `/${file.replace(/\\/g, '/')}`;
  return SKILL_FOLDERS.some((folder) => normalized.includes(`/${folder}/`));
}

/** The file on disk the model means, or undefined when none exists. */
function resolve(file: string, cwd: string): string | undefined {
  const root = path.basename(cwd.replace(/[\\/]+$/, ''));
  const segments = file.split(/[\\/]/);
  const candidates = path.isAbsolute(file)
    ? [file]
    : [...(segments[0] === root ? [path.join(cwd, ...segments.slice(1))] : []), path.join(cwd, file)];
  return candidates.find((candidate) => {
    try {
      return fs.statSync(candidate).isFile();
    } catch {
      return false;
    }
  });
}

/** Lines of a file, read by chunks: the file is never held whole in memory. */
function lineCount(file: string): number {
  const chunk = Buffer.alloc(1 << 16);
  const descriptor = fs.openSync(file, 'r');
  try {
    let newlines = 0;
    let last = -1;
    for (let read; (read = fs.readSync(descriptor, chunk, 0, chunk.length, null)) > 0;) {
      for (let at = chunk.indexOf(10); at !== -1 && at < read; at = chunk.indexOf(10, at + 1)) {
        newlines += 1;
      }
      last = chunk[read - 1];
    }
    // A last line without a newline is a line too.
    return last === -1 ? 0 : newlines + (last === 10 ? 0 : 1);
  } finally {
    fs.closeSync(descriptor);
  }
}

/** Lines read_file returns, with its rules: start 1 and end of file by default, end at least start. */
function windowLines(total: number, start: unknown, end: unknown): number {
  const first = Math.max(Number.isInteger(start) ? (start as number) : 1, 1);
  const last = Math.min(Math.max(Number.isInteger(end) ? (end as number) : total, first), total);
  return Math.max(last - first + 1, 0);
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

const WHAT_TO_DO = 'Search first with grep (or a symbol search tool, if you have one) for the lines you need, '
  + `then read_file a window of at most ${MAX_LINES} lines around them, with start_line and end_line.`;

function main(): void {
  let input: any;
  try {
    input = JSON.parse(fs.readFileSync(0, 'utf8'));
  } catch {
    return;
  }
  const file = input?.tool_input?.path;
  if (input?.hook_event_name !== 'PreToolUse' || input.tool_name !== 'read_file'
    || typeof file !== 'string' || typeof input.cwd !== 'string' || isSkill(file)) {
    return;
  }
  const found = resolve(file, input.cwd);
  if (found === undefined) {
    return;
  }
  const { start_line: start, end_line: end } = input.tool_input;
  let what: string;
  if (Number.isInteger(start) && Number.isInteger(end)) {
    // Both bounds: decided on what the model asked for, without reading the file.
    const asked = Math.max(end, start) - Math.max(start, 1) + 1;
    if (asked <= MAX_LINES) {
      return;
    }
    what = `This window asks for ${asked} lines, more than ${MAX_LINES}.`;
  } else {
    const lines = windowLines(lineCount(found), start, end);
    if (lines <= MAX_LINES) {
      return;
    }
    what = start == null && end == null
      ? `This file has ${lines} lines, more than ${MAX_LINES}, so it is not read whole.`
      : `This window has ${lines} lines, more than ${MAX_LINES}.`;
  }
  process.stdout.write(JSON.stringify(deny(`${what} ${WHAT_TO_DO}`)));
}

try {
  main();
} catch (error) {
  process.stderr.write(`read-window: ${error instanceof Error ? error.message : String(error)}; nothing decided\n`);
}
