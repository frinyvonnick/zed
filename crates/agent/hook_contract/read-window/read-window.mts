// Read window hook for Zed. A whole read_file of a long file, sent again with
// every request, fills the context with lines the model does not need. On
// PreToolUse, this hook denies a read_file without a window on a file of more
// than MAX_LINES lines, and any window of more than MAX_LINES lines. The reason
// tells the model to search first, then to read a window around what it found.
//
// Declare it as a command hook on PreToolUse, matcher read_file:
//   node read-window.mts
//
// - A file over AUTO_OUTLINE_SIZE read without a window passes: Zed answers
//   with its outline (symbols and their lines, no content), a cheap way to aim.
// - Skills are read whole: a path under .agents/skills or .claude/skills passes.
// - Zed paths often start with the name of the project root folder (the last
//   segment of `cwd`), or are absolute; the hook resolves both against `cwd`.
// - A file it cannot find, an input it cannot read, or another tool gives no
//   output at all: Zed runs the call as the model wrote it.
import fs from 'node:fs';
import path from 'node:path';

/** Most lines one read_file may return. */
const MAX_LINES = 120;

/**
 * Beyond this size, a read_file without a window returns the file's outline instead of its
 * content: AUTO_OUTLINE_SIZE in crates/agent/src/outline.rs:10, compared with `>` to the
 * length of the buffer text (outline.rs:33-35), where each \r\n is a single \n.
 */
const AUTO_OUTLINE_SIZE = 16384;

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

/**
 * Lines of a file and its length as Zed holds it (each \r\n a single \n), read by chunks:
 * the file is never held whole in memory.
 */
function measure(file: string): { lines: number; length: number } {
  const chunk = Buffer.alloc(1 << 16);
  const descriptor = fs.openSync(file, 'r');
  try {
    let bytes = 0;
    let newlines = 0;
    let crlf = 0;
    let last = -1;
    for (let read; (read = fs.readSync(descriptor, chunk, 0, chunk.length, null)) > 0;) {
      for (let at = chunk.indexOf(10); at !== -1 && at < read; at = chunk.indexOf(10, at + 1)) {
        newlines += 1;
        if ((at > 0 ? chunk[at - 1] : last) === 13) {
          crlf += 1;
        }
      }
      bytes += read;
      last = chunk[read - 1];
    }
    // A last line without a newline is a line too.
    return { lines: last === -1 ? 0 : newlines + (last === 10 ? 0 : 1), length: bytes - crlf };
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

// The reasons follow the words of Zed's own answer for a large file
// (crates/agent/src/tools/read_file_tool.rs:522-530), so that the model meets one way of saying it.
const NEXT_STEPS = `NEXT STEPS: To read a specific section, call read_file with the same path plus start_line and end_line around the lines grep found, at most ${MAX_LINES} lines apart.\n`
  + 'For example, to read a function grep found at L100, use start_line: 100 and end_line: 150.';

function fileTooLarge(lines: number): string {
  return `This file is too large to read all at once: it has ${lines} lines, and read_file returns at most ${MAX_LINES} lines at a time.\n\n`
    + 'IMPORTANT: Do NOT retry this call without line numbers - you will get the same answer.\n'
    + 'Instead, find the lines you need with grep first, then read them by calling this tool again with start_line and end_line parameters.\n\n'
    + NEXT_STEPS;
}

function windowTooLarge(lines: number): string {
  return `This window is too large to read at once: it covers ${lines} lines, and read_file returns at most ${MAX_LINES} lines at a time.\n\n`
    + 'IMPORTANT: Do NOT retry this call with the same window - you will get the same answer.\n'
    + `Instead, find the lines you need with grep first, then read them by calling this tool again with start_line and end_line at most ${MAX_LINES} lines apart.\n\n`
    + NEXT_STEPS;
}

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
  let reason: string;
  if (Number.isInteger(start) && Number.isInteger(end)) {
    // Both bounds: decided on what the model asked for, without reading the file.
    const asked = Math.max(end, start) - Math.max(start, 1) + 1;
    if (asked <= MAX_LINES) {
      return;
    }
    reason = windowTooLarge(asked);
  } else {
    const whole = start == null && end == null;
    // At most one byte in two is the \r of a \r\n: beyond twice the limit, Zed answers with the outline.
    if (whole && fs.statSync(found).size > 2 * AUTO_OUTLINE_SIZE) {
      return;
    }
    const measured = measure(found);
    if (whole && measured.length > AUTO_OUTLINE_SIZE) {
      return;
    }
    const lines = windowLines(measured.lines, start, end);
    if (lines <= MAX_LINES) {
      return;
    }
    reason = whole ? fileTooLarge(lines) : windowTooLarge(lines);
  }
  process.stdout.write(JSON.stringify(deny(reason)));
}

try {
  main();
} catch (error) {
  process.stderr.write(`read-window: ${error instanceof Error ? error.message : String(error)}; nothing decided\n`);
}
