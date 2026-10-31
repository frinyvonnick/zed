// Short grep hook for Zed. A grep result longer than MAX_CHARS stays in the
// conversation and is sent again with every request. On BeforeModel, this hook
// replaces it with its first lines and the list of its matches by file, each
// with its lines, so that read_file can show any of them in full.
//
// Declare it as a command hook on BeforeModel:
//   node grep-court.mts
//
// - Zed applies a replacement to one request only, and sends the hook only the
//   messages it has not seen. The hook keeps, in ZED_HOOK_DATA_DIR, one file per
//   thread with the replacements it made, and repeats them with the same text
//   on every request, so the server can reuse its cached prompt prefix.
// - It repeats a replacement only while its message is among the known ones:
//   Zed rejects a replacement for a result that is not in the request.
// - An input it cannot read changes nothing.
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

/** Beyond this length, a grep result is shortened. */
const MAX_CHARS = 6000;

/** At most this many lines and characters of the original result are kept at its top. */
const HEAD_LINES = 12;
const HEAD_CHARS = 1500;

interface Replacement {
  /** Position of its message among the non-system messages. */
  index: number;
  tool_use_id: string;
  text: string;
}

/** The short form of a grep result, or undefined when it would not be shorter. */
function shorten(text: string): string | undefined {
  const lines = text.split('\n');
  const matches = new Map<string, string[]>();
  let file: string | undefined;
  let inSnippet = false;
  for (const line of lines) {
    if (line.startsWith('```')) {
      inSnippet = !inSnippet;
      continue;
    }
    const header = inSnippet ? null : /^## Matches in (.+)$/.exec(line);
    if (header) {
      file = header[1];
      continue;
    }
    const match = inSnippet ? null : /^### (?:.* › )?(L\d+(?:-\d+)?)$/.exec(line);
    if (file && match) {
      matches.set(file, [...(matches.get(file) ?? []), match[1]]);
    }
  }
  const head: string[] = [];
  let headChars = 0;
  let headInSnippet = false;
  for (const line of lines.slice(0, HEAD_LINES)) {
    const room = HEAD_CHARS - headChars;
    if (room <= 0) {
      break;
    }
    head.push(line.length > room ? `${line.slice(0, room)}…` : line);
    headChars += line.length;
    if (line.startsWith('```')) {
      headInSnippet = !headInSnippet;
    }
  }
  if (headInSnippet) {
    head.push('```');
  }
  const byFile = [...matches]
    .sort(([a, linesA], [b, linesB]) => linesB.length - linesA.length || a.localeCompare(b))
    .map(([file, labels]) => `${file} (${labels.length}): ${labels.join(', ')}`);
  const short = [
    ...head,
    '…',
    '',
    `This grep result had ${text.length} characters, so it shows its first lines, then every match by file with its lines. ` +
      'read_file shows any of them in full. For fewer and closer matches, narrow the regex or add an include_pattern, such as `**/src/**/*.ts`.',
    '',
    'Matches by file (most matches first):',
    ...byFile,
  ].join('\n');
  return short.length < text.length ? short : undefined;
}

/** The replacements kept for this thread, or none when the file is missing or unreadable. */
function readReplacements(file: string): Replacement[] {
  try {
    const stored = JSON.parse(fs.readFileSync(file, 'utf8'));
    return Array.isArray(stored)
      ? stored.filter((r) => Number.isInteger(r?.index) && typeof r.tool_use_id === 'string' && typeof r.text === 'string')
      : [];
  } catch {
    return [];
  }
}

function main(): void {
  const input = JSON.parse(fs.readFileSync(0, 'utf8'));
  const known: number = input.known_message_count;
  const thread: string = input.zed?.thread_id ?? input.session_id;
  if (input.hook_event_name !== 'BeforeModel' || !Number.isInteger(known) || !thread) {
    return;
  }
  const dir = process.env.ZED_HOOK_DATA_DIR ?? path.join(os.tmpdir(), 'zed-grep-court');
  const file = path.join(dir, `${createHash('sha256').update(thread).digest('hex').slice(0, 32)}.json`);
  const replacements = readReplacements(file).filter((replacement) => replacement.index < known);
  (input.new_messages ?? []).forEach((message: any, offset: number) => {
    for (const part of message.content ?? []) {
      if (part.type === 'tool_result' && part.tool_name === 'grep' && !part.is_error
        && typeof part.text === 'string' && part.text.length > MAX_CHARS) {
        const text = shorten(part.text);
        if (text !== undefined) {
          replacements.push({ index: known + offset, tool_use_id: part.tool_use_id, text });
        }
      }
    }
  });
  if (replacements.length > 0) {
    process.stdout.write(JSON.stringify({
      hookSpecificOutput: {
        hookEventName: 'BeforeModel',
        tool_result_replacements: replacements.map(({ tool_use_id, text }) => ({ tool_use_id, text })),
      },
    }));
  }
  // Saved after the output, so that a failed save loses only the next requests' replacements.
  // Written then renamed, so that a hook run of the same thread never reads half a file.
  try {
    fs.mkdirSync(dir, { recursive: true });
    const temporary = `${file}.${process.pid}.tmp`;
    fs.writeFileSync(temporary, JSON.stringify(replacements));
    fs.renameSync(temporary, file);
  } catch (error) {
    process.stderr.write(`grep-court: could not save its replacements: ${error instanceof Error ? error.message : String(error)}\n`);
  }
}

try {
  main();
} catch (error) {
  process.stderr.write(`grep-court: ${error instanceof Error ? error.message : String(error)}; nothing replaced\n`);
}
