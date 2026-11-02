// Terminal working directory hook for Zed. The `terminal` tool requires a `cd` field, and Qwen writes
// it under another name: `cwd` seven times in a row on 02/11 (tour 26b), `c` on 01/11 (tour 19). Zed
// answers « missing field `cd` », which names the field but not the fix, and the agent concludes its
// terminal is broken. On PreToolUse, this hook renames `cwd` or `c` to `cd`, or sets `cd: "."` (the
// project root) when no working directory is given. It decides nothing: no allow, no deny, no stop.
//
// Declare it as a command hook on PreToolUse:
//   node terminal-cd.mts
//
// - The tool's schema allows no extra field, so the misnamed one is removed.
// - An input it cannot read, or a call that already has `cd`, gives no output at all: Zed runs the
//   call as the model wrote it.
import fs from 'node:fs';

const TERMINAL_TOOLS = new Set(['terminal', 'sandboxed_terminal']);
const MISNAMED_FIELDS = ['cwd', 'c'];

function main(): void {
  let input: any;
  try {
    input = JSON.parse(fs.readFileSync(0, 'utf8'));
  } catch {
    return;
  }
  const toolInput = input?.tool_input;
  if (input?.hook_event_name !== 'PreToolUse' || !TERMINAL_TOOLS.has(input.tool_name)
    || typeof toolInput !== 'object' || toolInput === null || Array.isArray(toolInput)
    || Object.hasOwn(toolInput, 'cd')) {
    return;
  }
  const corrected: Record<string, unknown> = {};
  let directory: unknown = '.';
  for (const [field, value] of Object.entries(toolInput)) {
    if (MISNAMED_FIELDS.includes(field)) {
      if (typeof value === 'string' && value !== '') directory = value;
    } else {
      corrected[field] = value;
    }
  }
  corrected.cd = directory;
  process.stdout.write(JSON.stringify({
    hookSpecificOutput: {
      hookEventName: 'PreToolUse',
      updatedInput: corrected,
    },
  }));
}

main();
