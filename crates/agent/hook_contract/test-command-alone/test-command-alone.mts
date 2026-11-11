// Command-alone hook for Zed. In the eval bench (MXC), a few commands may leave the sandbox, by exact
// prefix (list in ZED_EVAL_MXC_UNSANDBOXED, `;`-separated). Zed lets one out only when it is just that
// prefix and its arguments. Composed (`git add … && <prefix>`, or `<prefix> > f.log`), it stays inside the
// sandbox, where it fails with error 0xC0000142 (other node commands do run there, 11/11 RDSO), and nothing tells the model why: on 11/11 (run
// N11111529-RDPD-a, requests 36 and 44) the agent hunted an antivirus for 18 requests.
// On PreToolUse, this hook refuses a terminal command that holds such a prefix without being reduced to
// it, and says what to do. Any other command runs as written. The sandbox list is never widened.
//
// Declare it as a command hook on PreToolUse:
//   node test-command-alone.mts
//
// - The list is read from the environment of the process, where Zed reads it: no copy to keep in step.
//   Without the list (no bench), the hook decides nothing.
// - « Alone » follows Zed's own test (`mxc_unsandboxed_prefix`, crates/acp_thread/src/terminal.rs): once
//   trimmed, the command starts with the prefix followed by the end or a space, and holds none of
//   `; & | \` $ ( ) < >` nor a line break. The working directory check of Zed is not repeated here.
// - An input it cannot read gives no output at all.
import fs from 'node:fs';

const TERMINAL_TOOLS = new Set(['terminal', 'sandboxed_terminal']);
const FORBIDDEN = new Set([';', '&', '|', '`', '$', '(', ')', '<', '>', '\n', '\r']);

function prefixesFrom(list: string | undefined): string[] {
  return (list ?? '').split(';').map((prefix) => prefix.trim()).filter((prefix) => prefix !== '');
}

/** True when `prefix` occurs in `command` as a whole command word run, not inside a longer word. */
function holds(command: string, prefix: string): boolean {
  for (let at = command.indexOf(prefix); at !== -1; at = command.indexOf(prefix, at + 1)) {
    const next = command[at + prefix.length];
    if (next === undefined || /\s/.test(next) || FORBIDDEN.has(next)) return true;
  }
  return false;
}

function isAlone(command: string, prefix: string): boolean {
  const trimmed = command.trim();
  if ([...trimmed].some((character) => FORBIDDEN.has(character))) return false;
  const rest = trimmed.startsWith(prefix) ? trimmed.slice(prefix.length) : null;
  return rest !== null && (rest === '' || rest.startsWith(' '));
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

function main(): void {
  let input: any;
  try {
    input = JSON.parse(fs.readFileSync(0, 'utf8'));
  } catch {
    return;
  }
  const command = input?.tool_input?.command;
  if (input?.hook_event_name !== 'PreToolUse' || !TERMINAL_TOOLS.has(input.tool_name) || typeof command !== 'string') {
    return;
  }
  const prefixes = prefixesFrom(process.env.ZED_EVAL_MXC_UNSANDBOXED);
  if (prefixes.some((prefix) => isAlone(command, prefix))) return;
  const held = prefixes.find((prefix) => holds(command, prefix));
  if (held === undefined) return;
  process.stdout.write(JSON.stringify(deny(
    `This command must run alone to leave the sandbox: run exactly \`${held} …\` with no \`&&\`, \`;\`, pipe or redirection. `
    + 'Run any other step (git add, reading the output) as a separate command. '
    + 'Inside the sandbox this test command fails with error 0xC0000142, even though other node commands run there.',
  )));
}

main();
