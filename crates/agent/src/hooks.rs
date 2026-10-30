use crate::{
    AgentToolOutput, AnyAgentTool, Thread, ToolCallEventStream, ToolInput, ToolInputPayload,
    ToolInputSender,
};
use agent_client_protocol::schema::v2 as acp_v2;
use agent_settings::{AgentSettings, COMPACTION_PROMPT};
use anyhow::{Context as _, Result, anyhow};
use collections::{HashMap, HashSet};
use futures::{AsyncWriteExt as _, FutureExt as _, StreamExt as _, channel::oneshot};
use gpui::{App, AsyncApp, Entity, Global, SharedString, Task, WeakEntity};
use language_model::{
    CompactedContext, CompletionIntent, LanguageModel, LanguageModelCompletionError,
    LanguageModelCompletionEvent, LanguageModelCompletionStream, LanguageModelProvider,
    LanguageModelProviderId, LanguageModelRequest, LanguageModelRequestMessage,
    LanguageModelToolResultContent, LanguageModelToolUseInput, MessageContent, Role, StopReason,
    TokenUsage,
};
use regex::Regex;
use serde_json::json;
use settings::Settings as _;
use sha2::{Digest as _, Sha256};
use std::{
    collections::BTreeMap,
    hash::{DefaultHasher, Hash as _, Hasher as _},
    path::{Path, PathBuf},
    process::Stdio,
    sync::Arc,
    time::{Duration, Instant},
};
use uuid::Uuid;

/// Version of the hook contract, sent to every hook as `zed.contract_version`.
pub const HOOK_CONTRACT_VERSION: &str = "0.1.0";

/// Runs one hook handler with the JSON input of an event and returns the
/// handler's JSON output.
pub trait HookHandler: 'static {
    fn call(&self, input: serde_json::Value, cx: &mut App) -> Task<Result<serde_json::Value>>;

    /// Identifies handlers that do the same thing, so that one declared
    /// twice runs once.
    fn dedup_key(&self) -> Option<String> {
        None
    }
}

/// Where an event fired; serialized into the fields shared by every hook input.
#[derive(Clone, Debug)]
pub struct HookContext {
    pub session_id: String,
    pub cwd: String,
    pub thread_id: String,
    pub parent_thread_id: Option<String>,
    pub depth: u32,
}

/// The hooks the user configured, by event.
#[derive(Default)]
pub struct AgentHooks {
    pub pre_tool_use: Vec<HookMatcherGroup>,
    pub post_tool_use: Vec<HookMatcherGroup>,
    pub before_model: Vec<HookMatcherGroup>,
    pub after_model: Vec<HookMatcherGroup>,
}

impl Global for AgentHooks {}

/// Replaces the directory command hooks keep their data in, by default
/// `hooks` in Zed's data directory; tests point it at a temporary directory.
pub struct HookDataRoot(pub PathBuf);

impl Global for HookDataRoot {}

impl AgentHooks {
    /// Reads the `agent.hooks` setting, and refuses it, naming where, when it
    /// asks for something Zed does not do yet.
    ///
    /// Each command hook gets its own data directory under `data_root`, keyed
    /// by its `id`, or by a hash of its command and arguments without one.
    pub fn from_settings(hooks: &serde_json::Value, data_root: &Path) -> Result<Self> {
        check_hook_settings(hooks)?;
        Ok(Self {
            pre_tool_use: matcher_groups_from_settings(&hooks["PreToolUse"], data_root),
            post_tool_use: matcher_groups_from_settings(&hooks["PostToolUse"], data_root),
            before_model: matcher_groups_from_settings(&hooks["BeforeModel"], data_root),
            after_model: matcher_groups_from_settings(&hooks["AfterModel"], data_root),
        })
    }
}

/// Names the data directory of a command hook: its `id`, or a hash of its
/// command and arguments when it declares none.
fn hook_data_key(handler: &serde_json::Value) -> Result<String> {
    if let Some(id) = handler.get("id") {
        let id = id.as_str().context("hook id must be a string")?;
        anyhow::ensure!(
            is_valid_hook_id(id),
            "invalid hook id {id:?}: use 1 to 64 lowercase letters, digits, `-` or `_`, \
             starting with a letter or a digit"
        );
        return Ok(id.to_string());
    }
    let mut hasher = Sha256::new();
    hasher.update(handler["command"].as_str().unwrap_or_default());
    for arg in handler["args"].as_array().into_iter().flatten() {
        hasher.update([0]);
        hasher.update(arg.as_str().unwrap_or_default());
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Matches `^[a-z0-9][a-z0-9_-]{0,63}$`: a directory name safe everywhere.
fn is_valid_hook_id(id: &str) -> bool {
    let mut chars = id.chars();
    chars
        .next()
        .is_some_and(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
        && id.len() <= 64
        && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, '-' | '_'))
}

/// Refuses settings Zed would not honour, naming where they are: an invalid
/// matcher, a command hook without a command, non-text arguments, an
/// unsupported `onFailure`, an invalid id, and one id declared by two
/// different hooks, which would silently share one data directory. The same
/// hook declared on several events keeps its id.
fn check_hook_settings(hooks: &serde_json::Value) -> Result<()> {
    let mut seen = HashMap::<String, (String, &serde_json::Value)>::default();
    for (event, groups) in hooks.as_object().into_iter().flatten() {
        anyhow::ensure!(
            SUPPORTED_EVENTS.contains(&event.as_str()),
            "hooks for `{event}` are not supported yet; Zed runs {}",
            SUPPORTED_EVENTS.join(", ")
        );
        for (group_ix, group) in groups.as_array().into_iter().flatten().enumerate() {
            if let Some(matcher) = group["matcher"].as_str()
                && !is_name_matcher(matcher)
            {
                Regex::new(matcher)
                    .with_context(|| format!("hook matcher {event}[{group_ix}].matcher"))?;
            }
            if let Some(matcher) = group["matcher"].as_str()
                && matches!(event.as_str(), "BeforeModel" | "AfterModel")
                && matcher_matches(Some(matcher), "thread_summarization")
                && !MODEL_INTENTS
                    .iter()
                    .all(|intent| matcher_matches(Some(matcher), intent))
            {
                anyhow::bail!(
                    "hook matcher {event}[{group_ix}].matcher: title requests \
                     (`thread_summarization`) do not run hooks"
                );
            }
            let handlers = group["hooks"].as_array().into_iter().flatten();
            for (handler_ix, handler) in handlers.enumerate() {
                let location = format!("{event}[{group_ix}].hooks[{handler_ix}]");
                check_command_handler(handler).with_context(|| format!("hook {location}"))?;
                if handler.get("id").is_none() {
                    continue;
                }
                let id = hook_data_key(handler).with_context(|| format!("hook {location}"))?;
                match seen.get(&id) {
                    Some((first, first_handler))
                        if first_handler["command"] != handler["command"]
                            || first_handler["args"] != handler["args"] =>
                    {
                        anyhow::bail!(
                            "hooks {first} and {location} both declare the id {id:?}; \
                             give each hook its own id"
                        );
                    }
                    Some(_) => {}
                    None => {
                        seen.insert(id, (location, handler));
                    }
                }
            }
        }
    }
    Ok(())
}

/// The events this version of Zed runs hooks for.
const SUPPORTED_EVENTS: [&str; 4] = ["PreToolUse", "PostToolUse", "BeforeModel", "AfterModel"];

fn check_command_handler(handler: &serde_json::Value) -> Result<()> {
    anyhow::ensure!(
        handler["type"] == "command",
        "hooks of type {} are not supported yet; only `command` hooks run",
        handler["type"]
    );
    for field in ["if", "async", "once"] {
        anyhow::ensure!(
            handler.get(field).is_none(),
            "`{field}` is not supported yet"
        );
    }
    if let Some(shell) = handler.get("shell") {
        anyhow::ensure!(
            matches!(shell.as_str(), Some("bash" | "powershell")),
            "`shell` must be \"bash\" or \"powershell\""
        );
    }
    if let Some(timeout) = handler.get("timeout") {
        anyhow::ensure!(
            timeout.as_u64().is_some_and(|timeout| timeout >= 1),
            "`timeout` must be a whole number of seconds, at least 1"
        );
    }
    anyhow::ensure!(
        handler["command"]
            .as_str()
            .is_some_and(|command| !command.is_empty()),
        "a command hook needs a `command`"
    );
    if let Some(args) = handler.get("args") {
        anyhow::ensure!(
            args.as_array()
                .is_some_and(|args| args.iter().all(serde_json::Value::is_string)),
            "`args` must be a list of strings"
        );
    }
    anyhow::ensure!(
        handler["onFailure"] != "block",
        "`onFailure: \"block\"` is not supported yet"
    );
    Ok(())
}

fn matcher_groups_from_settings(
    groups: &serde_json::Value,
    data_root: &Path,
) -> Vec<HookMatcherGroup> {
    let Some(groups) = groups.as_array() else {
        return Vec::new();
    };
    groups
        .iter()
        .map(|group| HookMatcherGroup {
            matcher: group["matcher"].as_str().map(str::to_string),
            hooks: group["hooks"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|handler| {
                    let command = handler["command"].as_str()?.to_string();
                    let args = handler["args"].as_array().map(|args| {
                        args.iter()
                            .filter_map(|arg| arg.as_str().map(str::to_string))
                            .collect()
                    });
                    Some(Arc::new(CommandHook {
                        command,
                        args,
                        shell: handler["shell"].as_str().map(str::to_string),
                        timeout: handler["timeout"].as_u64(),
                        data_dir: data_root.join(hook_data_key(handler).ok()?),
                    }) as Arc<dyn HookHandler>)
                })
                .collect(),
        })
        .collect()
}

/// Runs a command with the hook input as JSON on stdin and reads its JSON
/// output on stdout. Exit code 2 denies or blocks with stderr as the reason.
pub struct CommandHook {
    pub command: String,
    pub args: Option<Vec<String>>,
    pub shell: Option<String>,
    pub timeout: Option<u64>,
    /// Persistent directory the hook keeps its own state in, passed as
    /// `ZED_HOOK_DATA_DIR`. Zed creates it and never reads or deletes it.
    pub data_dir: PathBuf,
}

impl CommandHook {
    /// The program and arguments to run: the executable itself when `args`
    /// is set, otherwise `command` through a shell. Without an explicit
    /// `shell`, Windows uses PowerShell, so that paths in the hook input stay
    /// Windows paths; other platforms use bash.
    fn command_line(&self) -> (String, Vec<String>) {
        let powershell = || {
            let flags = [
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
            ];
            let mut args = flags.map(String::from).to_vec();
            args.extend(["-Command".into(), self.command.clone()]);
            ("powershell".into(), args)
        };
        let bash = || ("bash".into(), vec!["-c".into(), self.command.clone()]);
        match (&self.args, self.shell.as_deref()) {
            (Some(args), _) => (self.command.clone(), args.clone()),
            (None, Some("powershell")) => powershell(),
            (None, Some("bash")) => bash(),
            (None, _) if cfg!(windows) => powershell(),
            (None, _) => bash(),
        }
    }
}

impl HookHandler for CommandHook {
    fn dedup_key(&self) -> Option<String> {
        // The data directory stands for the hook's `id`.
        Some(format!(
            "{:?}",
            (
                &self.command,
                &self.args,
                &self.shell,
                self.timeout,
                &self.data_dir
            )
        ))
    }

    fn call(&self, input: serde_json::Value, cx: &mut App) -> Task<Result<serde_json::Value>> {
        let (program, args) = self.command_line();
        let mut command = util::command::new_command(program);
        command.args(args);
        command.env(HOOK_DATA_DIR_VAR, &self.data_dir);
        // On Windows, `bash` is often WSL's, which only receives the variables
        // listed in `WSLENV`; `/p` also translates the path for Linux.
        #[cfg(windows)]
        {
            let wslenv = std::env::var("WSLENV").unwrap_or_default();
            let entry = format!("{HOOK_DATA_DIR_VAR}/p");
            command.env(
                "WSLENV",
                if wslenv.is_empty() {
                    entry
                } else {
                    format!("{wslenv}:{entry}")
                },
            );
        }
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        let hook_event_name = input["hook_event_name"]
            .as_str()
            .unwrap_or_default()
            .to_string();
        let timeout = Duration::from_secs(self.timeout.unwrap_or(DEFAULT_HOOK_TIMEOUT_SECS));
        let timer = cx.background_executor().timer(timeout);
        let data_dir = self.data_dir.clone();
        cx.background_executor().spawn(async move {
            std::fs::create_dir_all(&data_dir)
                .with_context(|| format!("creating hook data directory {data_dir:?}"))?;
            // Like Claude Code, hooks run in the project directory, when it is
            // on this machine (a remote project's is not).
            if let Some(cwd) = input["cwd"].as_str().map(Path::new)
                && cwd.is_dir()
            {
                command.current_dir(cwd);
            }
            let input = serde_json::to_vec(&input)?;
            let mut child = command.spawn()?;
            let mut stdin = child.stdin.take().context("hook stdin is not piped")?;
            let write_input = async move {
                let written = async {
                    stdin.write_all(&input).await?;
                    // Closing flushes: on Windows the pipe is buffered and
                    // dropping it would lose the input.
                    stdin.close().await
                };
                // A hook may exit without reading its input; its exit code
                // and outputs decide then, not the closed pipe.
                if let Err(error) = written.await {
                    log::debug!("agent hook did not read all its input: {error}");
                }
            };
            // Writing the input and reading the outputs run together under
            // the timeout, so a hook that reads nothing cannot block the agent.
            // On timeout, dropping the child kills it.
            let run = futures::future::join(write_input, child.output());
            let output = futures::select_biased! {
                (_, output) = run.fuse() => output?,
                _ = timer.fuse() => anyhow::bail!("hook timed out after {timeout:?}"),
            };
            parse_command_output(
                &hook_event_name,
                output.status.code(),
                &String::from_utf8_lossy(&output.stdout),
                &String::from_utf8_lossy(&output.stderr),
            )
        })
    }
}

/// Environment variable holding the hook's own data directory.
const HOOK_DATA_DIR_VAR: &str = "ZED_HOOK_DATA_DIR";

/// Seconds before a hook is cancelled, as in Claude Code.
const DEFAULT_HOOK_TIMEOUT_SECS: u64 = 600;

/// Turns what a command hook printed into the hook output. Exit code 0 reads
/// stdout as JSON, exit code 2 denies (or blocks) with stderr as the reason,
/// and any other exit code is a hook failure.
fn parse_command_output(
    hook_event_name: &str,
    exit_code: Option<i32>,
    stdout: &str,
    stderr: &str,
) -> Result<serde_json::Value> {
    // Stdout counts as JSON only when it is one object with nothing but
    // whitespace around it; anything else is plain text.
    let stdout = stdout.trim();
    let object = (stdout.starts_with('{') && stdout.ends_with('}'))
        .then(|| serde_json::from_str::<serde_json::Value>(stdout));
    match exit_code {
        Some(0) => match object {
            Some(object) => object.context("hook printed invalid JSON"),
            None => {
                if !stdout.is_empty() {
                    log::info!("agent hook printed plain text, ignored: {stdout}");
                }
                Ok(json!({}))
            }
        },
        Some(2) => {
            let reason = stderr.trim();
            Ok(match hook_event_name {
                "PreToolUse" => json!({
                    "hookSpecificOutput": {
                        "hookEventName": hook_event_name,
                        "permissionDecision": "deny",
                        "permissionDecisionReason": reason,
                    }
                }),
                _ => json!({ "decision": "block", "reason": reason }),
            })
        }
        // Any other exit code: a valid JSON object decides and the code is ignored.
        _ => match object {
            Some(Ok(object)) => Ok(object),
            _ => anyhow::bail!("hook exited with {exit_code:?}: {}", stderr.trim()),
        },
    }
}

/// Handlers that run when `matcher` matches, with the Claude Code syntax.
pub struct HookMatcherGroup {
    pub matcher: Option<String>,
    pub hooks: Vec<Arc<dyn HookHandler>>,
}

/// Whether a matcher selects `value`, with the Claude Code syntax: `*`, empty
/// or absent selects everything; letters, digits, `_`, `-`, spaces, `,` and
/// `|` list exact names; anything else is an unanchored regular expression.
fn matcher_matches(matcher: Option<&str>, value: &str) -> bool {
    let matcher = matcher.unwrap_or_default();
    if matcher.is_empty() || matcher == "*" {
        return true;
    }
    if is_name_matcher(matcher) {
        return matcher.split([',', '|']).any(|name| name.trim() == value);
    }
    match Regex::new(matcher) {
        Ok(regex) => regex.is_match(value),
        Err(error) => {
            log::warn!("invalid agent hook matcher {matcher:?}: {error}");
            false
        }
    }
}

/// Whether a matcher is `*`, empty, or a list of exact names rather than a
/// regular expression.
fn is_name_matcher(matcher: &str) -> bool {
    matcher == "*"
        || matcher
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | ' ' | ',' | '|'))
}

/// The hooks whose matcher selects `value`, in declaration order. A handler
/// declared in several matching groups runs once, as in Claude Code.
fn matching_hooks(groups: &[HookMatcherGroup], value: &str) -> Vec<Arc<dyn HookHandler>> {
    let mut seen = HashSet::default();
    groups
        .iter()
        .filter(|group| matcher_matches(group.matcher.as_deref(), value))
        .flat_map(|group| group.hooks.iter().cloned())
        .filter(|hook| hook.dedup_key().is_none_or(|key| seen.insert(key)))
        .collect()
}

/// Calls `f` with the configured hooks, if any load.
fn with_configured_hooks<R>(cx: &App, f: impl FnOnce(&AgentHooks) -> R) -> Option<R> {
    // A global, when set, replaces the `agent.hooks` setting; tests use it to
    // install in-process hooks.
    if let Some(hooks) = cx.try_global::<AgentHooks>() {
        return Some(f(hooks));
    }
    let settings = AgentSettings::get_global(cx).hooks.as_ref()?;
    let data_root = cx
        .try_global::<HookDataRoot>()
        .map(|root| root.0.clone())
        .unwrap_or_else(|| paths::data_dir().join("hooks"));
    match AgentHooks::from_settings(settings, &data_root) {
        Ok(hooks) => Some(f(&hooks)),
        Err(error) => {
            log::error!("agent hooks are not loaded: {error:#}");
            None
        }
    }
}

/// Wraps every tool that a configured tool hook matches in a [`HookedTool`].
pub(crate) fn wrap_tools(
    thread: &Thread,
    tools: &mut BTreeMap<SharedString, Arc<dyn AnyAgentTool>>,
    cx: &App,
) {
    with_configured_hooks(cx, |hooks| wrap_matching_tools(hooks, thread, tools, cx));
}

fn wrap_matching_tools(
    hooks: &AgentHooks,
    thread: &Thread,
    tools: &mut BTreeMap<SharedString, Arc<dyn AnyAgentTool>>,
    cx: &App,
) {
    let context = HookContext::for_thread(thread, cx);
    for (tool_name, tool) in tools.iter_mut() {
        let pre_tool_use = matching_hooks(&hooks.pre_tool_use, tool_name);
        let post_tool_use = matching_hooks(&hooks.post_tool_use, tool_name);
        if pre_tool_use.is_empty() && post_tool_use.is_empty() {
            continue;
        }
        *tool = Arc::new(HookedTool::new(
            tool_name.clone(),
            tool.clone(),
            context.clone(),
            pre_tool_use,
            post_tool_use,
        ));
    }
}

impl HookContext {
    fn for_thread(thread: &Thread, cx: &App) -> Self {
        let thread_id = thread.id().to_string();
        let parent_thread_id = thread.parent_thread_id().map(|id| id.to_string());
        let cwd = thread
            .project()
            .read(cx)
            .visible_worktrees(cx)
            .next()
            .map(|worktree| worktree.read(cx).abs_path().to_string_lossy().into_owned())
            .unwrap_or_default();
        Self {
            // The thread only knows its parent, so a subagent below depth 1
            // reports its parent instead of the root conversation.
            session_id: parent_thread_id
                .clone()
                .unwrap_or_else(|| thread_id.clone()),
            cwd,
            thread_id,
            parent_thread_id,
            depth: thread.depth().into(),
        }
    }

    /// Builds a hook input from the shared fields and the event's own fields.
    fn input(&self, hook_event_name: &str, event_fields: serde_json::Value) -> serde_json::Value {
        let mut input = json!({
            "session_id": self.session_id,
            "transcript_path": null,
            "cwd": self.cwd,
            "hook_event_name": hook_event_name,
            "zed": {
                "contract_version": HOOK_CONTRACT_VERSION,
                "thread_id": self.thread_id,
                "parent_thread_id": self.parent_thread_id,
                "is_subagent": self.parent_thread_id.is_some(),
                "depth": self.depth,
                "timestamp": chrono::Utc::now().to_rfc3339(),
            },
        });
        if let (Some(input), serde_json::Value::Object(event_fields)) =
            (input.as_object_mut(), event_fields)
        {
            input.extend(event_fields);
        }
        input
    }
}

enum HookOutcome {
    Output(serde_json::Value),
    /// The hook failed: the action continues as if it were not there.
    Failed,
    /// The user stopped the tool call while the hook ran.
    Cancelled,
}

/// Runs a hook until it answers or the user cancels the tool call. Dropping
/// the hook's task on cancellation kills its process.
async fn call_hook(
    hook: &Arc<dyn HookHandler>,
    input: serde_json::Value,
    event_stream: &ToolCallEventStream,
    cx: &mut AsyncApp,
) -> HookOutcome {
    let task = cx.update(|cx| hook.call(input, cx));
    futures::select_biased! {
        _ = event_stream.cancelled_by_user().fuse() => HookOutcome::Cancelled,
        result = task.fuse() => match result {
            Ok(output) => HookOutcome::Output(output),
            Err(error) => {
                log::warn!("agent hook failed, continuing: {error:#}");
                HookOutcome::Failed
            }
        },
    }
}

/// The `PreToolUse` decision of a hook output. Claude Code's deprecated
/// top-level `decision` still counts (`block` denies, `approve` allows); when
/// both are set, the most restrictive one wins: deny, then ask, then allow.
fn pre_tool_use_decision(output: &serde_json::Value) -> Option<&str> {
    let legacy = match output["decision"].as_str() {
        Some("block") => Some("deny"),
        Some("approve") => Some("allow"),
        _ => None,
    };
    let current = output["hookSpecificOutput"]["permissionDecision"].as_str();
    ["deny", "ask", "allow"]
        .into_iter()
        .find(|decision| legacy == Some(*decision) || current == Some(*decision))
}

/// Logs the output fields this version of Zed reads but does not apply yet.
fn log_unsupported_fields(output: &serde_json::Value) {
    for (field, value) in [
        ("continue", &output["continue"]),
        ("suppressOutput", &output["suppressOutput"]),
        (
            "hookSpecificOutput.updatedToolOutput",
            &output["hookSpecificOutput"]["updatedToolOutput"],
        ),
    ] {
        if !value.is_null() {
            log::info!("agent hook output field `{field}` is not supported yet; ignored");
        }
    }
}

fn cancelled_output() -> AgentToolOutput {
    anyhow!("Tool call cancelled by user").into()
}

/// The result the model gets for a call a hook refused: the reason first,
/// then the notes of the hooks that ran before.
fn refused_output(reason: String, notes: Vec<String>) -> AgentToolOutput {
    let mut output = AgentToolOutput::from(anyhow!(reason));
    append_notes(&mut output, notes);
    output
}

/// Whether `input` matches the tool's input schema. A schema that cannot be
/// compiled accepts everything; the tool still checks its own input.
fn input_matches_schema(schema: &serde_json::Value, input: &serde_json::Value) -> bool {
    match jsonschema::validator_for(schema) {
        Ok(validator) => validator.is_valid(input),
        Err(error) => {
            log::warn!("cannot check a hook's updatedInput against the tool schema: {error}");
            true
        }
    }
}

/// Forwards a streamed tool input as it arrives and returns the full input.
async fn forward_tool_input(
    mut input: ToolInput<serde_json::Value>,
    mut sender: ToolInputSender,
) -> Option<serde_json::Value> {
    loop {
        match input.next().await {
            Ok(ToolInputPayload::Partial(partial)) => sender.send_partial(partial),
            Ok(ToolInputPayload::Full(full)) => {
                sender.send_full(full.clone());
                return Some(full);
            }
            Ok(ToolInputPayload::InvalidJson { error_message }) => {
                sender.send_invalid_json(error_message);
                return None;
            }
            Err(_) => return None,
        }
    }
}

/// The tool result as the model will receive it, in the contract's
/// `ToolResponse` shape.
fn tool_response(is_error: bool, output: &AgentToolOutput) -> serde_json::Value {
    let content = output
        .llm_output
        .iter()
        .map(|part| match part {
            LanguageModelToolResultContent::Text(text) => json!({ "type": "text", "text": text }),
            LanguageModelToolResultContent::Image(image) => {
                json!({ "type": "image", "byte_length": image.len() })
            }
        })
        .collect::<Vec<_>>();
    json!({ "is_error": is_error, "content": content })
}

fn append_notes(output: &mut AgentToolOutput, notes: Vec<String>) {
    output.llm_output.extend(
        notes
            .into_iter()
            .map(|note| LanguageModelToolResultContent::Text(note.into())),
    );
}

/// Wraps a tool so that `PreToolUse` hooks run before it and `PostToolUse`
/// hooks run after it.
pub struct HookedTool {
    tool_name: SharedString,
    inner: Arc<dyn AnyAgentTool>,
    context: HookContext,
    pre_tool_use: Vec<Arc<dyn HookHandler>>,
    post_tool_use: Vec<Arc<dyn HookHandler>>,
}

impl HookedTool {
    pub fn new(
        tool_name: SharedString,
        inner: Arc<dyn AnyAgentTool>,
        context: HookContext,
        pre_tool_use: Vec<Arc<dyn HookHandler>>,
        post_tool_use: Vec<Arc<dyn HookHandler>>,
    ) -> Self {
        Self {
            tool_name,
            inner,
            context,
            pre_tool_use,
            post_tool_use,
        }
    }
}

impl AnyAgentTool for HookedTool {
    fn name(&self) -> SharedString {
        self.inner.name()
    }

    fn description(&self) -> SharedString {
        self.inner.description()
    }

    fn kind(&self) -> acp_v2::ToolKind {
        self.inner.kind()
    }

    fn initial_title(&self, input: serde_json::Value, cx: &mut App) -> SharedString {
        self.inner.initial_title(input, cx)
    }

    fn input_schema(&self) -> serde_json::Value {
        self.inner.input_schema()
    }

    fn supports_input_streaming(&self) -> bool {
        self.inner.supports_input_streaming()
    }

    fn supports_provider(&self, provider: &LanguageModelProviderId) -> bool {
        self.inner.supports_provider(provider)
    }

    fn allow_in_restricted_mode(&self) -> bool {
        self.inner.allow_in_restricted_mode()
    }

    fn run(
        self: Arc<Self>,
        input: ToolInput<serde_json::Value>,
        event_stream: ToolCallEventStream,
        cx: &mut App,
    ) -> Task<Result<AgentToolOutput, AgentToolOutput>> {
        cx.spawn(async move |cx| {
            let tool_use_id = event_stream.tool_use_id().to_string();
            let mut notes = Vec::new();

            let (tool_input, started_at, result) = if self.pre_tool_use.is_empty() {
                // No hook can change the input: forward it as it streams, so
                // streaming tools keep their live preview.
                let (sender, forwarded) = ToolInputSender::channel();
                let full_input = cx.spawn(async move |_| forward_tool_input(input, sender).await);
                let started_at = Instant::now();
                let result = cx
                    .update(|cx| self.inner.clone().run(forwarded, event_stream.clone(), cx))
                    .await;
                let tool_input = full_input.await.unwrap_or_default();
                (tool_input, started_at, result)
            } else {
                let mut tool_input = input.recv().await.map_err(AgentToolOutput::from)?;
                // Hooks run one after the other in declaration order: each one
                // sees the input as corrected by the previous ones, the first
                // refusal wins, and the notes for the model accumulate in order.
                for hook in &self.pre_tool_use {
                    let hook_input = self.context.input(
                        "PreToolUse",
                        json!({
                            "tool_name": self.tool_name,
                            "tool_input": tool_input,
                            "tool_use_id": tool_use_id,
                        }),
                    );
                    let output = match call_hook(hook, hook_input, &event_stream, cx).await {
                        HookOutcome::Output(output) => output,
                        HookOutcome::Failed => continue,
                        HookOutcome::Cancelled => return Err(cancelled_output()),
                    };
                    if let Err(error) = check_event_name("PreToolUse", &output) {
                        log::warn!("PreToolUse hook output ignored: {error:#}");
                        continue;
                    }
                    log_unsupported_fields(&output);
                    let specific = &output["hookSpecificOutput"];
                    let context = specific["additionalContext"].as_str();
                    // The decision comes first: a refusal stands whatever else
                    // the hook returned.
                    let reason = specific["permissionDecisionReason"]
                        .as_str()
                        .or(output["reason"].as_str())
                        .filter(|reason| !reason.trim().is_empty());
                    match pre_tool_use_decision(&output) {
                        Some("deny") => {
                            notes.extend(context.map(str::to_string));
                            let reason = reason.unwrap_or("A hook denied this tool call.");
                            return Err(refused_output(reason.to_string(), notes));
                        }
                        // Until Zed can ask the user from a hook, asking refuses.
                        Some("ask") => {
                            log::info!(
                                "agent hook asked to confirm a {} call; refused",
                                self.tool_name
                            );
                            notes.extend(context.map(str::to_string));
                            let mut message = "A hook asked to confirm this tool call, which Zed cannot do yet, so the call was refused."
                                .to_string();
                            if let Some(reason) = reason {
                                message.push_str(&format!(" Reason: {reason}"));
                            }
                            return Err(refused_output(message, notes));
                        }
                        _ => {}
                    }
                    // Scripts often print `null` for a field they do not set.
                    let updated_input = specific.get("updatedInput").filter(|v| !v.is_null());
                    if let Some(updated_input) = updated_input
                        && !(updated_input.is_object()
                            && input_matches_schema(&self.inner.input_schema(), updated_input))
                    {
                        log::warn!(
                            "agent hook returned an updatedInput that does not match the {} tool's schema; the hook counts as failed",
                            self.tool_name
                        );
                        continue;
                    }
                    notes.extend(context.map(str::to_string));
                    if let Some(updated_input) = updated_input {
                        tool_input = updated_input.clone();
                    }
                }

                let started_at = Instant::now();
                let result = cx
                    .update(|cx| {
                        self.inner.clone().run(
                            ToolInput::ready(tool_input.clone()),
                            event_stream.clone(),
                            cx,
                        )
                    })
                    .await;
                (tool_input, started_at, result)
            };
            let duration_ms = started_at.elapsed().as_millis() as u64;
            let (is_error, mut output) = match result {
                Ok(output) => (false, output),
                Err(output) => (true, output),
            };

            for hook in &self.post_tool_use {
                let hook_input = self.context.input(
                    "PostToolUse",
                    json!({
                        "tool_name": self.tool_name,
                        "tool_input": tool_input,
                        "tool_response": tool_response(is_error, &output),
                        "tool_use_id": tool_use_id,
                        "duration_ms": duration_ms,
                    }),
                );
                let hook_output = match call_hook(hook, hook_input, &event_stream, cx).await {
                    HookOutcome::Output(output) => output,
                    HookOutcome::Failed => continue,
                    // The tool already ran: keep its result, skip the other hooks.
                    HookOutcome::Cancelled => break,
                };
                if let Err(error) = check_event_name("PostToolUse", &hook_output) {
                    log::warn!("PostToolUse hook output ignored: {error:#}");
                    continue;
                }
                log_unsupported_fields(&hook_output);
                if hook_output["decision"] == "block"
                    && let Some(reason) = hook_output["reason"]
                        .as_str()
                        .filter(|reason| !reason.trim().is_empty())
                {
                    notes.push(reason.to_string());
                }
                if let Some(context) =
                    hook_output["hookSpecificOutput"]["additionalContext"].as_str()
                {
                    notes.push(context.to_string());
                }
            }

            append_notes(&mut output, notes);
            if is_error { Err(output) } else { Ok(output) }
        })
    }

    fn replay(
        &self,
        input: serde_json::Value,
        output: serde_json::Value,
        event_stream: ToolCallEventStream,
        cx: &mut App,
    ) -> Result<()> {
        self.inner.replay(input, output, event_stream, cx)
    }
}

/// Streams a completion of `request` like `provider.stream_completion`, and
/// runs the `BeforeModel` hooks before the request and the `AfterModel`
/// hooks when the response ends. Cancelling through `cancellation` while the
/// `BeforeModel` hooks run abandons them and sends no request.
pub(crate) async fn stream_completion(
    provider: &Arc<dyn LanguageModelProvider>,
    model: &LanguageModel,
    request: LanguageModelRequest,
    thread: &WeakEntity<Thread>,
    cancellation: Option<watch::Receiver<bool>>,
    cx: &AsyncApp,
) -> Result<LanguageModelCompletionStream, LanguageModelCompletionError> {
    match cx.update(|cx| ModelHooks::for_request(thread, &request, cx)) {
        Some(hooks) => {
            hooks
                .stream_completion(provider, model, request, cancellation, cx)
                .await
        }
        None => provider.stream_completion(model, request, cx).await,
    }
}

/// The hooks that run around one model request.
struct ModelHooks {
    context: HookContext,
    before_model: Vec<Arc<dyn HookHandler>>,
    after_model: Vec<Arc<dyn HookHandler>>,
}

impl ModelHooks {
    fn for_request(
        thread: &WeakEntity<Thread>,
        request: &LanguageModelRequest,
        cx: &mut App,
    ) -> Option<Self> {
        let intent = intent_name(request.intent).unwrap_or_default();
        let (before_model, after_model) = with_configured_hooks(cx, |hooks| {
            (
                matching_hooks(&hooks.before_model, intent),
                matching_hooks(&hooks.after_model, intent),
            )
        })?;
        if before_model.is_empty() && after_model.is_empty() {
            return None;
        }
        let thread = thread.upgrade()?;
        let context = HookContext::for_thread(thread.read(cx), cx);
        if !before_model.is_empty() {
            forget_sent_messages_on_release(&thread, &context.thread_id, cx);
        }
        Some(Self {
            context,
            before_model,
            after_model,
        })
    }

    async fn stream_completion(
        self,
        provider: &Arc<dyn LanguageModelProvider>,
        model: &LanguageModel,
        mut request: LanguageModelRequest,
        cancellation: Option<watch::Receiver<bool>>,
        cx: &AsyncApp,
    ) -> Result<LanguageModelCompletionStream, LanguageModelCompletionError> {
        let request_fields = json!({
            "request_id": Uuid::new_v4().to_string(),
            "intent": intent_name(request.intent),
            "model": {
                "provider": model.provider_id().to_string(),
                "id": model.id().0.to_string(),
            },
            "is_compaction": is_compaction_request(&request),
        });
        if !self.before_model.is_empty() {
            let cancelled = {
                let before_model = self.run_before_model(&request_fields, model, &mut request, cx);
                let cancellation = wait_for_cancellation(cancellation);
                futures::pin_mut!(before_model, cancellation);
                matches!(
                    futures::future::select(before_model, cancellation).await,
                    futures::future::Either::Right(_)
                )
            };
            // Dropping the hooks' tasks kills their processes; the caller sees
            // the cancellation and no request leaves.
            if cancelled {
                return Ok(futures::stream::empty().boxed());
            }
        }

        let started_at = Instant::now();
        let stream = provider.stream_completion(model, request, cx).await?;
        if self.after_model.is_empty() {
            return Ok(stream);
        }
        let (end_tx, end_rx) = oneshot::channel();
        cx.spawn(async move |cx| {
            // The sender is dropped when the stream fails or is abandoned:
            // AfterModel only observes generations that end normally.
            let Ok(end) = end_rx.await else {
                log::info!("AfterModel hooks skipped: the response stream did not end normally");
                return;
            };
            self.run_after_model(request_fields, end, started_at, cx)
                .await;
        })
        .detach();
        Ok(observe_response_end(stream, end_tx))
    }

    /// Runs the `BeforeModel` hooks in declaration order and inserts their
    /// texts into the system message of this request only.
    async fn run_before_model(
        &self,
        request_fields: &serde_json::Value,
        model: &LanguageModel,
        request: &mut LanguageModelRequest,
        cx: &AsyncApp,
    ) {
        let messages = request
            .messages
            .iter()
            .map(message_to_contract)
            .collect::<Vec<_>>();
        // Only the conversation counts: the system message changes with the
        // model, the date or the tools.
        let conversation = messages
            .iter()
            .zip(&request.messages)
            .filter(|(_, message)| message.role != Role::System)
            .map(|(contract, _)| contract)
            .collect::<Vec<_>>();
        let conversation_hashes = conversation
            .iter()
            .map(|message| {
                let mut hasher = DefaultHasher::new();
                message.to_string().hash(&mut hasher);
                hasher.finish()
            })
            .collect::<Vec<_>>();
        let mut tool_names = request
            .tools
            .iter()
            .map(|tool| tool.name.as_str())
            .collect::<Vec<_>>();
        tool_names.sort_unstable();
        let mut event_fields = request_fields.clone();
        if let Some(fields) = event_fields.as_object_mut() {
            fields.insert("total_message_count".into(), json!(messages.len()));
            fields.insert(
                "max_output_tokens".into(),
                json!(request.effective_max_output_tokens(model.max_output_tokens())),
            );
            fields.insert("tool_names".into(), json!(tool_names));
        }

        let mut inserts = Vec::new();
        let mut insert_chars = 0;
        let mut capped = false;
        for hook in &self.before_model {
            // Each hook gets the messages added since its own previous answer.
            let sent_key = (self.context.thread_id.clone(), hook_identity(hook));
            let known = cx.update(|cx| known_message_count(&sent_key, &conversation_hashes, cx));
            let mut hook_fields = event_fields.clone();
            if let Some(fields) = hook_fields.as_object_mut() {
                fields.insert("new_messages".into(), json!(conversation[known..]));
                fields.insert("known_message_count".into(), json!(known));
            }
            let input = self.context.input("BeforeModel", hook_fields);
            let output = match cx.update(|cx| hook.call(input, cx)).await {
                Ok(output) => output,
                Err(error) => {
                    log::warn!("BeforeModel hook failed, continuing: {error:#}");
                    continue;
                }
            };
            let insert = match check_model_hook_output("BeforeModel", &output)
                .and_then(|()| system_prompt_insert(&output))
            {
                Ok(insert) => insert,
                Err(error) => {
                    log::warn!("BeforeModel hook output ignored: {error:#}");
                    continue;
                }
            };
            // The hook took these messages: the next request sends it only the
            // ones that follow. After a failure, they are sent again.
            cx.update(|cx| {
                cx.default_global::<SentMessages>()
                    .by_hook
                    .insert(sent_key, conversation_hashes.clone());
            });
            log_unsupported_output_fields(
                &output,
                &[
                    "/systemMessage",
                    "/hookSpecificOutput/max_output_tokens",
                    "/hookSpecificOutput/tool_result_replacements",
                ],
            );
            let Some(insert) = insert else {
                continue;
            };
            // The cap applies to the joined text of all hooks.
            let separator = if inserts.is_empty() { 0 } else { 2 };
            let added = insert.chars().count() + separator;
            if capped || insert_chars + added > MAX_SYSTEM_PROMPT_INSERT_CHARS {
                capped = true;
                log::warn!(
                    "BeforeModel texts exceed {MAX_SYSTEM_PROMPT_INSERT_CHARS} characters \
                     together; the text of this hook is ignored"
                );
                continue;
            }
            insert_chars += added;
            inserts.push(insert.to_string());
        }
        // A request without a system message (the panel summary) has no place
        // for the text.
        if !inserts.is_empty() && applies_system_prompt_insert(request) {
            match insert_into_system_message(request, &inserts.join("\n\n")) {
                InsertOutcome::BeforeHeading | InsertOutcome::NoSystemMessage => {}
                InsertOutcome::AtEnd => log::info!(
                    "the system message has no `{SYSTEM_INFORMATION_HEADING}` heading; \
                     the BeforeModel text was added at its end"
                ),
            }
        }
    }

    /// Runs the `AfterModel` hooks in declaration order. They observe the
    /// generation; this version of Zed applies nothing they return.
    async fn run_after_model(
        &self,
        mut event_fields: serde_json::Value,
        end: ResponseEnd,
        started_at: Instant,
        cx: &AsyncApp,
    ) {
        if let Some(fields) = event_fields.as_object_mut() {
            fields.insert(
                "stop_reason".into(),
                json!(end.stop_reason.map(stop_reason_name)),
            );
            fields.insert(
                "usage".into(),
                json!({
                    "input_tokens": end.usage.input_tokens,
                    "output_tokens": end.usage.output_tokens,
                    "cache_creation_input_tokens": end.usage.cache_creation_input_tokens,
                    "cache_read_input_tokens": end.usage.cache_read_input_tokens,
                }),
            );
            fields.insert(
                "duration_ms".into(),
                json!(end.ended_at.duration_since(started_at).as_millis() as u64),
            );
        }
        for hook in &self.after_model {
            let input = self.context.input("AfterModel", event_fields.clone());
            let output = match cx.update(|cx| hook.call(input, cx)).await {
                Ok(output) => output,
                Err(error) => {
                    log::warn!("AfterModel hook failed: {error:#}");
                    continue;
                }
            };
            match check_model_hook_output("AfterModel", &output) {
                Ok(()) => log_unsupported_output_fields(
                    &output,
                    &["/systemMessage", "/hookSpecificOutput/usage"],
                ),
                Err(error) => log::warn!("AfterModel hook output ignored: {error:#}"),
            }
        }
    }
}

/// Resolves once `cancellation` is set; never without one.
async fn wait_for_cancellation(cancellation: Option<watch::Receiver<bool>>) {
    if let Some(mut cancellation) = cancellation {
        loop {
            if *cancellation.borrow() {
                return;
            }
            if cancellation.changed().await.is_err() {
                break;
            }
        }
    }
    futures::future::pending::<()>().await
}

/// Model hooks never refuse: a refusal (exit code 2 or a `decision`) is a
/// failure, as is an output for another event.
fn check_model_hook_output(event: &str, output: &serde_json::Value) -> Result<()> {
    if let Some(decision) = output.get("decision").filter(|value| !value.is_null()) {
        anyhow::bail!(
            "a {event} hook cannot refuse (decision {decision}); its reason: {}",
            output["reason"].as_str().unwrap_or("none given")
        );
    }
    check_event_name(event, output)
}

/// An output whose `hookSpecificOutput` names another event is a failure.
fn check_event_name(event: &str, output: &serde_json::Value) -> Result<()> {
    match output.get("hookSpecificOutput") {
        None | Some(serde_json::Value::Null) => Ok(()),
        Some(specific) => {
            anyhow::ensure!(
                specific["hookEventName"] == event,
                "the output is for {}, not {event}",
                specific["hookEventName"]
            );
            Ok(())
        }
    }
}

/// How a model response ended, as `AfterModel` reports it.
struct ResponseEnd {
    stop_reason: Option<StopReason>,
    usage: TokenUsage,
    ended_at: Instant,
}

/// Passes the response through unchanged, and sends how it ended once the
/// stream finishes without error.
fn observe_response_end(
    stream: LanguageModelCompletionStream,
    end_tx: oneshot::Sender<ResponseEnd>,
) -> LanguageModelCompletionStream {
    let end = ResponseEnd {
        stop_reason: None,
        usage: TokenUsage::default(),
        ended_at: Instant::now(),
    };
    futures::stream::unfold(
        (stream, Some((end_tx, end))),
        |(mut stream, mut end)| async move {
            let Some(event) = stream.next().await else {
                if let Some((end_tx, mut end)) = end.take() {
                    end.ended_at = Instant::now();
                    end_tx.send(end).ok();
                }
                return None;
            };
            match (&event, end.as_mut()) {
                (Ok(LanguageModelCompletionEvent::Stop(reason)), Some((_, end))) => {
                    end.stop_reason = Some(*reason);
                }
                (Ok(LanguageModelCompletionEvent::UsageUpdate(usage)), Some((_, end))) => {
                    end.usage = *usage;
                }
                (Err(_), _) => end = None,
                _ => {}
            }
            Some((event, (stream, end)))
        },
    )
    .boxed()
}

/// What each `BeforeModel` hook of each thread already took, so that it gets
/// only the new messages.
#[derive(Default)]
struct SentMessages {
    /// By thread id and hook identity: a hash of each non-system message of
    /// the last request the hook answered.
    by_hook: HashMap<(String, String), Vec<u64>>,
    /// Threads whose release clears their entries.
    watched_threads: HashSet<String>,
}

impl Global for SentMessages {}

/// Identifies a hook across requests: its configuration for a command hook,
/// which is rebuilt from the settings on each request.
fn hook_identity(hook: &Arc<dyn HookHandler>) -> String {
    hook.dedup_key()
        .unwrap_or_else(|| format!("{:p}", Arc::as_ptr(hook)))
}

/// How many of the non-system `messages` this hook of this thread already
/// took, unchanged and in order, whatever the request and the model.
fn known_message_count(sent_key: &(String, String), messages: &[u64], cx: &mut App) -> usize {
    cx.default_global::<SentMessages>()
        .by_hook
        .get(sent_key)
        .map_or(0, |sent| {
            sent.iter()
                .zip(messages)
                .take_while(|(sent, current)| sent == current)
                .count()
        })
}

/// Clears what the hooks of a thread received once the thread is released.
fn forget_sent_messages_on_release(thread: &Entity<Thread>, thread_id: &str, cx: &mut App) {
    if !cx
        .default_global::<SentMessages>()
        .watched_threads
        .insert(thread_id.to_string())
    {
        return;
    }
    let released_thread_id = thread_id.to_string();
    cx.observe_release(thread, move |_, cx| {
        let sent = cx.default_global::<SentMessages>();
        sent.by_hook
            .retain(|(thread_id, _), _| *thread_id != released_thread_id);
        sent.watched_threads.remove(&released_thread_id);
    })
    .detach();
}

#[cfg(test)]
pub(crate) fn remembered_thread_count(cx: &mut App) -> usize {
    cx.default_global::<SentMessages>()
        .by_hook
        .keys()
        .map(|(thread_id, _)| thread_id)
        .collect::<HashSet<_>>()
        .len()
}

const SYSTEM_INFORMATION_HEADING: &str = "## System Information";

/// Where a `BeforeModel` text went.
#[derive(Debug, PartialEq)]
enum InsertOutcome {
    BeforeHeading,
    AtEnd,
    NoSystemMessage,
}

/// Inserts `text` into the system message, right before its
/// `## System Information` heading, or at its end when the heading is
/// missing.
fn insert_into_system_message(request: &mut LanguageModelRequest, text: &str) -> InsertOutcome {
    let Some(system) = request
        .messages
        .iter_mut()
        .find(|message| message.role == Role::System)
    else {
        return InsertOutcome::NoSystemMessage;
    };
    for part in &mut system.content {
        if let MessageContent::Text(content) = part
            && let Some(ix) = content
                .match_indices(SYSTEM_INFORMATION_HEADING)
                .map(|(ix, _)| ix)
                .find(|&ix| ix == 0 || content[..ix].ends_with('\n'))
        {
            content.insert_str(ix, &format!("{text}\n\n"));
            return InsertOutcome::BeforeHeading;
        }
    }
    let last_text = system.content.iter_mut().rev().find_map(|part| match part {
        MessageContent::Text(content) => Some(content),
        _ => None,
    });
    match last_text {
        Some(content) => content.push_str(&format!("\n\n{text}")),
        None => system.content.push(MessageContent::Text(text.to_string())),
    }
    InsertOutcome::AtEnd
}

/// Whether the request is the compaction summary, recognised by its
/// instruction rather than by its intent alone.
fn is_compaction_request(request: &LanguageModelRequest) -> bool {
    request.intent == Some(CompletionIntent::ThreadContextSummarization)
        && request.messages.last().is_some_and(|message| {
            message
                .content
                .iter()
                .any(|part| matches!(part, MessageContent::Text(text) if text == COMPACTION_PROMPT))
        })
}

/// Whether a `system_prompt_insert` applies to this request: not to the
/// compaction summary.
fn applies_system_prompt_insert(request: &LanguageModelRequest) -> bool {
    !is_compaction_request(request)
}

/// The contract's names of the intents a model hook can match.
const MODEL_INTENTS: [&str; 10] = [
    "user_prompt",
    "subagent",
    "tool_results",
    "thread_summarization",
    "thread_context_summarization",
    "create_file",
    "edit_file",
    "inline_assist",
    "terminal_inline_assist",
    "generate_git_commit_message",
];

/// The contract's snake_case name of a completion intent.
fn intent_name(intent: Option<CompletionIntent>) -> Option<&'static str> {
    Some(match intent? {
        CompletionIntent::UserPrompt => "user_prompt",
        CompletionIntent::Subagent => "subagent",
        CompletionIntent::ToolResults => "tool_results",
        CompletionIntent::ThreadSummarization => "thread_summarization",
        CompletionIntent::ThreadContextSummarization => "thread_context_summarization",
        CompletionIntent::CreateFile => "create_file",
        CompletionIntent::EditFile => "edit_file",
        CompletionIntent::InlineAssist => "inline_assist",
        CompletionIntent::TerminalInlineAssist => "terminal_inline_assist",
        CompletionIntent::GenerateGitCommitMessage => "generate_git_commit_message",
    })
}

fn stop_reason_name(reason: StopReason) -> &'static str {
    match reason {
        StopReason::EndTurn => "end_turn",
        StopReason::MaxTokens => "max_tokens",
        StopReason::ToolUse => "tool_use",
        StopReason::Refusal => "refusal",
    }
}

/// A request message in the contract's provider-neutral shape.
fn message_to_contract(message: &LanguageModelRequestMessage) -> serde_json::Value {
    let role = match message.role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
    };
    let content = message
        .content
        .iter()
        .map(|part| match part {
            MessageContent::Text(text) => json!({ "type": "text", "text": text }),
            MessageContent::Thinking { text, .. } => json!({ "type": "thinking", "text": text }),
            MessageContent::RedactedThinking(_) => json!({ "type": "redacted_thinking" }),
            MessageContent::Image(image) => json!({ "type": "image", "byte_length": image.len() }),
            MessageContent::ToolUse(tool_use) => {
                let input = match &tool_use.input {
                    LanguageModelToolUseInput::Json(input) if input.is_object() => input.clone(),
                    LanguageModelToolUseInput::Json(input) => json!({ "input": input }),
                    LanguageModelToolUseInput::Text(input) => json!({ "input": input }),
                };
                json!({
                    "type": "tool_use",
                    "id": tool_use.id.to_string(),
                    "name": tool_use.name.to_string(),
                    "input": input,
                })
            }
            MessageContent::ToolResult(result) => {
                // Image results are reduced to an empty string.
                let text = result
                    .content
                    .iter()
                    .filter_map(|part| match part {
                        LanguageModelToolResultContent::Text(text) => Some(text.as_ref()),
                        LanguageModelToolResultContent::Image(_) => None,
                    })
                    .collect::<String>();
                json!({
                    "type": "tool_result",
                    "tool_use_id": result.tool_use_id.to_string(),
                    "tool_name": result.tool_name.to_string(),
                    "is_error": result.is_error,
                    "text": text,
                })
            }
            MessageContent::Compaction(CompactedContext::Summary { content, .. }) => {
                json!({ "type": "text", "text": content.as_ref() })
            }
            // Only the provider can read its own compacted state.
            MessageContent::Compaction(CompactedContext::ProviderState(_)) => {
                json!({ "type": "text", "text": "[Context compacted by the provider]" })
            }
        })
        .collect::<Vec<_>>();
    json!({ "role": role, "content": content })
}

/// The `system_prompt_insert` of a `BeforeModel` output. An empty text, or
/// one over 10,000 characters, makes the output invalid.
fn system_prompt_insert(output: &serde_json::Value) -> Result<Option<&str>> {
    let insert = match output.pointer("/hookSpecificOutput/system_prompt_insert") {
        None | Some(serde_json::Value::Null) => return Ok(None),
        Some(insert) => insert
            .as_str()
            .context("`system_prompt_insert` must be a string")?,
    };
    anyhow::ensure!(!insert.is_empty(), "`system_prompt_insert` is empty");
    let length = insert.chars().count();
    anyhow::ensure!(
        length <= MAX_SYSTEM_PROMPT_INSERT_CHARS,
        "`system_prompt_insert` has {length} characters, more than \
         {MAX_SYSTEM_PROMPT_INSERT_CHARS}"
    );
    Ok(Some(insert))
}

/// The cap Claude Code puts on `additionalContext`: the text is resent with
/// every request.
const MAX_SYSTEM_PROMPT_INSERT_CHARS: usize = 10_000;

/// Logs the output fields, given as JSON pointers, that this version of Zed
/// reads but does not apply yet.
fn log_unsupported_output_fields(output: &serde_json::Value, pointers: &[&str]) {
    for pointer in pointers {
        if output
            .pointer(pointer)
            .is_some_and(|value| !value.is_null())
        {
            log::info!("agent hook output field `{pointer}` is not supported yet; ignored");
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    use gpui::TestAppContext;
    use language_model::{LanguageModelImage, LanguageModelToolResult, LanguageModelToolUse};
    use serde_json::json;
    use std::{
        cell::{Cell, RefCell},
        rc::Rc,
        sync::atomic::{AtomicBool, Ordering},
    };

    /// A tool named `echo` that returns its `text`, and records how often it
    /// ran and how many partial inputs it received before the full one.
    #[derive(Default)]
    struct RecordingTool {
        runs: Cell<usize>,
        partial_inputs: Cell<usize>,
    }

    impl AnyAgentTool for RecordingTool {
        fn name(&self) -> SharedString {
            "echo".into()
        }

        fn description(&self) -> SharedString {
            "Echoes its text.".into()
        }

        fn kind(&self) -> acp_v2::ToolKind {
            acp_v2::ToolKind::Other
        }

        fn initial_title(&self, _input: serde_json::Value, _cx: &mut App) -> SharedString {
            "Echo".into()
        }

        fn input_schema(&self) -> serde_json::Value {
            json!({
                "type": "object",
                "properties": { "text": { "type": "string" } },
                "required": ["text"],
            })
        }

        fn supports_input_streaming(&self) -> bool {
            true
        }

        fn run(
            self: Arc<Self>,
            mut input: ToolInput<serde_json::Value>,
            _event_stream: ToolCallEventStream,
            cx: &mut App,
        ) -> Task<Result<AgentToolOutput, AgentToolOutput>> {
            self.runs.set(self.runs.get() + 1);
            cx.spawn(async move |_cx| {
                loop {
                    match input.next().await {
                        Ok(ToolInputPayload::Partial(_)) => {
                            self.partial_inputs.set(self.partial_inputs.get() + 1)
                        }
                        Ok(ToolInputPayload::Full(value)) => {
                            let text = value["text"].as_str().unwrap_or_default().to_string();
                            return Ok(AgentToolOutput {
                                llm_output: vec![LanguageModelToolResultContent::Text(
                                    text.clone().into(),
                                )],
                                raw_output: json!(text),
                            });
                        }
                        Ok(ToolInputPayload::InvalidJson { error_message }) => {
                            return Err(anyhow!(error_message).into());
                        }
                        Err(error) => return Err(error.into()),
                    }
                }
            })
        }

        fn replay(
            &self,
            _input: serde_json::Value,
            _output: serde_json::Value,
            _event_stream: ToolCallEventStream,
            _cx: &mut App,
        ) -> Result<()> {
            Ok(())
        }
    }

    /// A hook that never answers, and records when Zed abandons it.
    struct PendingHook {
        abandoned: Arc<AtomicBool>,
    }

    impl HookHandler for PendingHook {
        fn call(&self, _input: serde_json::Value, cx: &mut App) -> Task<Result<serde_json::Value>> {
            let abandoned = self.abandoned.clone();
            cx.background_executor().spawn(async move {
                let _abandoned = util::defer(move || abandoned.store(true, Ordering::SeqCst));
                futures::future::pending::<()>().await;
                unreachable!()
            })
        }
    }

    struct FakeHook {
        response: Result<serde_json::Value, String>,
        calls: Rc<RefCell<Vec<serde_json::Value>>>,
    }

    impl HookHandler for FakeHook {
        fn call(&self, input: serde_json::Value, _cx: &mut App) -> Task<Result<serde_json::Value>> {
            self.calls.borrow_mut().push(input);
            Task::ready(self.response.clone().map_err(anyhow::Error::msg))
        }
    }

    fn fake_hook(
        response: Result<serde_json::Value, String>,
    ) -> (Arc<dyn HookHandler>, Rc<RefCell<Vec<serde_json::Value>>>) {
        let calls = Rc::new(RefCell::new(Vec::new()));
        let hook = Arc::new(FakeHook {
            response,
            calls: calls.clone(),
        });
        (hook, calls)
    }

    fn pre_tool_use(specific: serde_json::Value) -> serde_json::Value {
        let mut specific = specific;
        specific["hookEventName"] = json!("PreToolUse");
        json!({ "hookSpecificOutput": specific })
    }

    fn hooked_echo(
        pre_tool_use: Vec<Arc<dyn HookHandler>>,
        post_tool_use: Vec<Arc<dyn HookHandler>>,
    ) -> (Arc<HookedTool>, Arc<RecordingTool>) {
        let echo = Arc::new(RecordingTool::default());
        let tool = Arc::new(HookedTool::new(
            "echo".into(),
            echo.clone(),
            HookContext {
                session_id: "session-1".into(),
                cwd: "/project".into(),
                thread_id: "session-1".into(),
                parent_thread_id: None,
                depth: 0,
            },
            pre_tool_use,
            post_tool_use,
        ));
        (tool, echo)
    }

    /// Runs the tool with `{"text": text}` and returns whether the model sees an
    /// error, and the text parts it sees.
    async fn run_echo(
        tool: Arc<HookedTool>,
        text: &str,
        cx: &mut TestAppContext,
    ) -> (bool, Vec<String>) {
        let (event_stream, _receiver) = ToolCallEventStream::test();
        let task =
            cx.update(|cx| tool.run(ToolInput::ready(json!({ "text": text })), event_stream, cx));
        let (is_error, output) = match task.await {
            Ok(output) => (false, output),
            Err(output) => (true, output),
        };
        let texts = output
            .llm_output
            .iter()
            .filter_map(|part| match part {
                LanguageModelToolResultContent::Text(text) => Some(text.to_string()),
                LanguageModelToolResultContent::Image(_) => None,
            })
            .collect();
        (is_error, texts)
    }

    #[test]
    fn test_parse_command_output_follows_exit_codes() {
        assert_eq!(
            parse_command_output("PreToolUse", Some(0), r#"{"decision":"block"}"#, "").unwrap(),
            json!({ "decision": "block" })
        );
        assert_eq!(
            parse_command_output("PreToolUse", Some(0), "  \n", "").unwrap(),
            json!({})
        );
        assert_eq!(
            parse_command_output("PreToolUse", Some(2), "", "Use grep instead.\n").unwrap(),
            json!({
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": "Use grep instead.",
                }
            })
        );
        assert_eq!(
            parse_command_output("PostToolUse", Some(2), "", "Run the tests.\n").unwrap(),
            json!({ "decision": "block", "reason": "Run the tests." })
        );
        assert!(parse_command_output("PreToolUse", Some(1), "", "boom").is_err());
        assert!(parse_command_output("PreToolUse", None, "", "").is_err());
        // Plain text is a success without fields; a broken object is a failure.
        assert_eq!(
            parse_command_output("PreToolUse", Some(0), "all good", "").unwrap(),
            json!({})
        );
        assert!(parse_command_output("PreToolUse", Some(0), "{not json}", "").is_err());
        // With any other exit code, a valid JSON object decides and the code is ignored.
        assert_eq!(
            parse_command_output("PostToolUse", Some(1), r#"{"decision":"block"}"#, "").unwrap(),
            json!({ "decision": "block" })
        );
    }

    #[test]
    fn test_hook_data_key_uses_id_or_command_hash() {
        assert_eq!(
            hook_data_key(&json!({ "command": "node", "args": ["garde.mjs"], "id": "garde" }))
                .unwrap(),
            "garde"
        );
        let first = hook_data_key(&json!({ "command": "node", "args": ["a.mjs"] })).unwrap();
        let first_again = hook_data_key(&json!({ "command": "node", "args": ["a.mjs"] })).unwrap();
        let second = hook_data_key(&json!({ "command": "node", "args": ["b.mjs"] })).unwrap();
        assert_eq!(first, first_again);
        assert_ne!(first, second);
        assert_eq!(first.len(), 64);
        assert!(first.chars().all(|c| c.is_ascii_hexdigit()));
        assert!(hook_data_key(&json!({ "command": "node", "id": "Bad Id" })).is_err());
    }

    #[test]
    fn test_duplicate_hook_ids_refuse_hook_settings() {
        let error = AgentHooks::from_settings(
            &json!({
                "PreToolUse": [{
                    "hooks": [{ "type": "command", "command": "node", "args": ["garde.mjs"], "id": "garde" }],
                }],
                "PostToolUse": [{
                    "hooks": [{ "type": "command", "command": "node", "args": ["other.mjs"], "id": "garde" }],
                }],
            }),
            Path::new("/data/hooks"),
        )
        .err()
        .expect("two different hooks with the same id must be refused")
        .to_string();
        assert!(error.contains("garde"), "{error}");
        assert!(error.contains("PreToolUse[0].hooks[0]"), "{error}");
        assert!(error.contains("PostToolUse[0].hooks[0]"), "{error}");

        // The same hook declared on two events keeps one id and one directory.
        let same_hook =
            json!({ "type": "command", "command": "node", "args": ["garde.mjs"], "id": "garde" });
        assert!(
            AgentHooks::from_settings(
                &json!({
                    "PreToolUse": [{ "hooks": [same_hook.clone()] }],
                    "PostToolUse": [{ "hooks": [same_hook] }],
                }),
                Path::new("/data/hooks"),
            )
            .is_ok()
        );
    }

    #[test]
    fn test_agent_hooks_from_settings_reads_command_handlers() {
        let hooks = AgentHooks::from_settings(
            &json!({
                "PreToolUse": [{
                    "matcher": "terminal",
                    "hooks": [{ "type": "command", "command": "check-terminal" }],
                }],
                "PostToolUse": [{
                    "hooks": [{ "type": "command", "command": "bash", "args": ["-c", "true"] }],
                }],
            }),
            Path::new("/data/hooks"),
        )
        .unwrap();

        assert_eq!(hooks.pre_tool_use.len(), 1);
        assert_eq!(hooks.pre_tool_use[0].matcher.as_deref(), Some("terminal"));
        assert_eq!(hooks.pre_tool_use[0].hooks.len(), 1);
        assert_eq!(hooks.post_tool_use.len(), 1);
        assert_eq!(hooks.post_tool_use[0].matcher, None);
        assert_eq!(hooks.post_tool_use[0].hooks.len(), 1);
    }

    fn bash_hook(script: &str) -> CommandHook {
        CommandHook {
            command: "bash".into(),
            args: Some(vec!["-c".into(), script.into()]),
            shell: None,
            timeout: Some(30),
            data_dir: std::env::temp_dir().join("zed-agent-hook-tests"),
        }
    }

    #[test]
    fn test_command_hook_default_shell_follows_platform() {
        let hook = |args: Option<Vec<String>>, shell: Option<&str>| CommandHook {
            command: "check.sh".into(),
            args,
            shell: shell.map(str::to_string),
            timeout: None,
            data_dir: PathBuf::new(),
        };
        let strings = |items: &[&str]| items.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let default = if cfg!(windows) {
            (
                "powershell".to_string(),
                strings(&[
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    "check.sh",
                ]),
            )
        } else {
            ("bash".to_string(), strings(&["-c", "check.sh"]))
        };

        assert_eq!(hook(None, None).command_line(), default);
        assert_eq!(
            hook(None, Some("bash")).command_line(),
            ("bash".to_string(), strings(&["-c", "check.sh"]))
        );
        assert_eq!(
            hook(None, Some("powershell")).command_line(),
            (
                "powershell".to_string(),
                strings(&[
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    "check.sh",
                ])
            )
        );
        assert_eq!(
            hook(Some(strings(&["a b"])), None).command_line(),
            ("check.sh".to_string(), strings(&["a b"]))
        );
    }

    #[gpui::test]
    async fn test_command_hook_runs_with_its_data_dir(cx: &mut TestAppContext) {
        cx.executor().allow_parking();
        let root = std::env::temp_dir().join(format!(
            "zed-agent-hook-data-{}-{}",
            std::process::id(),
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default()
        ));
        let data_dir = root.join("garde");
        let mut hook = bash_hook(
            r#"cat > /dev/null; [ -d "$ZED_HOOK_DATA_DIR" ] || exit 1; printf '{"decision":"block","reason":"%s"}' "${ZED_HOOK_DATA_DIR##*/}""#,
        );
        hook.data_dir = data_dir.clone();
        let input = json!({ "hook_event_name": "PostToolUse", "tool_name": "echo" });

        let output = cx.update(|cx| hook.call(input, cx)).await;
        let created = data_dir.is_dir();
        std::fs::remove_dir_all(&root).ok();

        assert_eq!(output.unwrap()["reason"], "garde");
        assert!(created);
    }

    const CONTRACT_SCHEMAS: [(&str, &str); 12] = [
        (
            "BeforeModel.input.schema.json",
            include_str!("../hook_contract/BeforeModel.input.schema.json"),
        ),
        (
            "BeforeModel.output.schema.json",
            include_str!("../hook_contract/BeforeModel.output.schema.json"),
        ),
        (
            "AfterModel.input.schema.json",
            include_str!("../hook_contract/AfterModel.input.schema.json"),
        ),
        (
            "AfterModel.output.schema.json",
            include_str!("../hook_contract/AfterModel.output.schema.json"),
        ),
        (
            "common-input.schema.json",
            include_str!("../hook_contract/common-input.schema.json"),
        ),
        (
            "common-output.schema.json",
            include_str!("../hook_contract/common-output.schema.json"),
        ),
        (
            "PreToolUse.input.schema.json",
            include_str!("../hook_contract/PreToolUse.input.schema.json"),
        ),
        (
            "PreToolUse.output.schema.json",
            include_str!("../hook_contract/PreToolUse.output.schema.json"),
        ),
        (
            "PostToolUse.input.schema.json",
            include_str!("../hook_contract/PostToolUse.input.schema.json"),
        ),
        (
            "PostToolUse.output.schema.json",
            include_str!("../hook_contract/PostToolUse.output.schema.json"),
        ),
        (
            "tool-inputs.schema.json",
            include_str!("../hook_contract/tool-inputs.schema.json"),
        ),
        (
            "model-message.schema.json",
            include_str!("../hook_contract/model-message.schema.json"),
        ),
    ];

    pub(crate) fn assert_matches_contract(schema_name: &str, value: &serde_json::Value) {
        if let Some(error) = contract_error(schema_name, value) {
            panic!("{schema_name}: {error}\n{value:#}");
        }
    }

    /// Validates `value` against a schema of the hook contract copied in
    /// `crates/agent/hook_contract`, and returns the first error.
    fn contract_error(schema_name: &str, value: &serde_json::Value) -> Option<String> {
        let base = "https://contracts.invalid/zed-agent-hooks/0.1/schemas/";
        let mut registry = jsonschema::Registry::new();
        let mut root = None;
        for (name, contents) in CONTRACT_SCHEMAS {
            let schema: serde_json::Value = serde_json::from_str(contents).unwrap();
            if name == schema_name {
                root = Some(schema.clone());
            }
            registry = registry.add(format!("{base}{name}"), schema).unwrap();
        }
        let registry = registry.prepare().unwrap();
        let validator = jsonschema::options()
            .with_registry(&registry)
            .build(&root.expect("unknown contract schema"))
            .unwrap();
        validator
            .validate(value)
            .err()
            .map(|error| error.to_string())
    }

    #[gpui::test]
    async fn test_exchanged_hook_json_matches_contract(cx: &mut TestAppContext) {
        let pre_output = pre_tool_use(json!({
            "permissionDecision": "allow",
            "updatedInput": { "text": "corrected" },
            "additionalContext": "note",
        }));
        let post_output = json!({
            "decision": "block",
            "reason": "check",
            "hookSpecificOutput": { "hookEventName": "PostToolUse", "additionalContext": "more" },
        });
        let (pre, pre_calls) = fake_hook(Ok(pre_output));
        let (post, post_calls) = fake_hook(Ok(post_output));

        run_echo(hooked_echo(vec![pre], vec![post]).0, "hello", cx).await;

        // The validator does reject what breaks the contract.
        let mut broken_input = pre_calls.borrow()[0].clone();
        broken_input.as_object_mut().unwrap().remove("tool_use_id");
        assert!(contract_error("PreToolUse.input.schema.json", &broken_input).is_some());

        assert_matches_contract("PreToolUse.input.schema.json", &pre_calls.borrow()[0]);
        assert_matches_contract("PostToolUse.input.schema.json", &post_calls.borrow()[0]);
        assert_matches_contract(
            "PreToolUse.output.schema.json",
            &parse_command_output("PreToolUse", Some(2), "", "no\n").unwrap(),
        );
        assert_matches_contract(
            "PostToolUse.output.schema.json",
            &parse_command_output("PostToolUse", Some(2), "", "no\n").unwrap(),
        );
    }

    #[gpui::test]
    async fn test_command_hook_reads_stdin_and_returns_stdout(cx: &mut TestAppContext) {
        cx.executor().allow_parking();
        let hook = bash_hook(
            r#"if grep -q '"tool_name":"echo"'; then printf '%s' '{"decision":"block","reason":"saw echo"}'; else exit 1; fi"#,
        );
        let input = json!({ "hook_event_name": "PostToolUse", "tool_name": "echo" });

        let output = cx.update(|cx| hook.call(input, cx)).await.unwrap();

        assert_eq!(output, json!({ "decision": "block", "reason": "saw echo" }));
    }

    #[gpui::test]
    async fn test_command_hook_exit_code_2_denies_with_stderr(cx: &mut TestAppContext) {
        cx.executor().allow_parking();
        let hook = bash_hook("cat > /dev/null; echo 'Use grep instead.' >&2; exit 2");
        let input = json!({ "hook_event_name": "PreToolUse", "tool_name": "terminal" });

        let output = cx.update(|cx| hook.call(input, cx)).await.unwrap();

        assert_eq!(
            output["hookSpecificOutput"]["permissionDecisionReason"],
            "Use grep instead."
        );
    }

    #[test]
    fn test_matcher_matches_with_claude_code_syntax() {
        for (matcher, value, expected) in [
            (None, "echo", true),
            (Some(""), "echo", true),
            (Some("*"), "echo", true),
            (Some("echo"), "echo", true),
            (Some("echo"), "echo_twice", false),
            (Some("grep|edit_file"), "edit_file", true),
            (Some("grep, edit_file"), "edit_file", true),
            (Some("grep|edit_file"), "read_file", false),
            (Some("mcp__.*"), "mcp__github__search", true),
            (Some("_file$"), "read_file", true),
            (Some("_file$"), "grep", false),
            (Some("(invalid"), "echo", false),
        ] {
            assert_eq!(
                matcher_matches(matcher, value),
                expected,
                "matcher {matcher:?} on {value:?}"
            );
        }
    }

    #[gpui::test]
    async fn test_pre_tool_use_deny_returns_reason_without_running_tool(cx: &mut TestAppContext) {
        let (deny, calls) = fake_hook(Ok(pre_tool_use(json!({
            "permissionDecision": "deny",
            "permissionDecisionReason": "Use grep instead.",
        }))));
        let (tool, echo) = hooked_echo(vec![deny], Vec::new());

        let (is_error, texts) = run_echo(tool, "hello", cx).await;

        assert!(is_error);
        assert_eq!(texts, vec!["Use grep instead.".to_string()]);
        assert_eq!(echo.runs.get(), 0);
        let calls = calls.borrow();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["hook_event_name"], "PreToolUse");
        assert_eq!(calls[0]["tool_name"], "echo");
        assert_eq!(calls[0]["tool_input"], json!({ "text": "hello" }));
        assert_eq!(calls[0]["session_id"], "session-1");
        assert_eq!(calls[0]["transcript_path"], serde_json::Value::Null);
        assert_eq!(calls[0]["cwd"], "/project");
        assert_eq!(calls[0]["zed"]["contract_version"], HOOK_CONTRACT_VERSION);
        assert_eq!(calls[0]["zed"]["thread_id"], "session-1");
        assert_eq!(calls[0]["zed"]["is_subagent"], false);
        assert_eq!(calls[0]["zed"]["depth"], 0);
    }

    #[gpui::test]
    async fn test_pre_tool_use_updated_input_reaches_tool(cx: &mut TestAppContext) {
        let (update, _) = fake_hook(Ok(pre_tool_use(json!({
            "updatedInput": { "text": "corrected" },
        }))));
        let (tool, echo) = hooked_echo(vec![update], Vec::new());

        let (is_error, texts) = run_echo(tool, "hello", cx).await;

        assert!(!is_error);
        assert_eq!(texts, vec!["corrected".to_string()]);
        assert_eq!(echo.runs.get(), 1);
    }

    #[gpui::test]
    async fn test_post_tool_use_block_adds_note_and_keeps_output(cx: &mut TestAppContext) {
        let (block, calls) = fake_hook(Ok(json!({
            "decision": "block",
            "reason": "Run the tests before editing again.",
        })));
        let (tool, echo) = hooked_echo(Vec::new(), vec![block]);

        let (is_error, texts) = run_echo(tool, "hello", cx).await;

        assert!(!is_error);
        assert_eq!(
            texts,
            vec![
                "hello".to_string(),
                "Run the tests before editing again.".to_string(),
            ]
        );
        assert_eq!(echo.runs.get(), 1);
        let calls = calls.borrow();
        assert_eq!(calls[0]["hook_event_name"], "PostToolUse");
        assert_eq!(calls[0]["tool_input"], json!({ "text": "hello" }));
        assert_eq!(
            calls[0]["tool_response"],
            json!({ "is_error": false, "content": [{ "type": "text", "text": "hello" }] })
        );
    }

    #[gpui::test]
    async fn test_additional_context_accumulates_in_order(cx: &mut TestAppContext) {
        let (first, _) = fake_hook(Ok(pre_tool_use(json!({ "additionalContext": "first" }))));
        let (second, _) = fake_hook(Ok(pre_tool_use(json!({ "additionalContext": "second" }))));
        let (third, _) = fake_hook(Ok(json!({
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "additionalContext": "third",
            }
        })));
        let (tool, echo) = hooked_echo(vec![first, second], vec![third]);

        let (_, texts) = run_echo(tool, "hello", cx).await;

        assert_eq!(
            texts,
            vec![
                "hello".to_string(),
                "first".to_string(),
                "second".to_string(),
                "third".to_string(),
            ]
        );
        assert_eq!(echo.runs.get(), 1);
    }

    #[gpui::test]
    async fn test_pre_tool_use_hooks_chain_in_declaration_order(cx: &mut TestAppContext) {
        let (update, update_calls) = fake_hook(Ok(pre_tool_use(json!({
            "updatedInput": { "text": "from first" },
            "additionalContext": "first ran",
        }))));
        let (deny, deny_calls) = fake_hook(Ok(pre_tool_use(json!({
            "permissionDecision": "deny",
            "permissionDecisionReason": "second denied",
        }))));
        let (never, never_calls) = fake_hook(Ok(pre_tool_use(json!({
            "permissionDecision": "allow",
        }))));
        let (tool, echo) = hooked_echo(vec![update, deny, never], Vec::new());

        let (is_error, texts) = run_echo(tool, "hello", cx).await;

        assert!(is_error);
        assert_eq!(
            texts,
            vec!["second denied".to_string(), "first ran".to_string()]
        );
        assert_eq!(
            update_calls.borrow()[0]["tool_input"],
            json!({ "text": "hello" })
        );
        assert_eq!(
            deny_calls.borrow()[0]["tool_input"],
            json!({ "text": "from first" })
        );
        assert!(never_calls.borrow().is_empty());
        assert_eq!(echo.runs.get(), 0);
    }

    #[gpui::test]
    async fn test_failing_hook_lets_tool_run(cx: &mut TestAppContext) {
        let (failing, _) = fake_hook(Err("hook crashed".into()));
        let (update, _) = fake_hook(Ok(pre_tool_use(json!({
            "updatedInput": { "text": "after failure" },
        }))));
        let (tool, echo) = hooked_echo(vec![failing, update], Vec::new());

        let (is_error, texts) = run_echo(tool, "hello", cx).await;

        assert!(!is_error);
        assert_eq!(texts, vec!["after failure".to_string()]);
        assert_eq!(echo.runs.get(), 1);
    }

    #[gpui::test]
    async fn test_pre_tool_use_ask_is_refused_until_supported(cx: &mut TestAppContext) {
        let (ask, _) = fake_hook(Ok(pre_tool_use(json!({
            "permissionDecision": "ask",
            "permissionDecisionReason": "Pushing to main.",
        }))));
        let (tool, echo) = hooked_echo(vec![ask], Vec::new());

        let (is_error, texts) = run_echo(tool, "hello", cx).await;

        assert!(is_error);
        assert_eq!(texts.len(), 1);
        assert!(texts[0].contains("confirm"), "{texts:?}");
        assert!(texts[0].contains("Pushing to main."), "{texts:?}");
        assert_eq!(echo.runs.get(), 0);
    }

    #[gpui::test]
    async fn test_invalid_updated_input_is_a_hook_failure(cx: &mut TestAppContext) {
        let (invalid, _) = fake_hook(Ok(pre_tool_use(json!({
            "updatedInput": { "text": 42 },
            "additionalContext": "ignored with its hook",
        }))));
        let (next, next_calls) = fake_hook(Ok(pre_tool_use(json!({
            "additionalContext": "kept",
        }))));
        let (tool, echo) = hooked_echo(vec![invalid, next], Vec::new());

        let (is_error, texts) = run_echo(tool, "hello", cx).await;

        assert!(!is_error);
        assert_eq!(texts, vec!["hello".to_string(), "kept".to_string()]);
        assert_eq!(
            next_calls.borrow()[0]["tool_input"],
            json!({ "text": "hello" })
        );
        assert_eq!(echo.runs.get(), 1);
    }

    #[gpui::test]
    async fn test_deny_wins_over_null_or_invalid_updated_input(cx: &mut TestAppContext) {
        for updated_input in [serde_json::Value::Null, json!({ "text": 42 })] {
            let (deny, _) = fake_hook(Ok(pre_tool_use(json!({
                "permissionDecision": "deny",
                "permissionDecisionReason": "No.",
                "updatedInput": updated_input,
            }))));
            let (tool, echo) = hooked_echo(vec![deny], Vec::new());

            let (is_error, texts) = run_echo(tool, "hello", cx).await;

            assert!(is_error, "updatedInput {updated_input}");
            assert_eq!(
                texts,
                vec!["No.".to_string()],
                "updatedInput {updated_input}"
            );
            assert_eq!(echo.runs.get(), 0, "updatedInput {updated_input}");
        }
    }

    #[gpui::test]
    async fn test_null_updated_input_keeps_the_input(cx: &mut TestAppContext) {
        let (allow, _) = fake_hook(Ok(pre_tool_use(json!({
            "permissionDecision": "allow",
            "updatedInput": null,
            "additionalContext": "noted",
        }))));
        let (tool, echo) = hooked_echo(vec![allow], Vec::new());

        let (is_error, texts) = run_echo(tool, "hello", cx).await;

        assert!(!is_error);
        assert_eq!(texts, vec!["hello".to_string(), "noted".to_string()]);
        assert_eq!(echo.runs.get(), 1);
    }

    #[gpui::test]
    async fn test_legacy_block_decision_denies_before_the_tool(cx: &mut TestAppContext) {
        // Alone, or next to a more permissive `permissionDecision`: the most
        // restrictive decision wins.
        for output in [
            json!({ "decision": "block", "reason": "Use grep instead." }),
            json!({
                "decision": "block",
                "reason": "Use grep instead.",
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "allow",
                },
            }),
        ] {
            let (block, _) = fake_hook(Ok(output.clone()));
            let (tool, echo) = hooked_echo(vec![block], Vec::new());

            let (is_error, texts) = run_echo(tool, "hello", cx).await;

            assert!(is_error, "{output}");
            assert_eq!(texts, vec!["Use grep instead.".to_string()], "{output}");
            assert_eq!(echo.runs.get(), 0, "{output}");
        }
    }

    #[gpui::test]
    async fn test_empty_deny_reason_uses_the_default_message(cx: &mut TestAppContext) {
        let (deny, _) = fake_hook(Ok(
            parse_command_output("PreToolUse", Some(2), "", "").unwrap()
        ));
        let (tool, echo) = hooked_echo(vec![deny], Vec::new());

        let (is_error, texts) = run_echo(tool, "hello", cx).await;

        assert!(is_error);
        assert_eq!(texts, vec!["A hook denied this tool call.".to_string()]);
        assert_eq!(echo.runs.get(), 0);
    }

    #[test]
    fn test_handlers_with_another_id_or_timeout_are_not_duplicates() {
        let hooks = AgentHooks::from_settings(
            &json!({
                "PreToolUse": [{
                    "hooks": [
                        { "type": "command", "command": "check", "id": "first" },
                        { "type": "command", "command": "check", "id": "second" },
                        { "type": "command", "command": "check", "timeout": 5 },
                        { "type": "command", "command": "check", "timeout": 10 },
                    ],
                }],
            }),
            Path::new("/data/hooks"),
        )
        .unwrap();

        assert_eq!(matching_hooks(&hooks.pre_tool_use, "echo").len(), 4);
    }

    #[gpui::test]
    async fn test_cancellation_abandons_hook_and_skips_tool(cx: &mut TestAppContext) {
        let abandoned = Arc::new(AtomicBool::new(false));
        let pending: Arc<dyn HookHandler> = Arc::new(PendingHook {
            abandoned: abandoned.clone(),
        });
        let (tool, echo) = hooked_echo(vec![pending], Vec::new());
        let (event_stream, _receiver, mut cancellation_tx) =
            ToolCallEventStream::test_with_cancellation();
        let mut task = cx.update(|cx| {
            tool.run(
                ToolInput::ready(json!({ "text": "hello" })),
                event_stream,
                cx,
            )
        });
        cx.run_until_parked();

        cancellation_tx.send(true).ok();
        cx.run_until_parked();

        let result = futures::FutureExt::now_or_never(&mut task)
            .expect("cancelling must stop the tool call while a hook runs");
        assert!(result.is_err());
        assert!(abandoned.load(Ordering::SeqCst));
        assert_eq!(echo.runs.get(), 0);
    }

    #[gpui::test]
    async fn test_post_tool_use_hooks_keep_input_streaming(cx: &mut TestAppContext) {
        let (post, post_calls) = fake_hook(Ok(json!({})));
        let (tool, echo) = hooked_echo(Vec::new(), vec![post]);
        let (mut sender, input) = ToolInputSender::channel();
        sender.send_partial(json!({ "text": "he" }));
        sender.send_full(json!({ "text": "hello" }));
        let (event_stream, _receiver) = ToolCallEventStream::test();

        let result = cx.update(|cx| tool.run(input, event_stream, cx)).await;

        assert!(result.is_ok());
        assert_eq!(echo.partial_inputs.get(), 1);
        assert_eq!(
            post_calls.borrow()[0]["tool_input"],
            json!({ "text": "hello" })
        );
    }

    #[test]
    fn test_identical_handlers_run_once_per_tool() {
        let check = json!({ "type": "command", "command": "check" });
        let hooks = AgentHooks::from_settings(
            &json!({
                "PreToolUse": [
                    { "matcher": "echo", "hooks": [check.clone()] },
                    { "matcher": "*", "hooks": [check, { "type": "command", "command": "other" }] },
                ],
            }),
            Path::new("/data/hooks"),
        )
        .unwrap();

        assert_eq!(matching_hooks(&hooks.pre_tool_use, "echo").len(), 2);
    }

    #[test]
    fn test_unsupported_hook_settings_are_refused_with_their_location() {
        for (settings, location) in [
            (
                json!({ "PreToolUse": [{ "hooks": [{ "type": "command", "command": "x", "onFailure": "block" }] }] }),
                "PreToolUse[0].hooks[0]",
            ),
            (
                json!({ "PreToolUse": [{ "hooks": [{ "type": "command" }] }] }),
                "PreToolUse[0].hooks[0]",
            ),
            (
                json!({ "PostToolUse": [{ "hooks": [{ "type": "command", "command": "x", "args": [1] }] }] }),
                "PostToolUse[0].hooks[0]",
            ),
            (
                json!({ "PreToolUse": [{ "matcher": "(", "hooks": [{ "type": "command", "command": "x" }] }] }),
                "PreToolUse[0].matcher",
            ),
        ]
        .into_iter()
        .chain(
            ["http", "mcp_tool", "prompt", "agent"].map(|kind| {
                (
                    json!({ "PreToolUse": [{ "hooks": [{ "type": kind, "url": "http://localhost:9000" }] }] }),
                    "PreToolUse[0].hooks[0]",
                )
            }),
        )
        .chain(
            [
                json!({ "if": "Bash(git *)" }),
                json!({ "async": true }),
                json!({ "once": true }),
                json!({ "shell": "zsh" }),
                json!({ "timeout": "10" }),
                json!({ "timeout": 1.5 }),
            ]
            .map(|extra| {
                let mut handler = json!({ "type": "command", "command": "x" });
                handler
                    .as_object_mut()
                    .unwrap()
                    .extend(extra.as_object().unwrap().clone());
                (
                    json!({ "PostToolUse": [{ "hooks": [handler] }] }),
                    "PostToolUse[0].hooks[0]",
                )
            }),
        ) {
            let error = AgentHooks::from_settings(&settings, Path::new("/data/hooks"))
                .err()
                .unwrap_or_else(|| panic!("settings must be refused: {settings}"))
                .to_string();
            assert!(error.contains(location), "{error}");
        }
    }

    /// An input bigger than any pipe buffer, so that a hook that does not
    /// read it blocks the writer.
    fn big_input(hook_event_name: &str) -> serde_json::Value {
        json!({ "hook_event_name": hook_event_name, "padding": "x".repeat(1 << 20) })
    }

    #[gpui::test]
    async fn test_command_hook_timeout_covers_unread_input(cx: &mut TestAppContext) {
        cx.executor().allow_parking();
        let mut hook = bash_hook("sleep 5");
        hook.timeout = Some(1);
        let started_at = Instant::now();

        let task = cx.update(|cx| hook.call(big_input("PreToolUse"), cx));
        cx.executor().advance_clock(Duration::from_secs(2));
        let result = task.await;

        let error = result.expect_err("the hook must time out").to_string();
        assert!(error.contains("timed out"), "{error}");
        assert!(started_at.elapsed() < Duration::from_secs(4));
    }

    #[gpui::test]
    async fn test_command_hook_exit_2_without_reading_input_denies(cx: &mut TestAppContext) {
        cx.executor().allow_parking();
        let hook = bash_hook("echo 'Use grep instead.' >&2; exit 2");

        let output = cx
            .update(|cx| hook.call(big_input("PreToolUse"), cx))
            .await
            .unwrap();

        assert_eq!(
            output["hookSpecificOutput"]["permissionDecisionReason"],
            "Use grep instead."
        );
    }

    #[gpui::test]
    async fn test_command_hook_runs_in_project_dir(cx: &mut TestAppContext) {
        cx.executor().allow_parking();
        let project_dir =
            std::env::temp_dir().join(format!("zed-agent-hook-project-{}", std::process::id()));
        std::fs::create_dir_all(&project_dir).unwrap();
        let hook = bash_hook(
            r#"cat > /dev/null; printf '{"decision":"block","reason":"%s"}' "${PWD##*/}""#,
        );
        let input = json!({
            "hook_event_name": "PostToolUse",
            "cwd": project_dir.to_string_lossy(),
        });

        let output = cx.update(|cx| hook.call(input, cx)).await;
        std::fs::remove_dir(&project_dir).ok();

        assert_eq!(
            output.unwrap()["reason"],
            project_dir.file_name().unwrap().to_string_lossy().as_ref()
        );
    }

    fn message(role: Role, content: Vec<MessageContent>) -> LanguageModelRequestMessage {
        LanguageModelRequestMessage {
            role,
            content,
            cache: false,
            reasoning_details: None,
        }
    }

    fn text_of(message: &LanguageModelRequestMessage) -> String {
        message
            .content
            .iter()
            .filter_map(|part| match part {
                MessageContent::Text(text) => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn test_insert_into_system_message_before_heading_or_at_end() {
        let user = message(Role::User, vec!["See ## System Information below.".into()]);
        let mut request = LanguageModelRequest {
            messages: vec![
                message(
                    Role::System,
                    vec!["You are Zed.\n\n## System Information\n\nOS: test\n".into()],
                ),
                user.clone(),
            ],
            ..Default::default()
        };

        assert_eq!(
            insert_into_system_message(&mut request, "## Agent Profile\nExplorer."),
            InsertOutcome::BeforeHeading
        );
        assert_eq!(
            text_of(&request.messages[0]),
            "You are Zed.\n\n## Agent Profile\nExplorer.\n\n## System Information\n\nOS: test\n"
        );
        assert_eq!(request.messages[1], user);

        let mut request = LanguageModelRequest {
            messages: vec![
                message(Role::System, vec!["You are Zed.".into()]),
                user.clone(),
            ],
            ..Default::default()
        };
        assert_eq!(
            insert_into_system_message(&mut request, "Extra."),
            InsertOutcome::AtEnd
        );
        assert_eq!(text_of(&request.messages[0]), "You are Zed.\n\nExtra.");
        assert_eq!(request.messages[1], user);

        let mut request = LanguageModelRequest {
            messages: vec![user.clone()],
            ..Default::default()
        };
        assert_eq!(
            insert_into_system_message(&mut request, "Extra."),
            InsertOutcome::NoSystemMessage
        );
        assert_eq!(request.messages, vec![user]);
    }

    #[test]
    fn test_system_prompt_insert_skips_compaction() {
        let conversation = LanguageModelRequest {
            intent: Some(CompletionIntent::UserPrompt),
            messages: vec![
                message(Role::System, vec!["You are Zed.".into()]),
                message(Role::User, vec!["Hello".into()]),
            ],
            ..Default::default()
        };
        assert!(applies_system_prompt_insert(&conversation));

        let mut compaction = conversation.clone();
        compaction.intent = Some(CompletionIntent::ThreadContextSummarization);
        compaction
            .messages
            .push(message(Role::User, vec![COMPACTION_PROMPT.into()]));
        assert!(!applies_system_prompt_insert(&compaction));
    }

    #[test]
    fn test_model_hook_matcher_naming_titles_is_refused() {
        for matcher in ["thread_summarization", "user_prompt|thread_summarization"] {
            for event in ["BeforeModel", "AfterModel"] {
                let error = AgentHooks::from_settings(
                    &json!({
                        event: [{ "matcher": matcher, "hooks": [{ "type": "command", "command": "x" }] }],
                    }),
                    Path::new("/data/hooks"),
                )
                .err()
                .unwrap_or_else(|| panic!("{event} matcher {matcher:?} must be refused"))
                .to_string();
                assert!(error.contains(&format!("{event}[0].matcher")), "{error}");
            }
        }
        for universal in ["*", "", ".*", ".+"] {
            assert!(
                AgentHooks::from_settings(
                    &json!({ "BeforeModel": [{ "matcher": universal, "hooks": [{ "type": "command", "command": "x" }] }] }),
                    Path::new("/data/hooks"),
                )
                .is_ok(),
                "{universal:?} must be accepted"
            );
        }
        assert!(
            AgentHooks::from_settings(
                &json!({ "AfterModel": [{ "matcher": "thread_.*", "hooks": [{ "type": "command", "command": "x" }] }] }),
                Path::new("/data/hooks"),
            )
            .is_err()
        );
    }

    #[gpui::test]
    async fn test_output_for_another_event_is_a_hook_failure(cx: &mut TestAppContext) {
        let (deny, _) = fake_hook(Ok(json!({
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": "Wrong event.",
            }
        })));
        let (note, _) = fake_hook(Ok(json!({
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "additionalContext": "Wrong event too.",
            }
        })));
        let (tool, echo) = hooked_echo(vec![deny], vec![note]);

        let (is_error, texts) = run_echo(tool, "hello", cx).await;

        assert!(!is_error);
        assert_eq!(texts, vec!["hello".to_string()]);
        assert_eq!(echo.runs.get(), 1);
    }

    #[test]
    fn test_model_messages_match_contract() {
        let messages = [
            message(Role::System, vec!["You are Zed.".into()]),
            message(
                Role::Assistant,
                vec![
                    MessageContent::Thinking {
                        text: "Let me look.".into(),
                        signature: Some("sig".into()),
                    },
                    MessageContent::RedactedThinking("opaque".into()),
                    MessageContent::ToolUse(LanguageModelToolUse {
                        id: "tool_1".into(),
                        name: "grep".into(),
                        raw_input: r#"{"regex":"fn"}"#.into(),
                        input: LanguageModelToolUseInput::Json(json!({ "regex": "fn" })),
                        is_input_complete: true,
                        thought_signature: None,
                    }),
                ],
            ),
            message(
                Role::User,
                vec![
                    MessageContent::ToolResult(LanguageModelToolResult {
                        tool_use_id: "tool_1".into(),
                        tool_name: "grep".into(),
                        is_error: false,
                        content: vec!["a".into(), "b".into()],
                        output: None,
                    }),
                    MessageContent::Image(LanguageModelImage {
                        source: "aGVsbG8=".into(),
                    }),
                ],
            ),
        ];

        let converted = messages.iter().map(message_to_contract).collect::<Vec<_>>();

        for message in &converted {
            assert_matches_contract("model-message.schema.json", message);
        }
        assert_eq!(
            converted[1]["content"],
            json!([
                { "type": "thinking", "text": "Let me look." },
                { "type": "redacted_thinking" },
                { "type": "tool_use", "id": "tool_1", "name": "grep", "input": { "regex": "fn" } },
            ])
        );
        assert_eq!(
            converted[2]["content"],
            json!([
                { "type": "tool_result", "tool_use_id": "tool_1", "tool_name": "grep", "is_error": false, "text": "ab" },
                { "type": "image", "byte_length": 8 },
            ])
        );
    }

    #[test]
    fn test_system_prompt_insert_is_validated() {
        let output = |insert: serde_json::Value| {
            json!({
                "hookSpecificOutput": {
                    "hookEventName": "BeforeModel",
                    "system_prompt_insert": insert,
                }
            })
        };

        assert_eq!(system_prompt_insert(&json!({})).unwrap(), None);
        let profile = output(json!("Profile."));
        assert_eq!(system_prompt_insert(&profile).unwrap(), Some("Profile."));
        let at_limit = output(json!("é".repeat(10_000)));
        assert_eq!(
            system_prompt_insert(&at_limit).unwrap().map(str::len),
            Some(20_000)
        );
        assert!(system_prompt_insert(&output(json!(""))).is_err());
        assert!(system_prompt_insert(&output(json!("x".repeat(10_001)))).is_err());
        assert!(system_prompt_insert(&output(json!(42))).is_err());
    }

    #[test]
    fn test_model_hooks_are_read_and_other_events_refused() {
        let check = json!({ "type": "command", "command": "check" });
        let hooks = AgentHooks::from_settings(
            &json!({
                "BeforeModel": [{ "matcher": "user_prompt|tool_results", "hooks": [check.clone()] }],
                "AfterModel": [{ "hooks": [check.clone()] }],
            }),
            Path::new("/data/hooks"),
        )
        .unwrap();
        assert_eq!(matching_hooks(&hooks.before_model, "tool_results").len(), 1);
        assert_eq!(matching_hooks(&hooks.before_model, "edit_file").len(), 0);
        assert_eq!(matching_hooks(&hooks.after_model, "user_prompt").len(), 1);

        let error = AgentHooks::from_settings(
            &json!({ "Stop": [{ "hooks": [check] }] }),
            Path::new("/data/hooks"),
        )
        .err()
        .expect("events Zed does not run yet must be refused")
        .to_string();
        assert!(error.contains("Stop"), "{error}");
    }
}
