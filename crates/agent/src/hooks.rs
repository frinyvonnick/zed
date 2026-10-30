use crate::{
    AgentMessageContent, AgentModelId, AgentToolOutput, AnyAgentTool, Message, SubagentHandle,
    TerminalHandle, Thread, ThreadEnvironment, ThreadEventStream, ToolCallEventStream, ToolInput,
    ToolInputPayload, ToolInputSender, UserMessage, UserMessageContent,
};
use acp_thread::ClientUserMessageId;
use agent_client_protocol::schema::v2 as acp_v2;
use agent_settings::{AgentSettings, COMPACTION_PROMPT};
use anyhow::{Context as _, Result, anyhow};
use collections::{HashMap, HashSet};
use futures::{
    AsyncWriteExt as _, FutureExt as _, StreamExt as _,
    channel::{mpsc, oneshot},
};
use gpui::{App, AsyncApp, Entity, Global, SharedString, Task, WeakEntity};
use language_model::{
    CompactedContext, CompletionIntent, LanguageModel, LanguageModelCompletionError,
    LanguageModelCompletionEvent, LanguageModelCompletionStream, LanguageModelProvider,
    LanguageModelProviderId, LanguageModelRequest, LanguageModelRequestMessage,
    LanguageModelToolResultContent, LanguageModelToolUseInput, MessageContent,
    ProviderErrorCategory, Role, StopReason, TokenUsage,
};
use regex::Regex;
use serde_json::json;
use settings::Settings as _;
use sha2::{Digest as _, Sha256};
use std::{
    cell::RefCell,
    collections::BTreeMap,
    hash::{DefaultHasher, Hash as _, Hasher as _},
    path::{Path, PathBuf},
    process::Stdio,
    rc::Rc,
    sync::{Arc, LazyLock},
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
    pub pre_compact: Vec<HookMatcherGroup>,
    pub post_compact: Vec<HookMatcherGroup>,
    pub zed_model_error: Vec<HookMatcherGroup>,
    pub subagent_start: Vec<HookMatcherGroup>,
    pub subagent_stop: Vec<HookMatcherGroup>,
    pub stop: Vec<HookMatcherGroup>,
    pub user_prompt_submit: Vec<HookMatcherGroup>,
    pub session_start: Vec<HookMatcherGroup>,
    pub session_end: Vec<HookMatcherGroup>,
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
            pre_compact: matcher_groups_from_settings(&hooks["PreCompact"], data_root),
            post_compact: matcher_groups_from_settings(&hooks["PostCompact"], data_root),
            zed_model_error: matcher_groups_from_settings(&hooks["ZedModelError"], data_root),
            subagent_start: matcher_groups_from_settings(&hooks["SubagentStart"], data_root),
            subagent_stop: matcher_groups_from_settings(&hooks["SubagentStop"], data_root),
            stop: matcher_groups_from_settings(&hooks["Stop"], data_root),
            user_prompt_submit: matcher_groups_from_settings(&hooks["UserPromptSubmit"], data_root),
            session_start: matcher_groups_from_settings(&hooks["SessionStart"], data_root),
            session_end: matcher_groups_from_settings(&hooks["SessionEnd"], data_root),
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
                check_handler(handler).with_context(|| format!("hook {location}"))?;
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
const SUPPORTED_EVENTS: [&str; 13] = [
    "PreToolUse",
    "PostToolUse",
    "BeforeModel",
    "AfterModel",
    "PreCompact",
    "PostCompact",
    "ZedModelError",
    "SubagentStart",
    "SubagentStop",
    "Stop",
    "UserPromptSubmit",
    "SessionStart",
    "SessionEnd",
];

fn check_handler(handler: &serde_json::Value) -> Result<()> {
    match handler["type"].as_str() {
        Some("command") => check_command_handler(handler),
        Some("http") => check_http_handler(handler),
        _ => anyhow::bail!(
            "hooks of type {} are not supported yet; only `command` and `http` hooks run",
            handler["type"]
        ),
    }?;
    if let Some(timeout) = handler.get("timeout") {
        anyhow::ensure!(
            timeout.as_u64().is_some_and(|timeout| timeout >= 1),
            "`timeout` must be a whole number of seconds, at least 1"
        );
    }
    if let Some(on_failure) = handler.get("onFailure") {
        anyhow::ensure!(
            matches!(on_failure.as_str(), Some("continue" | "block")),
            "`onFailure` must be \"continue\" or \"block\""
        );
    }
    if let Some(status_message) = handler.get("statusMessage") {
        anyhow::ensure!(
            status_message.is_string(),
            "`statusMessage` must be a string"
        );
    }
    Ok(())
}

/// The fields of an `http` hook. It has no `id` and no data directory: its
/// server keeps its own state.
const HTTP_HOOK_FIELDS: [&str; 7] = [
    "type",
    "url",
    "headers",
    "allowedEnvVars",
    "timeout",
    "onFailure",
    "statusMessage",
];

fn check_http_handler(handler: &serde_json::Value) -> Result<()> {
    if let Some(field) = handler
        .as_object()
        .into_iter()
        .flatten()
        .map(|(field, _)| field)
        .find(|field| !HTTP_HOOK_FIELDS.contains(&field.as_str()))
    {
        anyhow::bail!("`{field}` is not a field of an http hook");
    }
    let url = handler["url"]
        .as_str()
        .context("an http hook needs a `url`")?;
    let parsed = url::Url::parse(url).with_context(|| format!("invalid `url` {url:?}"))?;
    anyhow::ensure!(
        matches!(parsed.scheme(), "http" | "https"),
        "the `url` of an http hook must be http or https"
    );
    if let Some(headers) = handler.get("headers") {
        anyhow::ensure!(
            headers
                .as_object()
                .is_some_and(|headers| headers.values().all(serde_json::Value::is_string)),
            "`headers` must map names to strings"
        );
        for name in headers
            .as_object()
            .into_iter()
            .flatten()
            .map(|(name, _)| name)
        {
            http_client::http::HeaderName::from_bytes(name.as_bytes())
                .with_context(|| format!("invalid header name {name:?}"))?;
        }
    }
    if let Some(variables) = handler.get("allowedEnvVars") {
        anyhow::ensure!(
            variables
                .as_array()
                .is_some_and(|variables| variables.iter().all(serde_json::Value::is_string)),
            "`allowedEnvVars` must be a list of names"
        );
    }
    Ok(())
}

fn check_command_handler(handler: &serde_json::Value) -> Result<()> {
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
                    if handler["type"] == "http" {
                        return Some(
                            Arc::new(HttpHook::from_settings(handler)?) as Arc<dyn HookHandler>
                        );
                    }
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
                        block_on_failure: handler["onFailure"] == "block",
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
    /// `onFailure: "block"`: a failure acts like exit code 2, so that a slow
    /// or broken guard cannot let an action through.
    pub block_on_failure: bool,
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
                &self.data_dir,
                self.block_on_failure
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
        let timeout = self
            .timeout
            .map(Duration::from_secs)
            .unwrap_or_else(|| default_hook_timeout(&hook_event_name));
        let timer = cx.background_executor().timer(timeout);
        let data_dir = self.data_dir.clone();
        // As in Claude Code, `onFailure` has no effect on Stop and SubagentStop.
        let block_on_failure =
            self.block_on_failure && !matches!(hook_event_name.as_str(), "Stop" | "SubagentStop");
        cx.background_executor().spawn(async move {
            let result: Result<serde_json::Value> = async {
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
                let output = parse_command_output(
                    &hook_event_name,
                    output.status.code(),
                    &String::from_utf8_lossy(&output.stdout),
                    &String::from_utf8_lossy(&output.stderr),
                )?;
                // An output for another event is invalid: a failure.
                check_event_name(&hook_event_name, &output)?;
                Ok(output)
            }
            .await;
            settle_hook_result(result, &hook_event_name, block_on_failure)
        })
    }
}

/// Ends a hook call: with `onFailure: "block"`, a failure acts like exit
/// code 2 on its event.
fn settle_hook_result(
    result: Result<serde_json::Value>,
    hook_event_name: &str,
    block_on_failure: bool,
) -> Result<serde_json::Value> {
    match result {
        Err(error) if block_on_failure => {
            log::warn!("agent hook {hook_event_name} failed, which blocks: {error:#}");
            Ok(blocking_output(
                hook_event_name,
                &format!("A hook failed: {error:#}"),
            ))
        }
        result => result,
    }
}

/// POSTs the hook input as JSON to a URL, and reads the output in a 2xx
/// body, as Claude Code does. A 2xx empty body is a success without fields;
/// any other answer is a failure. A server that stays up keeps its own
/// state: no data directory.
pub struct HttpHook {
    pub url: String,
    /// Extra headers, their allowed `$VAR` already replaced.
    pub headers: Vec<(String, String)>,
    pub timeout: Option<u64>,
    pub block_on_failure: bool,
}

impl HttpHook {
    fn from_settings(handler: &serde_json::Value) -> Option<Self> {
        let allowed = handler["allowedEnvVars"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|name| name.as_str().map(str::to_string))
            .collect::<Vec<_>>();
        let headers = handler["headers"]
            .as_object()
            .into_iter()
            .flatten()
            .filter_map(|(name, value)| {
                let value =
                    interpolate_header(value.as_str()?, &allowed, |name| std::env::var(name).ok());
                Some((name.clone(), value))
            })
            .collect();
        Some(Self {
            url: handler["url"].as_str()?.to_string(),
            headers,
            timeout: handler["timeout"].as_u64(),
            block_on_failure: handler["onFailure"] == "block",
        })
    }
}

impl HookHandler for HttpHook {
    fn dedup_key(&self) -> Option<String> {
        Some(format!(
            "{:?}",
            (
                &self.url,
                &self.headers,
                self.timeout,
                self.block_on_failure
            )
        ))
    }

    fn call(&self, input: serde_json::Value, cx: &mut App) -> Task<Result<serde_json::Value>> {
        let hook_event_name = input["hook_event_name"]
            .as_str()
            .unwrap_or_default()
            .to_string();
        let timeout = self
            .timeout
            .map(Duration::from_secs)
            .unwrap_or_else(|| default_hook_timeout(&hook_event_name));
        let timer = cx.background_executor().timer(timeout);
        // As in Claude Code, `onFailure` has no effect on Stop and SubagentStop.
        let block_on_failure =
            self.block_on_failure && !matches!(hook_event_name.as_str(), "Stop" | "SubagentStop");
        let client = cx.http_client();
        // A redirect would send the conversation, and the headers, to
        // another host: a 3xx is a failure.
        let mut request = http_client::HttpRequestExt::follow_redirects(
            http_client::Request::builder(),
            http_client::RedirectPolicy::NoFollow,
        )
        .method(http_client::Method::POST)
        .uri(self.url.as_str())
        .header("Content-Type", "application/json");
        for (name, value) in &self.headers {
            // A header of the user replaces Zed's, such as `Content-Type`.
            if let Some(headers) = request.headers_mut()
                && let (Ok(name), Ok(value)) = (
                    http_client::http::HeaderName::from_bytes(name.as_bytes()),
                    http_client::http::HeaderValue::from_str(value),
                )
            {
                headers.insert(name, value);
            } else {
                log::warn!("an http hook header cannot be sent; it is left out");
            }
        }
        cx.background_executor().spawn(async move {
            let result: Result<serde_json::Value> = async {
                let request =
                    request.body(http_client::AsyncBody::from(serde_json::to_vec(&input)?))?;
                // The timeout covers the whole exchange, body included: a
                // server that never ends its body cannot hold the agent.
                let exchange = async {
                    let mut response = client.send(request).await?;
                    anyhow::ensure!(
                        response.status().is_success(),
                        "hook answered HTTP {}",
                        response.status()
                    );
                    let mut body = Vec::new();
                    futures::AsyncReadExt::read_to_end(
                        &mut futures::AsyncReadExt::take(
                            response.body_mut(),
                            MAX_HTTP_HOOK_BODY_BYTES as u64 + 1,
                        ),
                        &mut body,
                    )
                    .await?;
                    anyhow::ensure!(
                        body.len() <= MAX_HTTP_HOOK_BODY_BYTES,
                        "hook answered more than {MAX_HTTP_HOOK_BODY_BYTES} bytes"
                    );
                    Ok(String::from_utf8(body)?)
                };
                let body = futures::select_biased! {
                    body = exchange.fuse() => body?,
                    _ = timer.fuse() => anyhow::bail!("hook timed out after {timeout:?}"),
                };
                let body = body.trim();
                let output = if body.is_empty() {
                    json!({})
                } else {
                    let output = serde_json::from_str::<serde_json::Value>(body)
                        .context("hook answered invalid JSON")?;
                    anyhow::ensure!(
                        output.is_object(),
                        "hook answered JSON that is not an object"
                    );
                    output
                };
                // An output for another event is invalid: a failure.
                check_event_name(&hook_event_name, &output)?;
                Ok(output)
            }
            .await;
            settle_hook_result(result, &hook_event_name, block_on_failure)
        })
    }
}

/// Replaces `$VAR` and `${VAR}` in a header value, as Claude Code does:
/// a variable listed in `allowedEnvVars` takes its value, any other one
/// becomes empty, so that a header cannot leak what the user did not allow.
fn interpolate_header(
    value: &str,
    allowed: &[String],
    lookup: impl Fn(&str) -> Option<String>,
) -> String {
    static VARIABLE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")
            .expect("valid variable pattern")
    });
    VARIABLE
        .replace_all(value, |variable: &regex::Captures| {
            let name = variable
                .get(1)
                .or(variable.get(2))
                .map_or("", |name| name.as_str());
            if allowed.iter().any(|allowed| allowed == name) {
                lookup(name).unwrap_or_default()
            } else {
                String::new()
            }
        })
        .into_owned()
}

/// The largest body an http hook may answer.
const MAX_HTTP_HOOK_BODY_BYTES: usize = 1 << 20;

/// Environment variable holding the hook's own data directory.
const HOOK_DATA_DIR_VAR: &str = "ZED_HOOK_DATA_DIR";

/// Seconds before a hook is cancelled, as in Claude Code.
const DEFAULT_HOOK_TIMEOUT_SECS: u64 = 600;

/// The timeout of a hook without one: as in Claude Code, shorter for
/// `UserPromptSubmit`, which holds the user's message.
fn default_hook_timeout(hook_event_name: &str) -> Duration {
    Duration::from_secs(match hook_event_name {
        "UserPromptSubmit" => 30,
        _ => DEFAULT_HOOK_TIMEOUT_SECS,
    })
}

/// What exit code 2 means for an event: a denial for `PreToolUse`, a block
/// for the others, with `reason`.
fn blocking_output(hook_event_name: &str, reason: &str) -> serde_json::Value {
    match hook_event_name {
        "PreToolUse" => json!({
            "hookSpecificOutput": {
                "hookEventName": hook_event_name,
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        }),
        _ => json!({ "decision": "block", "reason": reason }),
    }
}

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
    // A hook that succeeds can still warn, for example about settings it
    // does not apply.
    if exit_code == Some(0) && !stderr.trim().is_empty() {
        log::info!("agent hook {hook_event_name}: {}", stderr.trim());
    }
    match exit_code {
        Some(0) => match object {
            Some(object) => object.context("hook printed invalid JSON"),
            // As in Claude Code, plain text from `UserPromptSubmit` is context
            // for the model.
            None if hook_event_name == "UserPromptSubmit" && !stdout.is_empty() => Ok(json!({
                "hookSpecificOutput": {
                    "hookEventName": hook_event_name,
                    "additionalContext": stdout,
                }
            })),
            None => {
                if !stdout.is_empty() {
                    log::info!("agent hook printed plain text, ignored: {stdout}");
                }
                Ok(json!({}))
            }
        },
        Some(2) => Ok(blocking_output(hook_event_name, stderr.trim())),
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
    distinct_hooks(
        groups
            .iter()
            .filter(|group| matcher_matches(group.matcher.as_deref(), value)),
    )
}

/// The hooks of an event that has no field to match, in declaration order:
/// as in Claude Code, a matcher is ignored.
fn all_hooks(groups: &[HookMatcherGroup]) -> Vec<Arc<dyn HookHandler>> {
    distinct_hooks(groups.iter())
}

fn distinct_hooks<'a>(
    groups: impl Iterator<Item = &'a HookMatcherGroup>,
) -> Vec<Arc<dyn HookHandler>> {
    let mut seen = HashSet::default();
    groups
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

/// The options of the question a `PreToolUse` ask puts to the user.
const HOOK_ASK_ALLOW: &str = "allow";
const HOOK_ASK_DENY: &str = "deny";

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
                let mut asked: Option<String> = None;
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
                    // A pause comes before any decision: it refuses the call
                    // with its reason and stops the chain, like any refusal.
                    if let Some(stop_reason) = requested_pause(&output) {
                        cx.update(|cx| record_pause(&self.context.thread_id, stop_reason, cx));
                        notes.extend(context.map(str::to_string));
                        return Err(refused_output(stop_reason.to_string(), notes));
                    }
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
                        // An ask lets the chain go on, so that a later
                        // refusal still wins; the user answers once it ends.
                        Some("ask") => {
                            asked.get_or_insert_with(|| {
                                reason
                                    .unwrap_or("A hook asks to confirm this tool call.")
                                    .to_string()
                            });
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
                // A hook asked: the user decides, with the hook's reason.
                if let Some(reason) = asked {
                    let prompt = cx.update(|cx| {
                        event_stream.prompt_for_decision(
                            None,
                            Some(reason),
                            vec![
                                acp_v2::PermissionOption::new(
                                    acp_v2::PermissionOptionId::new(HOOK_ASK_ALLOW),
                                    "Allow",
                                    acp_v2::PermissionOptionKind::AllowOnce,
                                ),
                                acp_v2::PermissionOption::new(
                                    acp_v2::PermissionOptionId::new(HOOK_ASK_DENY),
                                    "Deny",
                                    acp_v2::PermissionOptionKind::RejectOnce,
                                ),
                            ],
                            cx,
                        )
                    });
                    let choice = futures::select_biased! {
                        _ = event_stream.cancelled_by_user().fuse() => return Err(cancelled_output()),
                        choice = prompt.fuse() => choice,
                    };
                    if !choice.is_ok_and(|choice| choice.0.as_ref() == HOOK_ASK_ALLOW) {
                        return Err(refused_output(
                            "The user refused this tool call, which a hook asked to confirm."
                                .to_string(),
                            notes,
                        ));
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

            // A pause lets the chain run to its end.
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
                if let Some(stop_reason) = requested_pause(&hook_output) {
                    cx.update(|cx| record_pause(&self.context.thread_id, stop_reason, cx));
                }
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
    match cx.update(|cx| ModelHooks::for_request(thread, &request, None, cx)) {
        Some(hooks) => {
            hooks
                .stream_completion(provider, model, request, cancellation, cx)
                .await
        }
        None => provider.stream_completion(model, request, cx).await,
    }
}

/// Streams the compaction summary like [`stream_completion`], and also runs
/// the `PreCompact` hooks, whose instructions extend the summary request, and
/// the `PostCompact` hooks, whose passages end the summary Zed keeps.
pub(crate) async fn stream_compaction_completion(
    provider: &Arc<dyn LanguageModelProvider>,
    model: &LanguageModel,
    request: LanguageModelRequest,
    thread: &WeakEntity<Thread>,
    manual: bool,
    cx: &AsyncApp,
) -> Result<LanguageModelCompletionStream, LanguageModelCompletionError> {
    let trigger = if manual {
        CompactionTrigger::Manual
    } else {
        CompactionTrigger::Auto
    };
    match cx.update(|cx| ModelHooks::for_request(thread, &request, Some(trigger), cx)) {
        Some(hooks) => {
            hooks
                .stream_completion(provider, model, request, None, cx)
                .await
        }
        None => provider.stream_completion(model, request, cx).await,
    }
}

#[derive(Clone, Copy)]
enum CompactionTrigger {
    Manual,
    Auto,
}

impl CompactionTrigger {
    fn name(self) -> &'static str {
        match self {
            Self::Manual => "manual",
            Self::Auto => "auto",
        }
    }
}

/// The hooks that run around one model request.
struct ModelHooks {
    context: HookContext,
    before_model: Vec<Arc<dyn HookHandler>>,
    after_model: Vec<Arc<dyn HookHandler>>,
    /// Set for the compaction summary request only.
    compaction_trigger: Option<CompactionTrigger>,
    pre_compact: Vec<Arc<dyn HookHandler>>,
    post_compact: Vec<Arc<dyn HookHandler>>,
    zed_model_error: Vec<Arc<dyn HookHandler>>,
}

impl ModelHooks {
    fn for_request(
        thread: &WeakEntity<Thread>,
        request: &LanguageModelRequest,
        compaction_trigger: Option<CompactionTrigger>,
        cx: &mut App,
    ) -> Option<Self> {
        let intent = intent_name(request.intent).unwrap_or_default();
        let compaction_trigger = compaction_trigger.filter(|_| is_compaction_request(request));
        let (before_model, after_model, pre_compact, post_compact, zed_model_error) =
            with_configured_hooks(cx, |hooks| {
                let compaction_hooks = |groups: &[HookMatcherGroup]| {
                    compaction_trigger
                        .map(|trigger| matching_hooks(groups, trigger.name()))
                        .unwrap_or_default()
                };
                (
                    matching_hooks(&hooks.before_model, intent),
                    matching_hooks(&hooks.after_model, intent),
                    compaction_hooks(&hooks.pre_compact),
                    compaction_hooks(&hooks.post_compact),
                    matching_hooks(&hooks.zed_model_error, intent),
                )
            })?;
        if before_model.is_empty()
            && after_model.is_empty()
            && pre_compact.is_empty()
            && post_compact.is_empty()
            && zed_model_error.is_empty()
        {
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
            compaction_trigger,
            pre_compact,
            post_compact,
            zed_model_error,
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
        if let Some(trigger) = self.compaction_trigger
            && !self.pre_compact.is_empty()
        {
            self.run_pre_compact(trigger, &mut request, cx).await;
        }
        let request_fields = json!({
            "request_id": Uuid::new_v4().to_string(),
            "intent": intent_name(request.intent),
            "model": {
                "provider": model.provider_id().to_string(),
                "id": model.id().0.to_string(),
            },
            "is_compaction": is_compaction_request(&request),
        });
        let error_cancellation = cancellation.clone();
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
        let error_hooks = (!self.zed_model_error.is_empty()).then(|| ErrorHooks {
            context: self.context.clone(),
            hooks: self.zed_model_error.clone(),
            request_fields: request_fields.clone(),
        });
        let mut stream = match provider.stream_completion(model, request, cx).await {
            Ok(stream) => stream,
            Err(error) => {
                return Err(match &error_hooks {
                    Some(error_hooks) => error_hooks.translate(error, error_cancellation, cx).await,
                    None => error,
                });
            }
        };
        if let Some(error_hooks) = error_hooks {
            stream = translate_stream_errors(stream, error_hooks, cx);
        }
        if let Some(trigger) = self.compaction_trigger
            && !self.post_compact.is_empty()
        {
            stream = append_post_compact(
                stream,
                self.post_compact.clone(),
                self.context.clone(),
                trigger,
                cx,
            );
        }
        if self.after_model.is_empty() {
            return Ok(stream);
        }
        let (end_tx, end_rx) = oneshot::channel();
        let (mut usage_tx, usage_rx) = oneshot::channel();
        cx.spawn(async move |cx| {
            // The sender is dropped when the stream fails or is abandoned:
            // AfterModel only observes generations that end normally.
            let Ok(end) = end_rx.await else {
                log::info!("AfterModel hooks skipped: the response stream did not end normally");
                return;
            };
            // Closing the stream, for example on Stop, abandons the hooks.
            let usage = {
                let run = self.run_after_model(request_fields, end, started_at, cx);
                let abandoned = usage_tx.cancellation();
                futures::pin_mut!(run, abandoned);
                match futures::future::select(run, abandoned).await {
                    futures::future::Either::Left((usage, _)) => usage,
                    futures::future::Either::Right(_) => return,
                }
            };
            usage_tx.send(usage).ok();
        })
        .detach();
        Ok(observe_response_end(stream, end_tx, usage_rx))
    }

    /// Runs the `PreCompact` hooks in declaration order and appends their
    /// instructions to the summary instruction. They never prevent the
    /// compaction.
    async fn run_pre_compact(
        &self,
        trigger: CompactionTrigger,
        request: &mut LanguageModelRequest,
        cx: &AsyncApp,
    ) {
        let mut instructions = Vec::new();
        for hook in &self.pre_compact {
            let input = self.context.input(
                "PreCompact",
                json!({ "trigger": trigger.name(), "custom_instructions": null }),
            );
            let output = match cx.update(|cx| hook.call(input, cx)).await {
                Ok(output) => output,
                Err(error) => {
                    log::warn!("PreCompact hook failed, continuing: {error:#}");
                    continue;
                }
            };
            if let Err(error) = check_event_name("PreCompact", &output) {
                log::warn!("PreCompact hook output ignored: {error:#}");
                continue;
            }
            if output.get("decision").is_some_and(|value| !value.is_null()) {
                log::warn!(
                    "a PreCompact hook cannot prevent the compaction; its refusal is ignored \
                     (reason: {})",
                    output["reason"].as_str().unwrap_or("none given")
                );
            }
            log_unsupported_output_fields(&output, &UNSUPPORTED_COMPACTION_FIELDS);
            warn_on_unknown_zed_fields("PreCompact", &output, &["additional_instructions"]);
            match hook_text(&output, "/hookSpecificOutput/zed/additional_instructions") {
                Ok(text) => instructions.extend(text.map(str::to_string)),
                Err(error) => log::warn!("PreCompact hook output ignored: {error:#}"),
            }
        }
        if !instructions.is_empty()
            && let Some(instruction) = request.messages.last_mut()
        {
            instruction.content.push(MessageContent::Text(format!(
                "\n\n{}",
                instructions.join("\n\n")
            )));
        }
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
        // Only the conversation counts: the system message changes with the
        // model, the date or the tools. What the hooks already took is
        // remembered as the conversation Zed holds, before any replacement.
        let conversation_hashes = conversation_messages(request)
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
            fields.insert("total_message_count".into(), json!(request.messages.len()));
            fields.insert("tool_names".into(), json!(tool_names));
            fields.insert("skill_names".into(), json!(catalogue_skill_names(request)));
        }

        let mut inserts = Vec::new();
        let mut insert_chars = 0;
        let mut capped = false;
        let mut reminders = Vec::new();
        let mut reminder_chars = 0;
        let mut reminders_capped = false;
        let mut removed_tools = Vec::<String>::new();
        let mut removed_skills = Vec::<String>::new();
        for hook in &self.before_model {
            // Each hook gets the messages added since its own previous answer.
            let sent_key = (self.context.thread_id.clone(), hook_identity(hook));
            let known = cx.update(|cx| known_message_count(&sent_key, &conversation_hashes, cx));
            // The chain: each hook gets the replacements and the output reserve
            // of the previous ones.
            let conversation = conversation_messages(request);
            let mut hook_fields = event_fields.clone();
            if let Some(fields) = hook_fields.as_object_mut() {
                fields.insert("new_messages".into(), json!(conversation[known..]));
                fields.insert("known_message_count".into(), json!(known));
                fields.insert(
                    "max_output_tokens".into(),
                    json!(request.effective_max_output_tokens(model.max_output_tokens())),
                );
            }
            let input = self.context.input("BeforeModel", hook_fields);
            let output = match cx.update(|cx| hook.call(input, cx)).await {
                Ok(output) => output,
                Err(error) => {
                    log::warn!("BeforeModel hook failed, continuing: {error:#}");
                    continue;
                }
            };
            let (
                insert,
                hook_reminders,
                replacements,
                max_output_tokens,
                hook_removed_tools,
                hook_removed_skills,
            ) = match check_model_hook_output("BeforeModel", &output).and_then(|()| {
                Ok((
                    system_prompt_insert(&output)?,
                    ephemeral_messages(&output)?,
                    tool_result_replacements(&output, request)?,
                    hook_max_output_tokens(&output)?,
                    remove_tools(&output)?,
                    remove_skills(&output)?,
                ))
            }) {
                Ok(fields) => fields,
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
            log_unsupported_output_fields(&output, &["/continue", "/systemMessage"]);
            replace_tool_results(request, replacements);
            // The union of the lists of all hooks.
            for (removed, names) in [
                (&mut removed_tools, hook_removed_tools),
                (&mut removed_skills, hook_removed_skills),
            ] {
                for name in names {
                    if !removed.iter().any(|removed| removed == name) {
                        removed.push(name.to_string());
                    }
                }
            }
            if let Some(max_output_tokens) = max_output_tokens {
                // Only lowers the reserve; a higher value is ignored.
                let current = request.effective_max_output_tokens(model.max_output_tokens());
                if current.is_none_or(|current| max_output_tokens < current) {
                    request.max_output_tokens = Some(max_output_tokens);
                }
            }
            if !hook_reminders.is_empty() {
                // The cap applies to the reminders of all hooks together.
                let added = hook_reminders
                    .iter()
                    .map(|reminder| reminder.chars().count())
                    .sum::<usize>();
                if reminders_capped || reminder_chars + added > MAX_SYSTEM_PROMPT_INSERT_CHARS {
                    reminders_capped = true;
                    log::warn!(
                        "BeforeModel reminders exceed {MAX_SYSTEM_PROMPT_INSERT_CHARS} \
                         characters together; the reminders of this hook are ignored"
                    );
                } else {
                    reminder_chars += added;
                    reminders.extend(hook_reminders.into_iter().map(str::to_string));
                }
            }
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
        remove_request_tools(request, &removed_tools);
        remove_request_skills(request, &removed_skills);
        // Reminders go last, in this request only: the conversation stays an
        // identical prefix from one request to the next. Summaries get none.
        if applies_ephemeral_messages(request) {
            request
                .messages
                .extend(reminders.iter().map(|reminder| reminder_message(reminder)));
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

    /// Runs the `AfterModel` hooks in declaration order. A `usage` they return
    /// replaces the reported one for the next hooks; the last one is returned,
    /// for Zed to record as the provider's.
    async fn run_after_model(
        &self,
        mut event_fields: serde_json::Value,
        end: ResponseEnd,
        started_at: Instant,
        cx: &AsyncApp,
    ) -> Option<TokenUsage> {
        let mut reported_usage = None;
        if let Some(fields) = event_fields.as_object_mut() {
            fields.insert(
                "stop_reason".into(),
                json!(end.stop_reason.map(stop_reason_name)),
            );
            fields.insert("usage".into(), usage_to_contract(&end.usage));
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
            let usage = match check_model_hook_output("AfterModel", &output)
                .and_then(|()| hook_usage(&output))
            {
                Ok(usage) => usage,
                Err(error) => {
                    log::warn!("AfterModel hook output ignored: {error:#}");
                    continue;
                }
            };
            log_unsupported_output_fields(&output, &["/continue", "/systemMessage"]);
            if let Some(usage) = usage {
                if let Some(fields) = event_fields.as_object_mut() {
                    fields.insert("usage".into(), usage_to_contract(&usage));
                }
                reported_usage = Some(usage);
            }
        }
        reported_usage
    }
}

/// The `ZedModelError` hooks of one request.
#[derive(Clone)]
struct ErrorHooks {
    context: HookContext,
    hooks: Vec<Arc<dyn HookHandler>>,
    request_fields: serde_json::Value,
}

impl ErrorHooks {
    /// Runs the hooks in declaration order on a provider rejection: each one
    /// may change the category Zed acts on, for example into a full context
    /// window, which makes Zed compact. Other errors pass unchanged, and so
    /// does the rejection when the request is cancelled while the hooks run.
    async fn translate(
        &self,
        error: LanguageModelCompletionError,
        cancellation: Option<watch::Receiver<bool>>,
        cx: &AsyncApp,
    ) -> LanguageModelCompletionError {
        let LanguageModelCompletionError::ProviderRejection {
            provider,
            status,
            code,
            message,
            retry_after,
            category,
        } = error
        else {
            return error;
        };
        let original_category = category.clone();
        let category = {
            let translated =
                self.translate_category(status, &code, &message, retry_after, &category, cx);
            let cancelled = wait_for_cancellation(cancellation);
            futures::pin_mut!(translated, cancelled);
            match futures::future::select(translated, cancelled).await {
                futures::future::Either::Left((translated, _)) => translated,
                // Dropping the hooks' tasks kills their processes.
                futures::future::Either::Right(_) => original_category,
            }
        };
        LanguageModelCompletionError::ProviderRejection {
            provider,
            status,
            code,
            message,
            retry_after,
            category,
        }
    }

    async fn translate_category(
        &self,
        status: Option<http_client::StatusCode>,
        code: &Option<String>,
        message: &str,
        retry_after: Option<Duration>,
        category: &ProviderErrorCategory,
        cx: &AsyncApp,
    ) -> ProviderErrorCategory {
        let mut category = category.clone();
        let request_id = self.request_fields["request_id"]
            .as_str()
            .unwrap_or_default();
        let short_message = message.chars().take(200).collect::<String>();
        for hook in &self.hooks {
            let mut fields = self.request_fields.clone();
            if let Some(fields) = fields.as_object_mut() {
                fields.remove("is_compaction");
                fields.insert(
                    "provider_error".into(),
                    json!({
                        "status": status.map(|status| status.as_u16()),
                        "code": code,
                        "message": message,
                        "zed_category": category_name(&category),
                        "retry_after_ms": retry_after.map(|delay| delay.as_millis() as u64),
                    }),
                );
            }
            let input = self.context.input("ZedModelError", fields);
            let output = match cx.update(|cx| hook.call(input, cx)).await {
                Ok(output) => output,
                Err(error) => {
                    log::warn!(
                        "ZedModelError hook failed, continuing (request {request_id}, status \
                         {status:?}, category {}): {error:#}",
                        category_name(&category)
                    );
                    continue;
                }
            };
            if let Err(error) = check_model_hook_output("ZedModelError", &output) {
                log::warn!(
                    "ZedModelError hook output ignored (request {request_id}, status \
                     {status:?}, category {}): {error:#}",
                    category_name(&category)
                );
                continue;
            }
            // Deviation from the 0.1 contract: `systemMessage` is not shown to
            // the user yet, only logged.
            log_unsupported_output_fields(&output, &["/continue", "/systemMessage"]);
            let name = match output.pointer("/hookSpecificOutput/category") {
                None | Some(serde_json::Value::Null) => continue,
                Some(name) => name.as_str().unwrap_or_default(),
            };
            let prompt_tokens = output
                .pointer("/hookSpecificOutput/prompt_tokens")
                .and_then(serde_json::Value::as_u64);
            // A hook that turns a rejection into `rate_limit`, `overloaded` or
            // `timeout` makes it transient: Zed retries the request, up to
            // `MAX_RETRY_ATTEMPTS`, and runs these hooks again each time.
            match parse_category(name, prompt_tokens) {
                Some(new_category) => {
                    if new_category != category {
                        log::info!(
                            "ZedModelError hook changed the category from {} to {} (request \
                             {request_id}, status {status:?}): {short_message}",
                            category_name(&category),
                            category_name(&new_category)
                        );
                    }
                    category = new_category;
                }
                None => log::warn!(
                    "ZedModelError hook named an unknown category {name:?} (request \
                     {request_id}, status {status:?})"
                ),
            }
        }
        category
    }
}

/// Runs the `ZedModelError` hooks on the provider rejections of a stream.
fn translate_stream_errors(
    stream: LanguageModelCompletionStream,
    hooks: ErrorHooks,
    cx: &AsyncApp,
) -> LanguageModelCompletionStream {
    let (errors_tx, mut errors_rx) = mpsc::unbounded::<(
        LanguageModelCompletionError,
        oneshot::Sender<LanguageModelCompletionError>,
    )>();
    // Not detached: dropping the stream drops this task, which abandons the
    // hooks and kills their processes.
    let translator = cx.spawn(async move |cx| {
        while let Some((error, reply_tx)) = errors_rx.next().await {
            reply_tx.send(hooks.translate(error, None, cx).await).ok();
        }
    });
    stream
        .then(move |event| {
            let _translator = &translator;
            let errors_tx = errors_tx.clone();
            async move {
                match event {
                    Err(error @ LanguageModelCompletionError::ProviderRejection { .. }) => {
                        let (reply_tx, reply_rx) = oneshot::channel();
                        if let Err(send_error) = errors_tx.unbounded_send((error, reply_tx)) {
                            return Err(send_error.into_inner().0);
                        }
                        Err(reply_rx.await.unwrap_or_else(|_| {
                            anyhow!("the ZedModelError hooks were abandoned").into()
                        }))
                    }
                    event => event,
                }
            }
        })
        .boxed()
}

/// A text field of a hook output, trimmed. A field that is present must be
/// a non-empty text.
fn hook_text<'a>(output: &'a serde_json::Value, pointer: &str) -> Result<Option<&'a str>> {
    match output.pointer(pointer) {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(value) => {
            let text = value
                .as_str()
                .map(str::trim)
                .filter(|text| !text.is_empty())
                .with_context(|| format!("`{pointer}` must be a non-empty text"))?;
            Ok(Some(text))
        }
    }
}

/// Warns about a `zed` object that holds none of the event's fields, or
/// other fields: a misspelt name would otherwise be ignored silently.
fn warn_on_unknown_zed_fields(event: &str, output: &serde_json::Value, fields: &[&str]) {
    let Some(zed) = output
        .pointer("/hookSpecificOutput/zed")
        .and_then(serde_json::Value::as_object)
    else {
        return;
    };
    let unknown = zed
        .keys()
        .filter(|key| !fields.contains(&key.as_str()))
        .collect::<Vec<_>>();
    if !unknown.is_empty() || !fields.iter().any(|field| zed.contains_key(*field)) {
        log::warn!(
            "{event} hook output: the `zed` object should hold {fields:?}; ignored fields: \
             {unknown:?}"
        );
    }
}

/// Output fields of the compaction events this version of Zed does not apply.
const UNSUPPORTED_COMPACTION_FIELDS: [&str; 3] = ["/continue", "/systemMessage", "/stopReason"];

/// Streams the compaction summary unchanged, then runs the `PostCompact`
/// hooks with it and streams their passages as its end, so that they become
/// part of the summary Zed keeps. An empty summary gets no passage: Zed still
/// refuses it.
fn append_post_compact(
    stream: LanguageModelCompletionStream,
    hooks: Vec<Arc<dyn HookHandler>>,
    context: HookContext,
    trigger: CompactionTrigger,
    cx: &AsyncApp,
) -> LanguageModelCompletionStream {
    let (summary_tx, summary_rx) = oneshot::channel::<String>();
    let (mut passage_tx, passage_rx) = oneshot::channel::<String>();
    cx.spawn(async move |cx| {
        let Ok(summary) = summary_rx.await else {
            return;
        };
        // Stop the hooks when the compaction is abandoned.
        let passage = {
            let run = run_post_compact(&hooks, &context, trigger, summary, cx);
            let abandoned = passage_tx.cancellation();
            futures::pin_mut!(run, abandoned);
            match futures::future::select(run, abandoned).await {
                futures::future::Either::Left((passage, _)) => passage,
                futures::future::Either::Right(_) => return,
            }
        };
        if let Some(passage) = passage {
            passage_tx.send(passage).ok();
        }
    })
    .detach();

    struct State {
        stream: LanguageModelCompletionStream,
        summary: String,
        summary_tx: Option<oneshot::Sender<String>>,
        passage_rx: Option<oneshot::Receiver<String>>,
    }
    let state = State {
        // Some provider streams panic when polled after their end.
        stream: stream.fuse().boxed(),
        summary: String::new(),
        summary_tx: Some(summary_tx),
        passage_rx: Some(passage_rx),
    };
    futures::stream::unfold(state, |mut state| async move {
        if let Some(event) = state.stream.next().await {
            match &event {
                Ok(LanguageModelCompletionEvent::Text(text)) => state.summary.push_str(text),
                Err(_) => {
                    state.summary_tx = None;
                    state.passage_rx = None;
                }
                Ok(_) => {}
            }
            return Some((event, state));
        }
        let summary_tx = state.summary_tx.take()?;
        let passage_rx = state.passage_rx.take()?;
        let summary = state.summary.trim();
        if summary.is_empty() {
            return None;
        }
        summary_tx.send(summary.to_string()).ok()?;
        let passage = passage_rx.await.ok()?;
        Some((
            Ok(LanguageModelCompletionEvent::Text(format!("\n\n{passage}"))),
            state,
        ))
    })
    .boxed()
}

/// Runs the `PostCompact` hooks in declaration order: each one gets the
/// summary with the passages of the previous ones. Returns the passages.
async fn run_post_compact(
    hooks: &[Arc<dyn HookHandler>],
    context: &HookContext,
    trigger: CompactionTrigger,
    mut summary: String,
    cx: &AsyncApp,
) -> Option<String> {
    let mut passages = Vec::new();
    for hook in hooks {
        let input = context.input(
            "PostCompact",
            json!({ "trigger": trigger.name(), "compact_summary": summary }),
        );
        let output = match cx.update(|cx| hook.call(input, cx)).await {
            Ok(output) => output,
            Err(error) => {
                log::warn!("PostCompact hook failed, continuing: {error:#}");
                continue;
            }
        };
        if let Err(error) = check_event_name("PostCompact", &output) {
            log::warn!("PostCompact hook output ignored: {error:#}");
            continue;
        }
        // A code 2 arrives as a refusal: Claude Code shows its stderr to the
        // user only.
        if output.get("decision").is_some_and(|value| !value.is_null()) {
            log::warn!(
                "a PostCompact hook has no decision to make; ignored (reason: {})",
                output["reason"].as_str().unwrap_or("none given")
            );
        }
        log_unsupported_output_fields(&output, &UNSUPPORTED_COMPACTION_FIELDS);
        warn_on_unknown_zed_fields("PostCompact", &output, &["append_to_summary"]);
        match hook_text(&output, "/hookSpecificOutput/zed/append_to_summary") {
            Ok(Some(passage)) => {
                summary = format!("{summary}\n\n{passage}");
                passages.push(passage.to_string());
            }
            Ok(None) => {}
            Err(error) => log::warn!("PostCompact hook output ignored: {error:#}"),
        }
    }
    (!passages.is_empty()).then(|| passages.join("\n\n"))
}

/// The non-system messages of a request, in the contract's shape.
fn conversation_messages(request: &LanguageModelRequest) -> Vec<serde_json::Value> {
    request
        .messages
        .iter()
        .filter(|message| message.role != Role::System)
        .map(message_to_contract)
        .collect()
}

/// The `tool_result_replacements` of a `BeforeModel` output. Each one must
/// target a tool result of this request.
fn tool_result_replacements(
    output: &serde_json::Value,
    request: &LanguageModelRequest,
) -> Result<Vec<(String, String)>> {
    let replacements = match output.pointer("/hookSpecificOutput/tool_result_replacements") {
        None | Some(serde_json::Value::Null) => return Ok(Vec::new()),
        Some(replacements) => replacements
            .as_array()
            .context("`tool_result_replacements` must be a list")?,
    };
    replacements
        .iter()
        .map(|replacement| {
            let tool_use_id = replacement["tool_use_id"]
                .as_str()
                .filter(|id| !id.is_empty())
                .context("a replacement needs a `tool_use_id`")?;
            let text = replacement["text"]
                .as_str()
                .context("a replacement needs a `text`")?;
            let targets_a_result = request
                .messages
                .iter()
                .flat_map(|message| &message.content)
                .any(|part| {
                    matches!(part, MessageContent::ToolResult(result)
                        if result.tool_use_id.to_string() == tool_use_id)
                });
            anyhow::ensure!(
                targets_a_result,
                "no tool result `{tool_use_id}` in this request"
            );
            Ok((tool_use_id.to_string(), text.to_string()))
        })
        .collect()
}

/// Replaces the text of tool results, in this request only.
fn replace_tool_results(request: &mut LanguageModelRequest, replacements: Vec<(String, String)>) {
    for (tool_use_id, text) in replacements {
        for part in request
            .messages
            .iter_mut()
            .flat_map(|message| &mut message.content)
        {
            if let MessageContent::ToolResult(result) = part
                && result.tool_use_id.to_string() == tool_use_id
            {
                result.content = vec![LanguageModelToolResultContent::Text(text.clone().into())];
            }
        }
    }
}

/// The `max_output_tokens` of a `BeforeModel` output: a whole number, at
/// least 1.
fn hook_max_output_tokens(output: &serde_json::Value) -> Result<Option<u64>> {
    match output.pointer("/hookSpecificOutput/max_output_tokens") {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(value) => {
            let max_output_tokens = value
                .as_u64()
                .filter(|max| *max >= 1)
                .context("`max_output_tokens` must be a whole number, at least 1")?;
            Ok(Some(max_output_tokens))
        }
    }
}

/// The `usage` of an `AfterModel` output, with its four counts.
fn hook_usage(output: &serde_json::Value) -> Result<Option<TokenUsage>> {
    let usage = match output.pointer("/hookSpecificOutput/usage") {
        None | Some(serde_json::Value::Null) => return Ok(None),
        Some(usage) => usage,
    };
    // At most 2^53 - 1, the largest integer JSON carries exactly; Zed adds the
    // counts together, so their sum must not overflow either.
    const MAX_COUNT: u64 = (1 << 53) - 1;
    let count = |field: &str| {
        usage[field]
            .as_u64()
            .filter(|count| *count <= MAX_COUNT)
            .with_context(|| format!("`usage.{field}` must be a whole number, at most {MAX_COUNT}"))
    };
    let usage = TokenUsage {
        input_tokens: count("input_tokens")?,
        output_tokens: count("output_tokens")?,
        cache_creation_input_tokens: count("cache_creation_input_tokens")?,
        cache_read_input_tokens: count("cache_read_input_tokens")?,
    };
    usage
        .input_tokens
        .checked_add(usage.output_tokens)
        .and_then(|total| total.checked_add(usage.cache_creation_input_tokens))
        .and_then(|total| total.checked_add(usage.cache_read_input_tokens))
        .context("the `usage` counts add up beyond what Zed can hold")?;
    Ok(Some(usage))
}

fn usage_to_contract(usage: &TokenUsage) -> serde_json::Value {
    json!({
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_creation_input_tokens": usage.cache_creation_input_tokens,
        "cache_read_input_tokens": usage.cache_read_input_tokens,
    })
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

/// Passes the response through unchanged. Once it ends without error, sends
/// how it ended to the `AfterModel` hooks and waits for them; a usage they
/// return ends the stream as one more usage update, which Zed records like
/// the provider's: in the turn, before its next request, and not for a
/// compaction, whose usage is not the context size.
fn observe_response_end(
    stream: LanguageModelCompletionStream,
    end_tx: oneshot::Sender<ResponseEnd>,
    usage_rx: oneshot::Receiver<Option<TokenUsage>>,
) -> LanguageModelCompletionStream {
    let end = ResponseEnd {
        stop_reason: None,
        usage: TokenUsage::default(),
        ended_at: Instant::now(),
    };
    futures::stream::unfold(
        // Some provider streams panic when polled after their end.
        (stream.fuse().boxed(), Some((end_tx, end, usage_rx))),
        |(mut stream, mut end)| async move {
            let Some(event) = stream.next().await else {
                let (end_tx, mut end, usage_rx) = end.take()?;
                end.ended_at = Instant::now();
                end_tx.send(end).ok()?;
                let usage = usage_rx.await.ok()??;
                return Some((
                    Ok(LanguageModelCompletionEvent::UsageUpdate(usage)),
                    (stream, None),
                ));
            };
            match (&event, end.as_mut()) {
                (Ok(LanguageModelCompletionEvent::Stop(reason)), Some((_, end, _))) => {
                    end.stop_reason = Some(*reason);
                }
                (Ok(LanguageModelCompletionEvent::UsageUpdate(usage)), Some((_, end, _))) => {
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
/// compaction summary, nor to the title.
fn applies_system_prompt_insert(request: &LanguageModelRequest) -> bool {
    !is_compaction_request(request) && request.intent != Some(CompletionIntent::ThreadSummarization)
}

/// The contract's snake_case name of an error category.
fn category_name(category: &ProviderErrorCategory) -> &'static str {
    match category {
        ProviderErrorCategory::PromptTooLarge { .. } => "prompt_too_large",
        ProviderErrorCategory::InvalidEncryptedContent => "invalid_encrypted_content",
        ProviderErrorCategory::Authentication => "authentication",
        ProviderErrorCategory::Permission => "permission",
        ProviderErrorCategory::ContentPolicy => "content_policy",
        ProviderErrorCategory::EndpointNotFound => "endpoint_not_found",
        ProviderErrorCategory::PaymentRequired => "payment_required",
        ProviderErrorCategory::RateLimit => "rate_limit",
        ProviderErrorCategory::Overloaded => "overloaded",
        ProviderErrorCategory::InvalidRequest => "invalid_request",
        ProviderErrorCategory::Conflict => "conflict",
        ProviderErrorCategory::Timeout => "timeout",
        ProviderErrorCategory::InternalServer => "internal_server",
        ProviderErrorCategory::Other => "other",
    }
}

/// The error category a hook names, with its token count for a full window.
fn parse_category(name: &str, prompt_tokens: Option<u64>) -> Option<ProviderErrorCategory> {
    Some(match name {
        "prompt_too_large" => ProviderErrorCategory::PromptTooLarge {
            tokens: prompt_tokens,
        },
        "invalid_encrypted_content" => ProviderErrorCategory::InvalidEncryptedContent,
        "authentication" => ProviderErrorCategory::Authentication,
        "permission" => ProviderErrorCategory::Permission,
        "content_policy" => ProviderErrorCategory::ContentPolicy,
        "endpoint_not_found" => ProviderErrorCategory::EndpointNotFound,
        "payment_required" => ProviderErrorCategory::PaymentRequired,
        "rate_limit" => ProviderErrorCategory::RateLimit,
        "overloaded" => ProviderErrorCategory::Overloaded,
        "invalid_request" => ProviderErrorCategory::InvalidRequest,
        "conflict" => ProviderErrorCategory::Conflict,
        "timeout" => ProviderErrorCategory::Timeout,
        "internal_server" => ProviderErrorCategory::InternalServer,
        "other" => ProviderErrorCategory::Other,
        _ => return None,
    })
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

/// The `ephemeral_messages` of a `BeforeModel` output: texts for the last
/// position of this request only.
fn ephemeral_messages(output: &serde_json::Value) -> Result<Vec<&str>> {
    let messages = match output.pointer("/hookSpecificOutput/ephemeral_messages") {
        None | Some(serde_json::Value::Null) => return Ok(Vec::new()),
        Some(messages) => messages
            .as_array()
            .context("`ephemeral_messages` must be a list of texts")?,
    };
    messages
        .iter()
        .map(|message| {
            let text = message
                .as_str()
                .context("`ephemeral_messages` must be a list of texts")?;
            anyhow::ensure!(!text.is_empty(), "an `ephemeral_messages` text is empty");
            let length = text.chars().count();
            anyhow::ensure!(
                length <= MAX_SYSTEM_PROMPT_INSERT_CHARS,
                "an `ephemeral_messages` text has {length} characters, more than \
                 {MAX_SYSTEM_PROMPT_INSERT_CHARS}"
            );
            Ok(text)
        })
        .collect()
}

/// Whether `ephemeral_messages` apply to this request.
/// Not to the title, the panel summary, nor the compaction.
fn applies_ephemeral_messages(request: &LanguageModelRequest) -> bool {
    !matches!(
        request.intent,
        Some(CompletionIntent::ThreadSummarization | CompletionIntent::ThreadContextSummarization)
    )
}

/// Neutralises a `<system-reminder>` tag inside a text, so that it cannot
/// close the reminder it is wrapped in. Stricter than Claude Code: any case,
/// and spaces inside the tag.
fn escape_reminder_tags(text: &str) -> std::borrow::Cow<'_, str> {
    static TAG: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?i)<(\s*/?\s*system-reminder)").unwrap());
    TAG.replace_all(text, "&lt;$1")
}

/// A reminder as Zed sends it: a user message wrapped in `<system-reminder>`.
/// The role is `user`: vLLM refuses a `system` message after the first one.
/// Like Claude Code, a `<system-reminder>` tag inside the text is neutralised.
fn reminder_message(text: &str) -> LanguageModelRequestMessage {
    let text = escape_reminder_tags(text);
    LanguageModelRequestMessage {
        role: Role::User,
        content: vec![MessageContent::Text(format!(
            "<system-reminder>\n{text}\n</system-reminder>"
        ))],
        cache: false,
        reasoning_details: None,
    }
}

/// The `remove_tools` of a `BeforeModel` output: distinct, non-empty tool
/// names.
fn remove_tools(output: &serde_json::Value) -> Result<Vec<&str>> {
    hook_names(output, "remove_tools")
}

/// The `remove_skills` of a `BeforeModel` output: distinct, non-empty skill
/// names.
fn remove_skills(output: &serde_json::Value) -> Result<Vec<&str>> {
    hook_names(output, "remove_skills")
}

fn hook_names<'a>(output: &'a serde_json::Value, field: &str) -> Result<Vec<&'a str>> {
    let names = match output.pointer(&format!("/hookSpecificOutput/{field}")) {
        None | Some(serde_json::Value::Null) => return Ok(Vec::new()),
        Some(names) => names
            .as_array()
            .with_context(|| format!("`{field}` must be a list of names"))?,
    };
    let mut removed = Vec::new();
    for name in names {
        let name = name
            .as_str()
            .filter(|name| !name.is_empty())
            .with_context(|| format!("`{field}` must be a list of names"))?;
        anyhow::ensure!(!removed.contains(&name), "`{field}` names `{name}` twice");
        removed.push(name);
    }
    Ok(removed)
}

/// One `<skill>` entry of the `<available_skills>` catalogue, with its name.
static SKILL_ENTRY: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)[ \t]*<skill>\s*<name>(.*?)</name>.*?</skill>\n?")
        .expect("valid skill entry pattern")
});

/// The names in the `<available_skills>` catalogue of the system message of
/// this request, sorted; none without a catalogue.
fn catalogue_skill_names(request: &LanguageModelRequest) -> Vec<String> {
    let mut names = request
        .messages
        .iter()
        .filter(|message| message.role == Role::System)
        .flat_map(|message| &message.content)
        .filter_map(|content| match content {
            MessageContent::Text(text) => Some(text),
            _ => None,
        })
        .filter_map(|text| {
            let start = text.find("<available_skills>")?;
            let end = text.find("</available_skills>")?;
            (start < end).then(|| &text[start..end])
        })
        .flat_map(|catalogue| SKILL_ENTRY.captures_iter(catalogue))
        .map(|entry| entry[1].to_string())
        .collect::<Vec<_>>();
    names.sort_unstable();
    names.dedup();
    names
}

/// Removes the `<skill>` entries of these names from the `<available_skills>`
/// catalogue in the system message of this request, not of the compaction
/// request. Hiding a skill does not refuse it.
fn remove_request_skills(request: &mut LanguageModelRequest, names: &[String]) {
    if names.is_empty() || is_compaction_request(request) {
        return;
    }
    let mut found = HashSet::default();
    for message in &mut request.messages {
        if message.role != Role::System {
            continue;
        }
        for content in &mut message.content {
            let MessageContent::Text(text) = content else {
                continue;
            };
            let (Some(start), Some(end)) = (
                text.find("<available_skills>"),
                text.find("</available_skills>"),
            ) else {
                continue;
            };
            if end < start {
                continue;
            }
            let catalogue = SKILL_ENTRY
                .replace_all(&text[start..end], |entry: &regex::Captures| {
                    let name = &entry[1];
                    if names.iter().any(|removed| removed == name) {
                        found.insert(name.to_string());
                        String::new()
                    } else {
                        entry[0].to_string()
                    }
                })
                .into_owned();
            text.replace_range(start..end, &catalogue);
        }
    }
    for name in names {
        if !found.contains(name) {
            log::info!(
                "a BeforeModel hook removes the skill `{name}`, which this request does not list"
            );
        }
    }
}

/// Removes tools from the list this request shows the model, not from the
/// compaction request; their calls are not refused.
fn remove_request_tools(request: &mut LanguageModelRequest, names: &[String]) {
    if names.is_empty() || is_compaction_request(request) {
        return;
    }
    for name in names {
        if !request.tools.iter().any(|tool| &tool.name == name) {
            log::info!("a BeforeModel hook removes `{name}`, which this request does not have");
        }
    }
    request.tools.retain(|tool| !names.contains(&tool.name));
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

/// Pauses asked with `continue: false`, by thread id, with their `stopReason`.
#[derive(Default)]
struct RequestedPauses(HashMap<String, String>);

impl Global for RequestedPauses {}

/// The `stopReason` of the pause a hook output asks for with
/// `continue: false`. A pause is not a cut: it comes before the next request
/// to the model.
fn requested_pause(output: &serde_json::Value) -> Option<&str> {
    match &output["continue"] {
        serde_json::Value::Bool(false) => Some(
            output["stopReason"]
                .as_str()
                .filter(|reason| !reason.trim().is_empty())
                .unwrap_or("A hook paused the agent."),
        ),
        serde_json::Value::Null | serde_json::Value::Bool(true) => None,
        value => {
            log::warn!("agent hook `continue` must be a boolean, not {value}; ignored");
            None
        }
    }
}

pub(crate) fn record_pause(thread_id: &str, stop_reason: &str, cx: &mut App) {
    cx.default_global::<RequestedPauses>()
        .0
        .insert(thread_id.to_string(), stop_reason.to_string());
}

pub(crate) fn take_pause(thread_id: &str, cx: &mut App) -> Option<String> {
    if !cx.has_global::<RequestedPauses>() {
        return None;
    }
    cx.global_mut::<RequestedPauses>().0.remove(thread_id)
}

fn pause_pending(thread_id: &str, cx: &App) -> bool {
    cx.try_global::<RequestedPauses>()
        .is_some_and(|pauses| pauses.0.contains_key(thread_id))
}

/// Ends the turn when a hook asked for a pause since it began: Zed adds the
/// pause to the conversation, and a subagent passes it on to its parent, so
/// that the conversation waits for the user. Returns whether it paused.
pub(crate) fn pause_turn(
    thread: &WeakEntity<Thread>,
    event_stream: &ThreadEventStream,
    cx: &mut AsyncApp,
) -> Result<bool> {
    let pause = thread.update(cx, |thread, cx| {
        let pause = take_pause(&thread.id().to_string(), cx)?;
        if let Some(parent_thread_id) = thread.parent_thread_id() {
            record_pause(&parent_thread_id.to_string(), &pause, cx);
        }
        Some(pause)
    })?;
    let Some(pause) = pause else {
        return Ok(false);
    };
    add_hook_message(
        thread,
        event_stream,
        None,
        format!("Paused by a hook: {pause}"),
        cx,
    )?;
    Ok(true)
}

/// Prefix of the ids of the messages hooks add to a conversation, so that
/// they are told apart from the user's own messages, even once reopened.
const HOOK_MESSAGE_ID_PREFIX: &str = "zed-hook-";

pub(crate) fn is_hook_message(id: &ClientUserMessageId) -> bool {
    serde_json::to_value(id)
        .ok()
        .and_then(|id| id.as_str().map(|id| id.starts_with(HOOK_MESSAGE_ID_PREFIX)))
        .unwrap_or(false)
}

/// Adds a message from a hook to the conversation, as a user message shown
/// the same way live and once the conversation is reopened. With `id`, it
/// replaces that message.
fn add_hook_message(
    thread: &WeakEntity<Thread>,
    event_stream: &ThreadEventStream,
    id: Option<ClientUserMessageId>,
    text: String,
    cx: &mut AsyncApp,
) -> Result<()> {
    let id = match id {
        Some(id) => id,
        None => {
            serde_json::from_value(json!(format!("{HOOK_MESSAGE_ID_PREFIX}{}", Uuid::new_v4())))?
        }
    };
    let message = UserMessage {
        id,
        content: [UserMessageContent::Text(text)].into(),
    };
    thread.update(cx, |thread, cx| {
        thread.upsert_user_message(message.id.clone(), message.content.clone(), cx)
    })?;
    event_stream.send_user_message(&message);
    Ok(())
}

/// The last message each root conversation submitted to `UserPromptSubmit`,
/// with its text and the block it got, by thread id: a retry of the same
/// message must not run the hooks again, or a counting guard would count it
/// twice.
#[derive(Default)]
struct SubmittedPrompts(HashMap<String, (ClientUserMessageId, String, Option<String>)>);

impl Global for SubmittedPrompts {}

/// Runs a hook of a conversation event until it answers or the turn is
/// cancelled; cancelling drops its task, which kills its process. A failure,
/// or an output for another event, counts as no output.
async fn call_conversation_hook(
    hook: &Arc<dyn HookHandler>,
    event: &str,
    input: serde_json::Value,
    cancellation: Option<watch::Receiver<bool>>,
    cx: &AsyncApp,
) -> Option<serde_json::Value> {
    let task = cx.update(|cx| hook.call(input, cx));
    let result = futures::select_biased! {
        _ = wait_for_cancellation(cancellation).fuse() => return None,
        result = task.fuse() => result,
    };
    let output = match result {
        Ok(output) => output,
        Err(error) => {
            log::warn!("{event} hook failed, continuing: {error:#}");
            return None;
        }
    };
    if let Err(error) = check_event_name(event, &output) {
        log::warn!("{event} hook output ignored: {error:#}");
        return None;
    }
    if !output["decision"].is_null() && output["decision"] != "block" {
        log::warn!(
            "{event} hook `decision` must be \"block\", not {}; ignored",
            output["decision"]
        );
    }
    Some(output)
}

/// Starts a turn: forgets a pause asked during an earlier turn, then runs the
/// `UserPromptSubmit` hooks when the turn starts from a message the user
/// sent in a root conversation. A block, or a pause, replaces the message
/// with the reason and ends the turn; the hooks' context follows the message
/// as a message of its own. Returns whether the turn goes on.
pub(crate) async fn before_turn(
    thread: &WeakEntity<Thread>,
    event_stream: &ThreadEventStream,
    user_message_id: Option<&ClientUserMessageId>,
    mut cancellation: watch::Receiver<bool>,
    cx: &mut AsyncApp,
) -> Result<bool> {
    let submit = thread.update(cx, |thread, cx| {
        take_pause(&thread.id().to_string(), cx);
        let Some(Message::User(message)) = thread.last_message() else {
            return None;
        };
        // A retry starts from a message a hook added: the hooks already ran.
        if thread.is_subagent()
            || is_hook_message(&message.id)
            || Some(&message.id) != user_message_id
        {
            return None;
        }
        let hooks = with_configured_hooks(cx, |hooks| all_hooks(&hooks.user_prompt_submit))?;
        (!hooks.is_empty()).then(|| (hooks, HookContext::for_thread(thread, cx), message.clone()))
    })?;
    let Some((hooks, context, message)) = submit else {
        return Ok(true);
    };
    let prompt = prompt_text(&message.content);
    let submitted = cx.update(|cx| {
        cx.try_global::<SubmittedPrompts>()
            .and_then(|submitted| submitted.0.get(&context.thread_id).cloned())
    });
    let retried = match submitted {
        Some((id, text, block)) if id == message.id && text == prompt => Some(block),
        _ => None,
    };
    if let Some(block) = retried {
        // A retry of the same message: its block holds, and its context is
        // still in the conversation.
        if let Some(reason) = block {
            add_hook_message(
                thread,
                event_stream,
                Some(message.id),
                format!("Message blocked by a UserPromptSubmit hook: {reason}"),
                cx,
            )?;
            return Ok(false);
        }
        return Ok(true);
    }
    let input = context.input("UserPromptSubmit", json!({ "prompt": prompt }));
    let mut notes = Vec::new();
    let mut pause = None;
    let mut block = None;
    for hook in &hooks {
        let output = call_conversation_hook(
            hook,
            "UserPromptSubmit",
            input.clone(),
            Some(cancellation.clone()),
            cx,
        )
        .await;
        // A cancelled turn leaves the conversation as it is.
        if *cancellation.borrow() {
            return Ok(false);
        }
        let Some(output) = output else {
            continue;
        };
        log_unsupported_output_fields(&output, &["/systemMessage"]);
        // A pause lets the chain run to its end; a block stops it.
        if let Some(stop_reason) = requested_pause(&output) {
            pause = Some(stop_reason.to_string());
        }
        if output["decision"] == "block" {
            let reason = output["reason"]
                .as_str()
                .filter(|reason| !reason.trim().is_empty())
                .unwrap_or("no reason given");
            block = Some(reason.to_string());
            break;
        }
        match hook_text(&output, "/hookSpecificOutput/additionalContext") {
            Ok(note) => notes.extend(note.map(str::to_string)),
            Err(error) => log::warn!("UserPromptSubmit hook context ignored: {error:#}"),
        }
    }
    let block = block.or(pause);
    cx.update(|cx| {
        cx.default_global::<SubmittedPrompts>().0.insert(
            context.thread_id.clone(),
            (message.id.clone(), prompt, block.clone()),
        )
    });
    if let Some(reason) = block {
        // The message stays, replaced by the block, so that it shows the same
        // live and once reopened; the model never sees its text.
        add_hook_message(
            thread,
            event_stream,
            Some(message.id),
            format!("Message blocked by a UserPromptSubmit hook: {reason}"),
            cx,
        )?;
        return Ok(false);
    }
    if !notes.is_empty() {
        let text = notes
            .iter()
            .map(|note| {
                format!(
                    "<system-reminder>\nUserPromptSubmit hook additional context: {}\n</system-reminder>",
                    escape_reminder_tags(note)
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        add_hook_message(thread, event_stream, None, text, cx)?;
    }
    Ok(true)
}

/// The text the user typed, with mentions as links.
fn prompt_text(content: &[UserMessageContent]) -> String {
    content
        .iter()
        .filter_map(|part| match part {
            UserMessageContent::Text(text) => Some(text.clone()),
            UserMessageContent::Mention { uri, .. } => Some(uri.as_link().to_string()),
            UserMessageContent::Image(_) => None,
        })
        .collect()
}

/// Runs the `Stop` hooks when a turn of a root conversation ends normally.
/// Returns whether one continued the turn, with its reason as a message; a
/// pause ends the turn with its own message instead.
pub(crate) async fn continue_after_stop(
    thread: &WeakEntity<Thread>,
    event_stream: &ThreadEventStream,
    stop_hook_active: &mut bool,
    mut cancellation: watch::Receiver<bool>,
    cx: &mut AsyncApp,
) -> Result<bool> {
    let stop = thread.update(cx, |thread, cx| {
        if thread.is_subagent() {
            return None;
        }
        let hooks = with_configured_hooks(cx, |hooks| all_hooks(&hooks.stop))?;
        (!hooks.is_empty()).then(|| {
            let last_message = match thread.last_message() {
                Some(Message::Agent(message)) => message
                    .content
                    .iter()
                    .filter_map(|content| match content {
                        AgentMessageContent::Text(text) => Some(text.as_str()),
                        _ => None,
                    })
                    .collect(),
                _ => String::new(),
            };
            (hooks, HookContext::for_thread(thread, cx), last_message)
        })
    })?;
    let Some((hooks, context, last_assistant_message)) = stop else {
        return Ok(false);
    };
    let mut pause = None;
    let mut feedback = None;
    for hook in &hooks {
        let input = context.input(
            "Stop",
            json!({
                "stop_hook_active": *stop_hook_active,
                "last_assistant_message": last_assistant_message,
            }),
        );
        let output =
            call_conversation_hook(hook, "Stop", input, Some(cancellation.clone()), cx).await;
        if *cancellation.borrow() {
            return Ok(false);
        }
        let Some(output) = output else {
            continue;
        };
        log_unsupported_output_fields(
            &output,
            &["/systemMessage", "/hookSpecificOutput/additionalContext"],
        );
        // The chain runs to its end, then the pause applies.
        if let Some(stop_reason) = requested_pause(&output) {
            pause = Some(stop_reason.to_string());
            continue;
        }
        if output["decision"] != "block" {
            continue;
        }
        match hook_text(&output, "/reason") {
            Ok(Some(reason)) if !reason.trim().is_empty() => {
                feedback = Some(reason.to_string());
                break;
            }
            _ => log::warn!("a Stop hook continued the turn without a reason; ignored"),
        }
    }
    if let Some(pause) = pause {
        add_hook_message(
            thread,
            event_stream,
            None,
            format!("Paused by a hook: {pause}"),
            cx,
        )?;
        return Ok(false);
    }
    let Some(feedback) = feedback else {
        return Ok(false);
    };
    add_hook_message(
        thread,
        event_stream,
        None,
        format!("Stop hook feedback: {feedback}"),
        cx,
    )?;
    *stop_hook_active = true;
    Ok(true)
}

/// Runs the `SessionStart` hooks when Zed registers a root conversation: a
/// new one (`startup`), or one with history (`resume`). A subagent's hooks
/// would get its parent's `session_id`.
pub(crate) fn session_start(thread: &Entity<Thread>, cx: &mut App) {
    let thread = thread.read(cx);
    if thread.is_subagent() {
        return;
    }
    let source = if thread.last_message().is_none() {
        "startup"
    } else {
        "resume"
    };
    let Some(hooks) =
        with_configured_hooks(cx, |hooks| matching_hooks(&hooks.session_start, source))
    else {
        return;
    };
    let mut fields = json!({ "source": source });
    if let Some(model) = thread.model() {
        fields["model"] = json!(model.id().0.to_string());
    }
    let input = HookContext::for_thread(thread, cx).input("SessionStart", fields);
    notify_hooks("SessionStart", hooks, input, cx);
}

/// Forgets a pause the conversation never reached and its last prompt, then
/// runs the
/// `SessionEnd` hooks when Zed releases a root conversation. gpui does not
/// say why, so the reason is always `other`.
pub(crate) fn session_end(thread: &Entity<Thread>, cx: &mut App) {
    let (thread_id, is_subagent) = thread.read_with(cx, |thread, _| {
        (thread.id().to_string(), thread.is_subagent())
    });
    take_pause(&thread_id, cx);
    if cx.has_global::<SubmittedPrompts>() {
        cx.global_mut::<SubmittedPrompts>().0.remove(&thread_id);
    }
    if is_subagent {
        return;
    }
    let Some(hooks) =
        with_configured_hooks(cx, |hooks| matching_hooks(&hooks.session_end, "other"))
    else {
        return;
    };
    let input = HookContext::for_thread(thread.read(cx), cx)
        .input("SessionEnd", json!({ "reason": "other" }));
    notify_hooks("SessionEnd", hooks, input, cx);
}

/// Runs the hooks of a notification event in the background, in declaration
/// order. Their output decides nothing: a code 2 is logged, until Zed can
/// show it to the user.
fn notify_hooks(
    event: &'static str,
    hooks: Vec<Arc<dyn HookHandler>>,
    input: serde_json::Value,
    cx: &mut App,
) {
    if hooks.is_empty() {
        return;
    }
    cx.spawn(async move |cx| {
        for hook in &hooks {
            let Some(output) = call_conversation_hook(hook, event, input.clone(), None, cx).await
            else {
                continue;
            };
            if output["decision"] == "block" {
                log::warn!("{event} hook: {}", output["reason"].as_str().unwrap_or(""));
            }
            log_unsupported_output_fields(
                &output,
                &[
                    "/continue",
                    "/systemMessage",
                    "/hookSpecificOutput/additionalContext",
                ],
            );
        }
    })
    .detach();
}

/// Wraps the environment of a thread so that the subagents it creates run
/// the `SubagentStart` and `SubagentStop` hooks.
pub(crate) fn wrap_environment(
    thread: &Thread,
    environment: Rc<dyn ThreadEnvironment>,
    cx: &App,
) -> Rc<dyn ThreadEnvironment> {
    Rc::new(HookedEnvironment {
        inner: environment,
        // The hooks of a subagent fire in its parent: `zed.depth` and
        // `zed.thread_id` are the parent's; `agent_id` is the subagent's.
        context: HookContext::for_thread(thread, cx),
        subagents: Default::default(),
    })
}

struct HookedEnvironment {
    inner: Rc<dyn ThreadEnvironment>,
    context: HookContext,
    /// By subagent id, what this environment knows of each subagent.
    subagents: Rc<RefCell<HashMap<String, SubagentState>>>,
}

#[derive(Clone)]
struct SubagentState {
    label: String,
}

impl HookedEnvironment {
    /// Wraps a subagent in the hooks its label matches, if any.
    fn hooked_subagent(
        &self,
        subagent: Rc<dyn SubagentHandle>,
        label: String,
        cx: &App,
    ) -> Rc<dyn SubagentHandle> {
        let Some((start, stop)) = with_configured_hooks(cx, |hooks| {
            (
                matching_hooks(&hooks.subagent_start, &label),
                matching_hooks(&hooks.subagent_stop, &label),
            )
        }) else {
            return subagent;
        };
        if start.is_empty() && stop.is_empty() {
            return subagent;
        }
        Rc::new(HookedSubagent {
            inner: subagent,
            label,
            context: self.context.clone(),
            start,
            stop,
        })
    }
}

impl ThreadEnvironment for HookedEnvironment {
    fn create_terminal(
        &self,
        command: String,
        extra_env: Vec<(String, String)>,
        cwd: Option<PathBuf>,
        output_byte_limit: Option<u64>,
        sandbox_wrap: Option<acp_thread::SandboxWrap>,
        cx: &mut AsyncApp,
    ) -> Task<Result<Rc<dyn TerminalHandle>>> {
        self.inner
            .create_terminal(command, extra_env, cwd, output_byte_limit, sandbox_wrap, cx)
    }

    fn create_subagent(
        &self,
        label: String,
        model: Option<AgentModelId>,
        cx: &mut App,
    ) -> Result<Rc<dyn SubagentHandle>> {
        let subagent = self.inner.create_subagent(label.clone(), model, cx)?;
        self.subagents.borrow_mut().insert(
            subagent.id().to_string(),
            SubagentState {
                label: label.clone(),
            },
        );
        Ok(self.hooked_subagent(subagent, label, cx))
    }

    fn resume_subagent(
        &self,
        session_id: acp_v2::SessionId,
        cx: &mut App,
    ) -> Result<Rc<dyn SubagentHandle>> {
        let subagent = self.inner.resume_subagent(session_id, cx)?;
        let id = subagent.id().to_string();
        // After the parent is reopened, this environment never saw the
        // subagent: its label is the title of its thread.
        let label = self
            .subagents
            .borrow_mut()
            .entry(id.clone())
            .or_insert_with(|| SubagentState {
                label: subagent
                    .label(cx)
                    .map(|label| label.to_string())
                    .unwrap_or_else(|| id.clone()),
            })
            .label
            .clone();
        Ok(self.hooked_subagent(subagent, label, cx))
    }

    fn create_sibling_thread(
        &self,
        request: crate::SiblingThreadRequest,
        cx: &mut AsyncApp,
    ) -> Task<Result<crate::SiblingThreadInfo>> {
        self.inner.create_sibling_thread(request, cx)
    }

    fn list_available_agents(&self, cx: &mut App) -> Result<crate::AvailableAgents> {
        self.inner.list_available_agents(cx)
    }
}

/// A subagent whose turns run the `SubagentStart` hooks until it has ever
/// run, and the `SubagentStop` hooks after each turn. A refused subagent
/// never got a message: resuming it runs `SubagentStart` again, so that the
/// refusal holds. Dropping a turn's task,
/// as stopping the parent does, abandons its hooks and the turns that would
/// follow.
struct HookedSubagent {
    inner: Rc<dyn SubagentHandle>,
    label: String,
    context: HookContext,
    start: Vec<Arc<dyn HookHandler>>,
    stop: Vec<Arc<dyn HookHandler>>,
}

impl SubagentHandle for HookedSubagent {
    fn id(&self) -> acp_v2::SessionId {
        self.inner.id()
    }

    fn num_entries(&self, cx: &App) -> usize {
        self.inner.num_entries(cx)
    }

    fn label(&self, cx: &App) -> Option<SharedString> {
        self.inner.label(cx)
    }

    fn send(&self, message: String, cx: &AsyncApp) -> Task<Result<String>> {
        let inner = self.inner.clone();
        let agent_id = self.inner.id().to_string();
        let label = self.label.clone();
        let context = self.context.clone();
        let start = self.start.clone();
        let stop = self.stop.clone();
        cx.spawn(async move |cx| {
            let mut message = message;
            // A subagent with entries already ran, before a reopen or in a
            // concurrent turn.
            let started = cx.update(|cx| inner.num_entries(cx)) > 0;
            if !started && !start.is_empty() {
                let notes = run_subagent_start(&start, &context, &agent_id, &label, cx).await?;
                for note in notes {
                    message.push_str(&format!(
                        "\n\n<system-reminder>\n{}\n</system-reminder>",
                        escape_reminder_tags(&note)
                    ));
                }
            }
            // A stop hook can continue the subagent with its reason, as many
            // times as it asks; `stop_hook_active` tells it it already did.
            let mut stop_hook_active = false;
            let mut first_answer = None;
            loop {
                let output = match inner.send(message, cx).await {
                    Ok(output) => output,
                    // A continuation failed: the parent still gets the answer the
                    // subagent gave first.
                    Err(error) => {
                        return Err(match first_answer {
                            Some(first_answer) => anyhow!(
                                "{error:#}\n\nThe subagent answered before a SubagentStop \
                                 hook asked it to continue:\n\n{first_answer}"
                            ),
                            None => error,
                        });
                    }
                };
                first_answer.get_or_insert_with(|| output.clone());
                // A paused subagent passed its pause on: the parent waits for
                // the user rather than continue it.
                if cx.update(|cx| pause_pending(&context.thread_id, cx)) {
                    return Ok(output);
                }
                let continuation = run_subagent_stop(
                    &stop,
                    &context,
                    &agent_id,
                    &label,
                    &output,
                    stop_hook_active,
                    cx,
                )
                .await;
                let Some(continuation) = continuation else {
                    return Ok(output);
                };
                message = continuation;
                stop_hook_active = true;
            }
        })
    }
}

/// Runs the `SubagentStart` hooks in declaration order. Returns the texts to
/// add to the subagent's first message, or the first refusal as an error:
/// the parent model gets it as the `spawn_agent` result.
async fn run_subagent_start(
    hooks: &[Arc<dyn HookHandler>],
    context: &HookContext,
    agent_id: &str,
    label: &str,
    cx: &AsyncApp,
) -> Result<Vec<String>> {
    let mut notes = Vec::new();
    for hook in hooks {
        let input = context.input(
            "SubagentStart",
            json!({ "agent_id": agent_id, "agent_type": label }),
        );
        let output = match cx.update(|cx| hook.call(input, cx)).await {
            Ok(output) => output,
            Err(error) => {
                log::warn!("SubagentStart hook failed, continuing: {error:#}");
                continue;
            }
        };
        if let Err(error) = check_event_name("SubagentStart", &output) {
            log::warn!("SubagentStart hook output ignored: {error:#}");
            continue;
        }
        // A code 2 arrives as a refusal: Claude Code shows its stderr to the
        // user only.
        if output.get("decision").is_some_and(|value| !value.is_null()) {
            log::warn!(
                "a SubagentStart hook has no decision to make; ignored (reason: {})",
                output["reason"].as_str().unwrap_or("none given")
            );
        }
        warn_on_unknown_zed_fields("SubagentStart", &output, &["deny_reason", "model"]);
        log_unsupported_output_fields(
            &output,
            &[
                "/continue",
                "/systemMessage",
                "/stopReason",
                // The model is chosen when Zed creates the subagent, before
                // the hooks run.
                "/hookSpecificOutput/zed/model",
            ],
        );
        let fields = hook_text(&output, "/hookSpecificOutput/zed/deny_reason").and_then(|deny| {
            Ok((
                deny,
                hook_text(&output, "/hookSpecificOutput/additionalContext")?,
            ))
        });
        match fields {
            Ok((Some(reason), _)) => return Err(anyhow!(reason.to_string())),
            Ok((None, note)) => notes.extend(note.map(str::to_string)),
            Err(error) => {
                log::warn!("SubagentStart hook output ignored, the subagent runs: {error:#}")
            }
        }
    }
    Ok(notes)
}

/// Runs the `SubagentStop` hooks in declaration order once a subagent turn
/// ended. Returns the reason of the first one that continues the turn.
async fn run_subagent_stop(
    hooks: &[Arc<dyn HookHandler>],
    context: &HookContext,
    agent_id: &str,
    label: &str,
    last_assistant_message: &str,
    stop_hook_active: bool,
    cx: &AsyncApp,
) -> Option<String> {
    for hook in hooks {
        let input = context.input(
            "SubagentStop",
            json!({
                "stop_hook_active": stop_hook_active,
                "agent_id": agent_id,
                "agent_type": label,
                "agent_transcript_path": null,
                "last_assistant_message": last_assistant_message,
            }),
        );
        let output = match cx.update(|cx| hook.call(input, cx)).await {
            Ok(output) => output,
            Err(error) => {
                log::warn!("SubagentStop hook failed, continuing: {error:#}");
                continue;
            }
        };
        if let Err(error) = check_event_name("SubagentStop", &output) {
            log::warn!("SubagentStop hook output ignored: {error:#}");
            continue;
        }
        log_unsupported_output_fields(
            &output,
            &[
                "/continue",
                "/systemMessage",
                "/stopReason",
                "/hookSpecificOutput/additionalContext",
            ],
        );
        if output["decision"] == "block" {
            match hook_text(&output, "/reason") {
                Ok(Some(reason)) => return Some(reason.to_string()),
                _ => log::warn!("a SubagentStop hook continued the turn without a reason; ignored"),
            }
        }
    }
    None
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
            block_on_failure: false,
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
            block_on_failure: false,
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

    const CONTRACT_SCHEMAS: [(&str, &str); 30] = [
        (
            "Stop.input.schema.json",
            include_str!("../hook_contract/Stop.input.schema.json"),
        ),
        (
            "Stop.output.schema.json",
            include_str!("../hook_contract/Stop.output.schema.json"),
        ),
        (
            "UserPromptSubmit.input.schema.json",
            include_str!("../hook_contract/UserPromptSubmit.input.schema.json"),
        ),
        (
            "UserPromptSubmit.output.schema.json",
            include_str!("../hook_contract/UserPromptSubmit.output.schema.json"),
        ),
        (
            "SessionStart.input.schema.json",
            include_str!("../hook_contract/SessionStart.input.schema.json"),
        ),
        (
            "SessionStart.output.schema.json",
            include_str!("../hook_contract/SessionStart.output.schema.json"),
        ),
        (
            "SessionEnd.input.schema.json",
            include_str!("../hook_contract/SessionEnd.input.schema.json"),
        ),
        (
            "SessionEnd.output.schema.json",
            include_str!("../hook_contract/SessionEnd.output.schema.json"),
        ),
        (
            "SubagentStart.input.schema.json",
            include_str!("../hook_contract/SubagentStart.input.schema.json"),
        ),
        (
            "SubagentStart.output.schema.json",
            include_str!("../hook_contract/SubagentStart.output.schema.json"),
        ),
        (
            "SubagentStop.input.schema.json",
            include_str!("../hook_contract/SubagentStop.input.schema.json"),
        ),
        (
            "SubagentStop.output.schema.json",
            include_str!("../hook_contract/SubagentStop.output.schema.json"),
        ),
        (
            "ZedModelError.input.schema.json",
            include_str!("../hook_contract/ZedModelError.input.schema.json"),
        ),
        (
            "ZedModelError.output.schema.json",
            include_str!("../hook_contract/ZedModelError.output.schema.json"),
        ),
        (
            "PreCompact.input.schema.json",
            include_str!("../hook_contract/PreCompact.input.schema.json"),
        ),
        (
            "PreCompact.output.schema.json",
            include_str!("../hook_contract/PreCompact.output.schema.json"),
        ),
        (
            "PostCompact.input.schema.json",
            include_str!("../hook_contract/PostCompact.input.schema.json"),
        ),
        (
            "PostCompact.output.schema.json",
            include_str!("../hook_contract/PostCompact.output.schema.json"),
        ),
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

    /// Runs the tool, which a hook makes ask the user, and answers with the
    /// option `answer`; returns whether the model sees an error, the text it
    /// sees, and the message the user was shown.
    async fn run_echo_confirming(
        tool: Arc<HookedTool>,
        answer: &str,
        cx: &mut TestAppContext,
    ) -> (bool, Vec<String>, String) {
        let (event_stream, mut receiver) = ToolCallEventStream::test();
        let task = cx.update(|cx| {
            tool.run(
                ToolInput::ready(json!({ "text": "hello" })),
                event_stream,
                cx,
            )
        });
        let authorization = receiver.expect_authorization().await;
        let message = format!("{:?}", authorization.tool_call.content);
        let kind = match answer {
            "allow" => acp_v2::PermissionOptionKind::AllowOnce,
            _ => acp_v2::PermissionOptionKind::RejectOnce,
        };
        authorization
            .response
            .send(acp_thread::SelectedPermissionOutcome::new(
                acp_v2::PermissionOptionId::new(answer),
                kind,
            ))
            .ok();
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
        (is_error, texts, message)
    }

    fn asking_hook() -> Arc<dyn HookHandler> {
        fake_hook(Ok(pre_tool_use(json!({
            "permissionDecision": "ask",
            "permissionDecisionReason": "Pushing to main.",
        }))))
        .0
    }

    #[gpui::test]
    async fn test_pre_tool_use_ask_runs_the_tool_the_user_allows(cx: &mut TestAppContext) {
        let (tool, echo) = hooked_echo(vec![asking_hook()], Vec::new());

        let (is_error, texts, message) = run_echo_confirming(tool, "allow", cx).await;

        assert!(message.contains("Pushing to main."), "{message}");
        assert!(!is_error);
        assert_eq!(texts, vec!["hello".to_string()]);
        assert_eq!(echo.runs.get(), 1);
    }

    #[gpui::test]
    async fn test_pre_tool_use_ask_refuses_the_tool_the_user_denies(cx: &mut TestAppContext) {
        let (tool, echo) = hooked_echo(vec![asking_hook()], Vec::new());

        let (is_error, texts, _) = run_echo_confirming(tool, "deny", cx).await;

        assert!(is_error);
        assert!(texts[0].contains("The user refused"), "{texts:?}");
        assert_eq!(echo.runs.get(), 0);
    }

    #[gpui::test]
    async fn test_pre_tool_use_ask_lets_a_later_denial_win(cx: &mut TestAppContext) {
        let (deny, _) = fake_hook(Ok(pre_tool_use(json!({
            "permissionDecision": "deny",
            "permissionDecisionReason": "Never on main.",
        }))));
        let (tool, echo) = hooked_echo(vec![asking_hook(), deny], Vec::new());

        // No question: the run would wait for an answer that never comes.
        let (is_error, texts) = run_echo(tool, "hello", cx).await;

        assert!(is_error);
        assert_eq!(texts, vec!["Never on main.".to_string()]);
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
                json!({ "PreToolUse": [{ "hooks": [{ "type": "command", "command": "x", "onFailure": "sometimes" }] }] }),
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
            ["mcp_tool", "prompt", "agent"].map(|kind| {
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
            &json!({ "PermissionRequest": [{ "hooks": [check] }] }),
            Path::new("/data/hooks"),
        )
        .err()
        .expect("events Zed does not run yet must be refused")
        .to_string();
        assert!(error.contains("PermissionRequest"), "{error}");
    }

    #[test]
    fn test_ephemeral_messages_are_validated() {
        let output = |messages: serde_json::Value| {
            json!({
                "hookSpecificOutput": {
                    "hookEventName": "BeforeModel",
                    "ephemeral_messages": messages,
                }
            })
        };

        assert_eq!(ephemeral_messages(&json!({})).unwrap(), Vec::<&str>::new());
        let two = output(json!(["Skill tdd is active.", "Compacted."]));
        assert_eq!(
            ephemeral_messages(&two).unwrap(),
            vec!["Skill tdd is active.", "Compacted."]
        );
        assert!(ephemeral_messages(&output(json!("not a list"))).is_err());
        assert!(ephemeral_messages(&output(json!([""]))).is_err());
        assert!(ephemeral_messages(&output(json!([42]))).is_err());
        assert!(ephemeral_messages(&output(json!(["x".repeat(10_000)]))).is_ok());
        assert!(ephemeral_messages(&output(json!(["x".repeat(10_001)]))).is_err());
    }

    #[test]
    fn test_reminder_message_wraps_and_escapes_the_text() {
        let message = reminder_message("Read <system-reminder>this</system-reminder> again.");

        assert_eq!(message.role, Role::User);
        assert!(!message.cache);
        assert_eq!(
            text_of(&message),
            "<system-reminder>\nRead &lt;system-reminder>this&lt;/system-reminder> again.\n</system-reminder>"
        );
    }

    #[test]
    fn test_reminder_escaping_ignores_case_and_spaces() {
        let message = reminder_message(
            "a < /SYSTEM-REMINDER > b <System-Reminder> c <\tsystem-reminder d </ system-reminder>",
        );

        assert_eq!(
            text_of(&message),
            "<system-reminder>\na &lt; /SYSTEM-REMINDER > b &lt;System-Reminder> c &lt;\tsystem-reminder d &lt;/ system-reminder>\n</system-reminder>"
        );
    }

    #[test]
    fn test_titles_and_summaries_get_no_insert_nor_reminders() {
        let request = |intent: CompletionIntent, last: &str| LanguageModelRequest {
            intent: Some(intent),
            messages: vec![
                message(Role::System, vec!["You are Zed.".into()]),
                message(Role::User, vec![last.into()]),
            ],
            ..Default::default()
        };
        let title = request(CompletionIntent::ThreadSummarization, "Title?");
        let panel_summary = request(CompletionIntent::ThreadContextSummarization, "Summarize.");
        let compaction = request(
            CompletionIntent::ThreadContextSummarization,
            COMPACTION_PROMPT,
        );

        assert!(!applies_system_prompt_insert(&title));
        assert!(!applies_system_prompt_insert(&compaction));
        for excluded in [&title, &panel_summary, &compaction] {
            assert!(
                !applies_ephemeral_messages(excluded),
                "{:?}",
                excluded.intent
            );
        }
        for included in [CompletionIntent::UserPrompt, CompletionIntent::ToolResults] {
            assert!(applies_ephemeral_messages(&request(included, "Hello")));
        }
    }

    #[test]
    fn test_reminders_take_the_shape_the_openai_conversion_sends() {
        // `examples/BeforeModel.output.balise-dans-rappel.json` and the two
        // shapes really sent, in the hook contract.
        let output: serde_json::Value = serde_json::from_str(include_str!(
            "../hook_contract/examples/BeforeModel.output.balise-dans-rappel.json"
        ))
        .unwrap();
        let sent: serde_json::Value = serde_json::from_str(include_str!(
            "../hook_contract/examples/BeforeModel.output.balise-dans-rappel.messages-envoyes.txt"
        ))
        .unwrap();
        let reminders = ephemeral_messages(&output)
            .unwrap()
            .into_iter()
            .map(reminder_message)
            .collect::<Vec<_>>();
        let convert = |mut messages: Vec<LanguageModelRequestMessage>| {
            messages.extend(reminders.iter().cloned());
            let request = open_ai::completion::into_open_ai(
                LanguageModelRequest {
                    messages,
                    ..Default::default()
                },
                "qwen",
                true,
                false,
                None,
                open_ai::completion::ChatCompletionMaxTokensParameter::MaxTokens,
                None,
                false,
            )
            .unwrap();
            serde_json::to_value(request.messages).unwrap()
        };

        // After the user's message, the reminders become parts of it.
        let after_prompt = convert(vec![
            message(Role::System, vec!["You are Zed.".into()]),
            message(Role::User, vec!["Corrige le test rouge.".into()]),
        ]);
        assert_eq!(
            after_prompt.as_array().unwrap()[1..],
            sent["apres_message_utilisateur"].as_array().unwrap()[..]
        );

        // After tool results, the first reminder opens a user message.
        let after_tools = convert(vec![
            message(Role::System, vec!["You are Zed.".into()]),
            message(Role::User, vec!["Où est le délai ?".into()]),
            message(
                Role::Assistant,
                vec![MessageContent::ToolUse(LanguageModelToolUse {
                    id: "call_9d0f".into(),
                    name: "grep".into(),
                    raw_input: r#"{"regex":"timeout"}"#.into(),
                    input: LanguageModelToolUseInput::Json(json!({ "regex": "timeout" })),
                    is_input_complete: true,
                    thought_signature: None,
                })],
            ),
            message(
                Role::User,
                vec![MessageContent::ToolResult(LanguageModelToolResult {
                    tool_use_id: "call_9d0f".into(),
                    tool_name: "grep".into(),
                    is_error: false,
                    content: vec!["crates/agent/src/thread.rs:3409 ...".into()],
                    output: None,
                })],
            ),
        ]);
        let after_tools = after_tools.as_array().unwrap();
        assert_eq!(
            after_tools[after_tools.len() - 2..],
            sent["apres_resultats_outil"].as_array().unwrap()[..]
        );
    }

    #[test]
    fn test_compaction_hooks_are_read_and_match_their_trigger() {
        let check = json!({ "type": "command", "command": "check" });
        let hooks = AgentHooks::from_settings(
            &json!({
                "PreCompact": [{ "matcher": "manual", "hooks": [check.clone()] }],
                "PostCompact": [{ "hooks": [check] }],
            }),
            Path::new("/data/hooks"),
        )
        .unwrap();

        assert_eq!(matching_hooks(&hooks.pre_compact, "manual").len(), 1);
        assert_eq!(matching_hooks(&hooks.pre_compact, "auto").len(), 0);
        assert_eq!(matching_hooks(&hooks.post_compact, "auto").len(), 1);
    }

    /// A provider stream that panics when polled again after its end, as
    /// Ollama's does.
    struct PanicsAfterEnd {
        events: std::collections::VecDeque<
            Result<LanguageModelCompletionEvent, LanguageModelCompletionError>,
        >,
        ended: bool,
    }

    impl futures::Stream for PanicsAfterEnd {
        type Item = Result<LanguageModelCompletionEvent, LanguageModelCompletionError>;

        fn poll_next(
            mut self: std::pin::Pin<&mut Self>,
            _cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Option<Self::Item>> {
            assert!(!self.ended, "the provider stream was polled after its end");
            let next = self.events.pop_front();
            self.ended = next.is_none();
            std::task::Poll::Ready(next)
        }
    }

    #[gpui::test]
    async fn test_post_compact_never_polls_the_summary_after_its_end(cx: &mut TestAppContext) {
        let (post, _) = fake_hook(Ok(json!({
            "hookSpecificOutput": {
                "hookEventName": "PostCompact",
                "zed": { "append_to_summary": "Plan." },
            }
        })));
        let summary = PanicsAfterEnd {
            events: vec![Ok(LanguageModelCompletionEvent::Text("summary".into()))].into(),
            ended: false,
        }
        .boxed();
        let context = HookContext {
            session_id: "session-1".into(),
            cwd: "/project".into(),
            thread_id: "session-1".into(),
            parent_thread_id: None,
            depth: 0,
        };

        let stream = append_post_compact(
            summary,
            vec![post],
            context,
            CompactionTrigger::Auto,
            &cx.to_async(),
        );
        let texts = stream
            .filter_map(|event| async move {
                match event {
                    Ok(LanguageModelCompletionEvent::Text(text)) => Some(text),
                    _ => None,
                }
            })
            .collect::<Vec<_>>()
            .await;

        assert_eq!(texts, ["summary", "\n\nPlan."]);
    }

    #[test]
    fn test_zed_model_error_is_read_and_zed_count_tokens_refused() {
        let check = json!({ "type": "command", "command": "check" });
        let hooks = AgentHooks::from_settings(
            &json!({ "ZedModelError": [{ "matcher": "user_prompt", "hooks": [check.clone()] }] }),
            Path::new("/data/hooks"),
        )
        .unwrap();
        assert_eq!(
            matching_hooks(&hooks.zed_model_error, "user_prompt").len(),
            1
        );

        let error = AgentHooks::from_settings(
            &json!({ "ZedCountTokens": [{ "hooks": [check] }] }),
            Path::new("/data/hooks"),
        )
        .err()
        .expect("ZedCountTokens is not run yet")
        .to_string();
        assert!(error.contains("ZedCountTokens"), "{error}");
    }

    #[test]
    fn test_error_categories_round_trip_through_their_names() {
        let categories = [
            ProviderErrorCategory::PromptTooLarge {
                tokens: Some(130_000),
            },
            ProviderErrorCategory::InvalidEncryptedContent,
            ProviderErrorCategory::Authentication,
            ProviderErrorCategory::Permission,
            ProviderErrorCategory::ContentPolicy,
            ProviderErrorCategory::EndpointNotFound,
            ProviderErrorCategory::PaymentRequired,
            ProviderErrorCategory::RateLimit,
            ProviderErrorCategory::Overloaded,
            ProviderErrorCategory::InvalidRequest,
            ProviderErrorCategory::Conflict,
            ProviderErrorCategory::Timeout,
            ProviderErrorCategory::InternalServer,
            ProviderErrorCategory::Other,
        ];
        for category in categories {
            let name = category_name(&category);
            let tokens = match category {
                ProviderErrorCategory::PromptTooLarge { tokens } => tokens,
                _ => None,
            };
            assert_eq!(parse_category(name, tokens), Some(category), "{name}");
        }
        assert_eq!(parse_category("not_a_category", None), None);
    }

    fn too_long_rejection() -> LanguageModelCompletionError {
        LanguageModelCompletionError::from_http_status(
            language_model::LanguageModelProviderName::new("vllm"),
            http_client::StatusCode::BAD_REQUEST,
            "Input is too long for this model.".to_string(),
            None,
        )
    }

    fn error_hooks(hooks: Vec<Arc<dyn HookHandler>>) -> ErrorHooks {
        ErrorHooks {
            context: HookContext {
                session_id: "session-1".into(),
                cwd: "/project".into(),
                thread_id: "session-1".into(),
                parent_thread_id: None,
                depth: 0,
            },
            hooks,
            request_fields: json!({
                "request_id": "6f1c2a7e-0b3d-4e5f-8a9b-1c2d3e4f5a6b",
                "intent": "user_prompt",
                "model": { "provider": "openai_compatible", "id": "qwen" },
                "is_compaction": false,
            }),
        }
    }

    fn category_of(error: &LanguageModelCompletionError) -> Option<ProviderErrorCategory> {
        match error {
            LanguageModelCompletionError::ProviderRejection { category, .. } => {
                Some(category.clone())
            }
            _ => None,
        }
    }

    fn to_full_window() -> (Arc<dyn HookHandler>, Rc<RefCell<Vec<serde_json::Value>>>) {
        fake_hook(Ok(json!({
            "hookSpecificOutput": { "hookEventName": "ZedModelError", "category": "prompt_too_large" }
        })))
    }

    #[test]
    fn test_a_vllm_400_with_an_integer_code_is_a_provider_rejection() {
        let body = json!({
            "object": "error",
            "message": "This model's maximum context length is 122880 tokens. However, you requested 122880 output tokens and your prompt contains 302 characters (more than 0 characters, which is the upper bound for 0 input tokens). Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_text, value=302)",
            "type": "BadRequestError",
            "param": "input_text",
            "code": 400,
        });
        let error = LanguageModelCompletionError::from(open_ai::RequestError::HttpResponseError {
            provider: "vllm".into(),
            status_code: http_client::StatusCode::BAD_REQUEST,
            body: body.to_string(),
            headers: Box::default(),
        });

        assert!(
            matches!(
                &error,
                LanguageModelCompletionError::ProviderRejection { status: Some(status), .. }
                    if status.as_u16() == 400
            ),
            "{error:?}"
        );
    }

    #[gpui::test]
    async fn test_translate_changes_only_a_provider_rejection(cx: &mut TestAppContext) {
        let (hook, calls) = to_full_window();
        let hooks = error_hooks(vec![hook]);
        let cx = cx.to_async();

        let translated = hooks.translate(too_long_rejection(), None, &cx).await;
        assert_eq!(
            category_of(&translated),
            Some(ProviderErrorCategory::PromptTooLarge { tokens: None })
        );

        let other = hooks
            .translate(anyhow!("connection reset").into(), None, &cx)
            .await;
        assert!(category_of(&other).is_none());
        assert_eq!(calls.borrow().len(), 1);
    }

    #[gpui::test]
    async fn test_translate_keeps_the_error_when_hooks_fail(cx: &mut TestAppContext) {
        let (failing, _) = fake_hook(Err("hook crashed".into()));
        let hooks = error_hooks(vec![failing]);

        let translated = hooks
            .translate(too_long_rejection(), None, &cx.to_async())
            .await;

        assert_eq!(
            category_of(&translated),
            Some(ProviderErrorCategory::InvalidRequest)
        );
    }

    #[gpui::test]
    async fn test_cancelling_abandons_the_error_hooks_at_send(cx: &mut TestAppContext) {
        let abandoned = Arc::new(AtomicBool::new(false));
        let pending: Arc<dyn HookHandler> = Arc::new(PendingHook {
            abandoned: abandoned.clone(),
        });
        let hooks = error_hooks(vec![pending]);
        let (mut cancellation_tx, cancellation_rx) = watch::channel(false);
        let task = cx.update(|cx| {
            cx.spawn(async move |cx| {
                hooks
                    .translate(too_long_rejection(), Some(cancellation_rx), cx)
                    .await
            })
        });
        cx.run_until_parked();

        cancellation_tx.send(true).ok();
        cx.run_until_parked();

        let mut task = task;
        let error = futures::FutureExt::now_or_never(&mut task)
            .expect("cancelling must stop the error hooks");
        assert_eq!(
            category_of(&error),
            Some(ProviderErrorCategory::InvalidRequest)
        );
        assert!(abandoned.load(Ordering::SeqCst));
    }

    #[gpui::test]
    async fn test_stream_errors_are_translated_after_text(cx: &mut TestAppContext) {
        let (hook, _) = to_full_window();
        let stream = futures::stream::iter(vec![
            Ok(LanguageModelCompletionEvent::Text("partial".into())),
            Err(too_long_rejection()),
        ])
        .boxed();

        let events = translate_stream_errors(stream, error_hooks(vec![hook]), &cx.to_async())
            .collect::<Vec<_>>()
            .await;

        assert!(matches!(
            &events[0],
            Ok(LanguageModelCompletionEvent::Text(text)) if text == "partial"
        ));
        assert_eq!(
            events[1].as_ref().err().and_then(category_of),
            Some(ProviderErrorCategory::PromptTooLarge { tokens: None })
        );
    }

    #[gpui::test]
    async fn test_stream_errors_other_than_rejections_pass_unchanged(cx: &mut TestAppContext) {
        let (hook, calls) = to_full_window();
        let stream = futures::stream::iter(vec![Err(anyhow!("connection reset").into())]).boxed();

        let events = translate_stream_errors(stream, error_hooks(vec![hook]), &cx.to_async())
            .collect::<Vec<_>>()
            .await;

        assert!(events[0].as_ref().err().and_then(category_of).is_none());
        assert!(calls.borrow().is_empty());
    }

    #[gpui::test]
    async fn test_dropping_the_stream_abandons_the_error_hooks(cx: &mut TestAppContext) {
        let abandoned = Arc::new(AtomicBool::new(false));
        let pending: Arc<dyn HookHandler> = Arc::new(PendingHook {
            abandoned: abandoned.clone(),
        });
        let stream = futures::stream::iter(vec![Err(too_long_rejection())]).boxed();
        let mut stream =
            translate_stream_errors(stream, error_hooks(vec![pending]), &cx.to_async());
        {
            let next = stream.next();
            futures::pin_mut!(next);
            assert!(futures::FutureExt::now_or_never(&mut next).is_none());
            cx.run_until_parked();
        }

        drop(stream);
        cx.run_until_parked();

        assert!(abandoned.load(Ordering::SeqCst));
    }

    #[test]
    fn test_an_absurd_usage_is_refused() {
        let output = |input_tokens: u64| {
            json!({
                "hookSpecificOutput": {
                    "hookEventName": "AfterModel",
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": 10,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                    },
                }
            })
        };

        assert!(hook_usage(&output(950_000)).unwrap().is_some());
        assert!(hook_usage(&output(u64::MAX - 1)).is_err());
        assert!(hook_usage(&output(1 << 53)).is_err());
    }

    /// A provider that rejects every request when it is sent, the way the
    /// OpenAI-compatible provider reports vLLM's 400.
    struct RejectingProvider;

    impl language_model::LanguageModelClient for RejectingProvider {
        fn stream_completion(
            &self,
            _model: &LanguageModel,
            _request: LanguageModelRequest,
            _cx: &AsyncApp,
        ) -> futures::future::BoxFuture<
            'static,
            Result<LanguageModelCompletionStream, LanguageModelCompletionError>,
        > {
            async { Err(too_long_rejection()) }.boxed()
        }
    }

    impl LanguageModelProvider for RejectingProvider {
        fn id(&self) -> LanguageModelProviderId {
            LanguageModelProviderId::new("rejecting")
        }

        fn name(&self) -> language_model::LanguageModelProviderName {
            language_model::LanguageModelProviderName::new("rejecting")
        }

        fn default_model(&self, _cx: &App) -> Option<LanguageModel> {
            None
        }

        fn default_fast_model(&self, _cx: &App) -> Option<LanguageModel> {
            None
        }

        fn provided_models(&self, _cx: &App) -> Vec<LanguageModel> {
            Vec::new()
        }

        fn is_authenticated(&self, _cx: &App) -> bool {
            true
        }

        fn authenticate(
            &self,
            _cx: &mut App,
        ) -> Task<Result<(), language_model::AuthenticateError>> {
            Task::ready(Ok(()))
        }

        fn settings_view(&self, _cx: &mut App) -> Option<language_model::ProviderSettingsView> {
            None
        }
    }

    fn error_model_hooks(zed_model_error: Vec<Arc<dyn HookHandler>>) -> ModelHooks {
        ModelHooks {
            context: error_hooks(Vec::new()).context,
            before_model: Vec::new(),
            after_model: Vec::new(),
            compaction_trigger: None,
            pre_compact: Vec::new(),
            post_compact: Vec::new(),
            zed_model_error,
        }
    }

    fn test_model() -> LanguageModel {
        language_model::fake_provider::FakeLanguageModelProvider::new(
            LanguageModelProviderId::new("fake"),
            language_model::LanguageModelProviderName::new("fake"),
        )
        .model("qwen")
    }

    #[gpui::test]
    async fn test_a_rejection_at_send_goes_through_the_error_hooks(cx: &mut TestAppContext) {
        let (hook, _) = to_full_window();
        let provider: Arc<dyn LanguageModelProvider> = Arc::new(RejectingProvider);

        let result = error_model_hooks(vec![hook])
            .stream_completion(
                &provider,
                &test_model(),
                LanguageModelRequest::default(),
                None,
                &cx.to_async(),
            )
            .await;

        let error = result.err().expect("the request is rejected");
        assert_eq!(
            category_of(&error),
            Some(ProviderErrorCategory::PromptTooLarge { tokens: None })
        );
    }

    #[gpui::test]
    async fn test_cancelling_a_rejection_at_send_abandons_the_error_hooks(cx: &mut TestAppContext) {
        let abandoned = Arc::new(AtomicBool::new(false));
        let pending: Arc<dyn HookHandler> = Arc::new(PendingHook {
            abandoned: abandoned.clone(),
        });
        let provider: Arc<dyn LanguageModelProvider> = Arc::new(RejectingProvider);
        let (mut cancellation_tx, cancellation_rx) = watch::channel(false);
        let mut task = cx.update(|cx| {
            cx.spawn(async move |cx| {
                error_model_hooks(vec![pending])
                    .stream_completion(
                        &provider,
                        &test_model(),
                        LanguageModelRequest::default(),
                        Some(cancellation_rx),
                        cx,
                    )
                    .await
            })
        });
        cx.run_until_parked();

        cancellation_tx.send(true).ok();
        cx.run_until_parked();

        let result = futures::FutureExt::now_or_never(&mut task)
            .expect("cancelling must stop the error hooks");
        let error = result.err().expect("the request is rejected");
        assert_eq!(
            category_of(&error),
            Some(ProviderErrorCategory::InvalidRequest)
        );
        assert!(abandoned.load(Ordering::SeqCst));
    }

    /// A hook that answers with the next of its responses, then `{}`, and
    /// records its inputs.
    struct SequenceHook {
        responses: RefCell<std::collections::VecDeque<serde_json::Value>>,
        calls: Rc<RefCell<Vec<serde_json::Value>>>,
    }

    impl HookHandler for SequenceHook {
        fn call(&self, input: serde_json::Value, _cx: &mut App) -> Task<Result<serde_json::Value>> {
            self.calls.borrow_mut().push(input);
            let response = self.responses.borrow_mut().pop_front();
            Task::ready(Ok(response.unwrap_or_else(|| json!({}))))
        }
    }

    /// A subagent that records the messages it gets and answers with its
    /// outputs, in order, then "done".
    struct RecordingSubagent {
        outputs: RefCell<std::collections::VecDeque<String>>,
        messages: Rc<RefCell<Vec<String>>>,
        /// Entries from turns before the test.
        entries: std::cell::Cell<usize>,
    }

    impl SubagentHandle for RecordingSubagent {
        fn id(&self) -> acp_v2::SessionId {
            acp_v2::SessionId::new("subagent-1")
        }

        fn num_entries(&self, _cx: &App) -> usize {
            self.entries.get() + self.messages.borrow().len()
        }

        fn label(&self, _cx: &App) -> Option<SharedString> {
            Some("explorer".into())
        }

        fn send(&self, message: String, _cx: &AsyncApp) -> Task<Result<String>> {
            self.messages.borrow_mut().push(message);
            let output = self.outputs.borrow_mut().pop_front();
            let output = output.unwrap_or_else(|| "done".to_string());
            Task::ready(match output.strip_prefix("!error:") {
                Some(error) => Err(anyhow!(error.to_string())),
                None => Ok(output),
            })
        }
    }

    struct SubagentEnvironment {
        subagent: Rc<RecordingSubagent>,
    }

    impl ThreadEnvironment for SubagentEnvironment {
        fn create_terminal(
            &self,
            _command: String,
            _extra_env: Vec<(String, String)>,
            _cwd: Option<PathBuf>,
            _output_byte_limit: Option<u64>,
            _sandbox_wrap: Option<acp_thread::SandboxWrap>,
            _cx: &mut AsyncApp,
        ) -> Task<Result<Rc<dyn TerminalHandle>>> {
            unimplemented!()
        }

        fn create_subagent(
            &self,
            _label: String,
            _model: Option<AgentModelId>,
            _cx: &mut App,
        ) -> Result<Rc<dyn SubagentHandle>> {
            Ok(self.subagent.clone())
        }

        fn resume_subagent(
            &self,
            _session_id: acp_v2::SessionId,
            _cx: &mut App,
        ) -> Result<Rc<dyn SubagentHandle>> {
            Ok(self.subagent.clone())
        }
    }

    fn hooked_environment(outputs: &[&str]) -> (HookedEnvironment, Rc<RefCell<Vec<String>>>) {
        hooked_environment_with_entries(outputs, 0)
    }

    /// A hooked environment whose subagent answers with `outputs`, after
    /// `entries` from earlier turns; returns it with the messages the
    /// subagent received.
    fn hooked_environment_with_entries(
        outputs: &[&str],
        entries: usize,
    ) -> (HookedEnvironment, Rc<RefCell<Vec<String>>>) {
        let messages = Rc::new(RefCell::new(Vec::new()));
        let subagent = Rc::new(RecordingSubagent {
            outputs: RefCell::new(outputs.iter().map(|output| output.to_string()).collect()),
            messages: messages.clone(),
            entries: std::cell::Cell::new(entries),
        });
        let environment = HookedEnvironment {
            inner: Rc::new(SubagentEnvironment { subagent }),
            context: error_hooks(Vec::new()).context,
            subagents: Default::default(),
        };
        (environment, messages)
    }

    fn subagent_hooks(
        start: Vec<(Option<&str>, Arc<dyn HookHandler>)>,
        stop: Vec<Arc<dyn HookHandler>>,
    ) -> AgentHooks {
        let groups = |hooks: Vec<(Option<&str>, Arc<dyn HookHandler>)>| {
            hooks
                .into_iter()
                .map(|(matcher, hook)| HookMatcherGroup {
                    matcher: matcher.map(str::to_string),
                    hooks: vec![hook],
                })
                .collect()
        };
        AgentHooks {
            subagent_start: groups(start),
            subagent_stop: groups(stop.into_iter().map(|hook| (None, hook)).collect()),
            ..Default::default()
        }
    }

    async fn spawn(
        environment: &HookedEnvironment,
        label: &str,
        message: &str,
        cx: &mut TestAppContext,
    ) -> Result<String> {
        let subagent = cx
            .update(|cx| environment.create_subagent(label.to_string(), None, cx))
            .unwrap();
        let task = cx.update(|cx| subagent.send(message.to_string(), &cx.to_async()));
        task.await
    }

    #[gpui::test]
    async fn test_subagent_start_gets_the_subagent_id_and_adds_context(cx: &mut TestAppContext) {
        let (start, start_calls) = fake_hook(Ok(json!({
            "hookSpecificOutput": {
                "hookEventName": "SubagentStart",
                "additionalContext": "Read only: do not edit files.",
            }
        })));
        cx.update(|cx| cx.set_global(subagent_hooks(vec![(None, start)], Vec::new())));
        let (environment, messages) = hooked_environment(&["found it"]);

        let output = spawn(&environment, "explorer", "Find the config", cx).await;

        assert_eq!(output.unwrap(), "found it");
        let calls = start_calls.borrow();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["agent_id"], "subagent-1");
        assert_eq!(calls[0]["agent_type"], "explorer");
        assert_matches_contract("SubagentStart.input.schema.json", &calls[0]);
        assert_eq!(
            *messages.borrow(),
            vec![
                "Find the config\n\n<system-reminder>\nRead only: do not edit files.\n</system-reminder>"
                    .to_string()
            ]
        );
    }

    #[gpui::test]
    async fn test_subagent_start_can_refuse_the_subagent(cx: &mut TestAppContext) {
        let (deny, _) = fake_hook(Ok(json!({
            "hookSpecificOutput": {
                "hookEventName": "SubagentStart",
                "zed": { "deny_reason": "Use the planner profile instead." },
            }
        })));
        cx.update(|cx| cx.set_global(subagent_hooks(vec![(None, deny)], Vec::new())));
        let (environment, messages) = hooked_environment(&[]);

        let output = spawn(&environment, "explorer", "Find the config", cx).await;

        let error = output.expect_err("the subagent is refused").to_string();
        assert!(
            error.contains("Use the planner profile instead."),
            "{error}"
        );
        assert!(messages.borrow().is_empty());
    }

    #[gpui::test]
    async fn test_subagent_stop_can_continue_the_subagent(cx: &mut TestAppContext) {
        let stop_calls = Rc::new(RefCell::new(Vec::new()));
        let stop: Arc<dyn HookHandler> = Arc::new(SequenceHook {
            responses: RefCell::new(
                vec![json!({ "decision": "block", "reason": "Run the tests first." })].into(),
            ),
            calls: stop_calls.clone(),
        });
        cx.update(|cx| cx.set_global(subagent_hooks(Vec::new(), vec![stop])));
        let (environment, messages) = hooked_environment(&["first answer", "second answer"]);

        let output = spawn(&environment, "explorer", "Find the config", cx).await;

        assert_eq!(output.unwrap(), "second answer");
        assert_eq!(
            *messages.borrow(),
            vec![
                "Find the config".to_string(),
                "Run the tests first.".to_string()
            ]
        );
        let calls = stop_calls.borrow();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0]["stop_hook_active"], false);
        assert_eq!(calls[0]["last_assistant_message"], "first answer");
        assert_eq!(calls[1]["stop_hook_active"], true);
        assert_eq!(calls[1]["last_assistant_message"], "second answer");
        assert_matches_contract("SubagentStop.input.schema.json", &calls[0]);
    }

    #[gpui::test]
    async fn test_subagent_hooks_match_the_label(cx: &mut TestAppContext) {
        let (start, start_calls) = fake_hook(Ok(json!({})));
        cx.update(|cx| cx.set_global(subagent_hooks(vec![(Some("explorer"), start)], Vec::new())));
        let (environment, _) = hooked_environment(&[]);

        spawn(&environment, "planner", "Plan it", cx).await.unwrap();

        assert!(start_calls.borrow().is_empty());
    }

    #[gpui::test]
    async fn test_a_resumed_subagent_runs_only_its_stop_hooks(cx: &mut TestAppContext) {
        let (start, start_calls) = fake_hook(Ok(json!({})));
        let (stop, stop_calls) = fake_hook(Ok(json!({})));
        cx.update(|cx| cx.set_global(subagent_hooks(vec![(None, start)], vec![stop])));
        let (environment, _) = hooked_environment(&[]);
        spawn(&environment, "explorer", "Find the config", cx)
            .await
            .unwrap();

        let resumed = cx
            .update(|cx| environment.resume_subagent(acp_v2::SessionId::new("subagent-1"), cx))
            .unwrap();
        let task = cx.update(|cx| resumed.send("And the tests?".to_string(), &cx.to_async()));
        task.await.unwrap();

        assert_eq!(start_calls.borrow().len(), 1);
        let stop_calls = stop_calls.borrow();
        assert_eq!(stop_calls.len(), 2);
        assert_eq!(stop_calls[1]["agent_type"], "explorer");
    }

    /// Answers like `hook`, after `delay`.
    struct DelayedHook {
        hook: Arc<dyn HookHandler>,
        delay: Duration,
    }

    impl HookHandler for DelayedHook {
        fn call(&self, input: serde_json::Value, cx: &mut App) -> Task<Result<serde_json::Value>> {
            let answer = self.hook.call(input, cx);
            let timer = cx.background_executor().timer(self.delay);
            cx.background_executor().spawn(async move {
                timer.await;
                answer.await
            })
        }
    }

    fn spawn_with_cancellation(
        environment: HookedEnvironment,
        cx: &mut TestAppContext,
    ) -> (
        Task<Result<crate::SpawnAgentToolOutput, crate::SpawnAgentToolOutput>>,
        watch::Sender<bool>,
    ) {
        use crate::AgentTool as _;
        let tool = Arc::new(crate::SpawnAgentTool::new(Rc::new(environment)));
        let (event_stream, _receiver, cancellation_tx) =
            ToolCallEventStream::test_with_cancellation();
        let task = cx.update(|cx| {
            tool.run(
                ToolInput::resolved(crate::SpawnAgentToolInput {
                    label: "explorer".to_string(),
                    message: "Find the config".to_string(),
                    session_id: None,
                    model: None,
                }),
                event_stream,
                cx,
            )
        });
        (task, cancellation_tx)
    }

    #[gpui::test]
    async fn test_cancelling_during_subagent_start_never_runs_the_subagent(
        cx: &mut TestAppContext,
    ) {
        let abandoned = Arc::new(AtomicBool::new(false));
        let pending: Arc<dyn HookHandler> = Arc::new(PendingHook {
            abandoned: abandoned.clone(),
        });
        cx.update(|cx| cx.set_global(subagent_hooks(vec![(None, pending)], Vec::new())));
        let (environment, messages) = hooked_environment(&[]);
        let (mut task, mut cancellation_tx) = spawn_with_cancellation(environment, cx);
        cx.run_until_parked();

        cancellation_tx.send(true).ok();
        cx.run_until_parked();

        assert!(
            futures::FutureExt::now_or_never(&mut task).is_some(),
            "cancelling must end spawn_agent"
        );
        assert!(abandoned.load(Ordering::SeqCst));
        assert!(messages.borrow().is_empty());
    }

    #[gpui::test]
    async fn test_cancelling_during_subagent_stop_never_continues_the_subagent(
        cx: &mut TestAppContext,
    ) {
        let (block, _) = fake_hook(Ok(json!({ "decision": "block", "reason": "Keep going." })));
        let slow_block: Arc<dyn HookHandler> = Arc::new(DelayedHook {
            hook: block,
            delay: Duration::from_secs(1),
        });
        cx.update(|cx| cx.set_global(subagent_hooks(Vec::new(), vec![slow_block])));
        let (environment, messages) = hooked_environment(&["first answer"]);
        let (mut task, mut cancellation_tx) = spawn_with_cancellation(environment, cx);
        cx.run_until_parked();
        assert_eq!(messages.borrow().len(), 1);

        cancellation_tx.send(true).ok();
        cx.run_until_parked();
        cx.executor().advance_clock(Duration::from_secs(2));
        cx.run_until_parked();

        assert!(futures::FutureExt::now_or_never(&mut task).is_some());
        assert_eq!(*messages.borrow(), vec!["Find the config".to_string()]);
    }

    #[gpui::test]
    async fn test_resuming_a_refused_subagent_runs_subagent_start_again(cx: &mut TestAppContext) {
        let start_calls = Rc::new(RefCell::new(Vec::new()));
        let deny = json!({
            "hookSpecificOutput": {
                "hookEventName": "SubagentStart",
                "zed": { "deny_reason": "Not this one." },
            }
        });
        let deny_twice: Arc<dyn HookHandler> = Arc::new(SequenceHook {
            responses: RefCell::new(vec![deny.clone(), deny].into()),
            calls: start_calls.clone(),
        });
        cx.update(|cx| cx.set_global(subagent_hooks(vec![(None, deny_twice)], Vec::new())));
        let (environment, messages) = hooked_environment(&[]);
        assert!(
            spawn(&environment, "explorer", "Find the config", cx)
                .await
                .is_err()
        );

        let resumed = cx
            .update(|cx| environment.resume_subagent(acp_v2::SessionId::new("subagent-1"), cx))
            .unwrap();
        let task = cx.update(|cx| resumed.send("Try again".to_string(), &cx.to_async()));

        assert!(task.await.is_err(), "the refusal must hold on a resume");
        assert_eq!(start_calls.borrow().len(), 2);
        assert!(messages.borrow().is_empty());
    }

    #[gpui::test]
    async fn test_a_resumed_subagent_keeps_its_label_after_a_reopen(cx: &mut TestAppContext) {
        let (stop, stop_calls) = fake_hook(Ok(json!({})));
        cx.update(|cx| {
            cx.set_global(AgentHooks {
                subagent_stop: vec![HookMatcherGroup {
                    matcher: Some("explorer".into()),
                    hooks: vec![stop],
                }],
                ..Default::default()
            })
        });
        // A fresh environment, as after reopening the parent thread: it never
        // saw the subagent being created.
        let (environment, _) = hooked_environment(&[]);

        let resumed = cx
            .update(|cx| environment.resume_subagent(acp_v2::SessionId::new("subagent-1"), cx))
            .unwrap();
        let task = cx.update(|cx| resumed.send("And the tests?".to_string(), &cx.to_async()));
        task.await.unwrap();

        let stop_calls = stop_calls.borrow();
        assert_eq!(stop_calls.len(), 1);
        assert_eq!(stop_calls[0]["agent_type"], "explorer");
    }

    #[gpui::test]
    async fn test_a_subagent_that_already_ran_is_not_started_again_after_a_reopen(
        cx: &mut TestAppContext,
    ) {
        let (start, start_calls) = fake_hook(Ok(json!({
            "hookSpecificOutput": {
                "hookEventName": "SubagentStart",
                "additionalContext": "Read only.",
            }
        })));
        cx.update(|cx| cx.set_global(subagent_hooks(vec![(None, start)], Vec::new())));
        // A fresh environment, as after reopening the parent, for a subagent
        // that already ran two turns.
        let (environment, messages) = hooked_environment_with_entries(&[], 2);

        let resumed = cx
            .update(|cx| environment.resume_subagent(acp_v2::SessionId::new("subagent-1"), cx))
            .unwrap();
        let task = cx.update(|cx| resumed.send("And the tests?".to_string(), &cx.to_async()));
        task.await.unwrap();

        assert!(start_calls.borrow().is_empty());
        assert_eq!(*messages.borrow(), vec!["And the tests?".to_string()]);
    }

    #[gpui::test]
    async fn test_a_failed_continuation_keeps_the_first_answer(cx: &mut TestAppContext) {
        let (block, _) = fake_hook(Ok(json!({ "decision": "block", "reason": "Check again." })));
        cx.update(|cx| cx.set_global(subagent_hooks(Vec::new(), vec![block])));
        let (environment, _) = hooked_environment(&["first answer", "!error:model overloaded"]);

        let error = spawn(&environment, "explorer", "Find the config", cx)
            .await
            .expect_err("the continuation failed")
            .to_string();

        assert!(error.contains("model overloaded"), "{error}");
        assert!(error.contains("first answer"), "{error}");
    }
    #[test]
    fn test_user_prompt_submit_plain_text_is_context() {
        assert_eq!(
            parse_command_output("UserPromptSubmit", Some(0), "Branch: feat/hooks\n", "").unwrap(),
            json!({
                "hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit",
                    "additionalContext": "Branch: feat/hooks",
                }
            })
        );
        // Other events still ignore plain text.
        assert_eq!(
            parse_command_output("SessionStart", Some(0), "hello", "").unwrap(),
            json!({})
        );
    }

    #[test]
    fn test_user_prompt_submit_hooks_time_out_sooner() {
        assert_eq!(
            default_hook_timeout("UserPromptSubmit"),
            Duration::from_secs(30)
        );
        assert_eq!(default_hook_timeout("Stop"), Duration::from_secs(600));
    }

    #[test]
    fn test_agent_hooks_from_settings_reads_conversation_events() {
        let groups = json!([{ "hooks": [{ "type": "command", "command": "check" }] }]);
        let hooks = AgentHooks::from_settings(
            &json!({
                "Stop": groups,
                "UserPromptSubmit": groups,
                "SessionStart": groups,
                "SessionEnd": groups,
            }),
            Path::new("/data/hooks"),
        )
        .unwrap();

        assert_eq!(hooks.stop.len(), 1);
        assert_eq!(hooks.user_prompt_submit.len(), 1);
        assert_eq!(hooks.session_start.len(), 1);
        assert_eq!(hooks.session_end.len(), 1);
    }

    #[gpui::test]
    async fn test_a_post_tool_use_pause_runs_the_whole_chain(cx: &mut TestAppContext) {
        let (pause, _) = fake_hook(Ok(json!({ "continue": false, "stopReason": "Stop here." })));
        let (later, later_calls) = fake_hook(Ok(json!({})));
        let (tool, _) = hooked_echo(Vec::new(), vec![pause, later]);

        let (is_error, texts) = run_echo(tool, "hello", cx).await;

        assert!(!is_error);
        assert_eq!(texts, vec!["hello".to_string()]);
        assert_eq!(later_calls.borrow().len(), 1);
        assert_eq!(
            cx.update(|cx| take_pause("session-1", cx)).as_deref(),
            Some("Stop here.")
        );
        assert_eq!(cx.update(|cx| take_pause("session-1", cx)), None);
    }

    #[gpui::test]
    async fn test_a_pre_tool_use_pause_refuses_the_call_and_stops_the_chain(
        cx: &mut TestAppContext,
    ) {
        let mut output = pre_tool_use(json!({ "updatedInput": { "text": "changed" } }));
        output["continue"] = json!(false);
        output["stopReason"] = json!("Out of scope.");
        let (pause, _) = fake_hook(Ok(output));
        let (later, later_calls) = fake_hook(Ok(json!({})));
        let (tool, echo) = hooked_echo(vec![pause, later], Vec::new());

        let (is_error, texts) = run_echo(tool, "hello", cx).await;

        assert!(is_error);
        assert_eq!(texts, vec!["Out of scope.".to_string()]);
        assert_eq!(echo.runs.get(), 0);
        assert!(later_calls.borrow().is_empty());
        assert_eq!(
            cx.update(|cx| take_pause("session-1", cx)).as_deref(),
            Some("Out of scope.")
        );
    }

    #[gpui::test]
    async fn test_a_paused_subagent_is_not_continued_by_subagent_stop(cx: &mut TestAppContext) {
        let (block, block_calls) =
            fake_hook(Ok(json!({ "decision": "block", "reason": "Again." })));
        cx.update(|cx| cx.set_global(subagent_hooks(Vec::new(), vec![block])));
        let (environment, messages) = hooked_environment(&["paused answer"]);
        // The subagent's turn ended on a pause, passed on to its parent.
        cx.update(|cx| record_pause("session-1", "Out of scope.", cx));

        let output = spawn(&environment, "explorer", "Find the config", cx).await;

        assert_eq!(output.unwrap(), "paused answer");
        assert!(block_calls.borrow().is_empty());
        assert_eq!(messages.borrow().len(), 1);
    }
    #[test]
    fn test_remove_tools_are_validated() {
        let output = |names: serde_json::Value| {
            json!({
                "hookSpecificOutput": { "hookEventName": "BeforeModel", "remove_tools": names }
            })
        };

        assert_eq!(remove_tools(&json!({})).unwrap(), Vec::<&str>::new());
        assert_eq!(
            remove_tools(&output(json!(["terminal", "edit_file"]))).unwrap(),
            vec!["terminal", "edit_file"]
        );
        assert!(remove_tools(&output(json!("terminal"))).is_err());
        assert!(remove_tools(&output(json!([""]))).is_err());
        assert!(remove_tools(&output(json!([42]))).is_err());
        assert!(remove_tools(&output(json!(["terminal", "terminal"]))).is_err());
    }

    #[test]
    fn test_removed_tools_leave_this_request_but_not_compaction() {
        let tool = |name: &str| {
            language_model::LanguageModelRequestTool::function(
                name.into(),
                String::new(),
                json!({}),
                false,
            )
        };
        let conversation = LanguageModelRequest {
            intent: Some(CompletionIntent::UserPrompt),
            messages: vec![
                message(Role::System, vec!["You are Zed.".into()]),
                message(Role::User, vec!["Hello".into()]),
            ],
            tools: vec![tool("read_file"), tool("terminal")],
            ..Default::default()
        };
        let names = |request: &LanguageModelRequest| {
            request
                .tools
                .iter()
                .map(|tool| tool.name.clone())
                .collect::<Vec<_>>()
        };

        let mut request = conversation.clone();
        remove_request_tools(&mut request, &["terminal".into(), "missing".into()]);
        assert_eq!(names(&request), vec!["read_file".to_string()]);

        let mut compaction = conversation.clone();
        compaction.intent = Some(CompletionIntent::ThreadContextSummarization);
        compaction
            .messages
            .push(message(Role::User, vec![COMPACTION_PROMPT.into()]));
        remove_request_tools(&mut compaction, &["terminal".into()]);
        assert_eq!(
            names(&compaction),
            vec!["read_file".to_string(), "terminal".to_string()]
        );
    }
    /// Whether node, which runs the agent profiles hook, is missing; the
    /// tests that need it are then skipped, with a message.
    pub(crate) fn node_is_missing() -> bool {
        let missing = std::process::Command::new("node")
            .arg("--version")
            .output()
            .is_err();
        if missing {
            eprintln!("skipped: node is not installed, and the agent profiles hook needs it");
        }
        missing
    }

    /// The agent profiles hook of the contract answers each event with an
    /// output its schema accepts.
    #[gpui::test]
    async fn test_agent_profiles_script_outputs_match_the_contract(cx: &mut TestAppContext) {
        if node_is_missing() {
            return;
        }
        cx.executor().allow_parking();
        let profiles = Path::new(env!("CARGO_MANIFEST_DIR")).join("hook_contract/profils");
        let data_dir = std::env::temp_dir().join(format!(
            "zed-agent-profiles-schema-{}-{}",
            std::process::id(),
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default()
        ));
        let hook = CommandHook {
            command: "node".into(),
            args: Some(
                [
                    profiles.join("agent-profiles.mts"),
                    "--agents".into(),
                    profiles.join("test-agents"),
                ]
                .iter()
                .map(|arg| arg.to_string_lossy().into_owned())
                .collect(),
            ),
            shell: None,
            timeout: Some(30),
            data_dir: data_dir.clone(),
            block_on_failure: false,
        };
        let root = HookContext {
            session_id: "root-1".into(),
            cwd: profiles.to_string_lossy().into_owned(),
            thread_id: "root-1".into(),
            parent_thread_id: None,
            depth: 0,
        };
        let subagent = HookContext {
            thread_id: "sub-1".into(),
            parent_thread_id: Some("root-1".into()),
            depth: 1,
            ..root.clone()
        };
        let before_model = |context: &HookContext| {
            context.input(
                "BeforeModel",
                json!({
                    "request_id": "request-1",
                    "intent": "user_prompt",
                    "model": { "provider": "fake", "id": "fake" },
                    "is_compaction": false,
                    "new_messages": [],
                    "known_message_count": 0,
                    "total_message_count": 2,
                    "max_output_tokens": 1000,
                    "tool_names": ["edit_file", "fetch", "read_file", "spawn_agent", "write_file"],
                    "skill_names": ["create-skill", "exploring-code", "tdd"],
                }),
            )
        };
        let spawn = |label: &str| {
            root.input(
                "PreToolUse",
                json!({
                    "tool_name": "spawn_agent",
                    "tool_input": { "label": label, "message": "Find it." },
                    "tool_use_id": "tool-1",
                }),
            )
        };
        let start = root.input(
            "SubagentStart",
            json!({ "agent_id": "sub-1", "agent_type": "[explorateur#a1b2] Find it." }),
        );
        let exchanges = [
            ("BeforeModel", before_model(&root)),
            ("PreToolUse", spawn("Find it.")),
            ("PreToolUse", spawn("[explorateur] Find it.")),
            ("SubagentStart", start),
            ("BeforeModel", before_model(&subagent)),
            (
                "PreToolUse",
                subagent.input(
                    "PreToolUse",
                    json!({
                        "tool_name": "skill",
                        "tool_input": { "name": "tdd" },
                        "tool_use_id": "tool-2",
                    }),
                ),
            ),
        ];

        for (event, input) in exchanges {
            assert_matches_contract(&format!("{event}.input.schema.json"), &input);
            let output = cx.update(|cx| hook.call(input, cx)).await.unwrap();
            // SubagentStart only records the profile.
            if event != "SubagentStart" {
                assert_ne!(output, json!({}), "{event} gave no output");
            }
            assert_matches_contract(&format!("{event}.output.schema.json"), &output);
        }
        std::fs::remove_dir_all(&data_dir).ok();
    }

    #[test]
    fn test_remove_skills_are_validated() {
        let output = |names: serde_json::Value| {
            json!({
                "hookSpecificOutput": { "hookEventName": "BeforeModel", "remove_skills": names }
            })
        };

        assert_eq!(remove_skills(&json!({})).unwrap(), Vec::<&str>::new());
        assert_eq!(
            remove_skills(&output(json!(["tdd", "create-skill"]))).unwrap(),
            vec!["tdd", "create-skill"]
        );
        assert!(remove_skills(&output(json!("tdd"))).is_err());
        assert!(remove_skills(&output(json!([""]))).is_err());
        assert!(remove_skills(&output(json!(["tdd", "tdd"]))).is_err());
    }

    #[test]
    fn test_removed_skills_leave_the_catalogue_of_this_request() {
        let system = indoc::indoc! {"
            You are Zed.

            <available_skills>
              <skill>
                <name>tdd</name>
                <description>Red, then green.</description>
                <location>/home/me/.agents/skills/tdd/SKILL.md</location>
              </skill>
              <skill>
                <name>exploring-code</name>
                <description>Locate, then read.</description>
                <location>/home/me/.agents/skills/exploring-code/SKILL.md</location>
              </skill>
            </available_skills>

            To use a Skill: <name>tdd</name> stays outside the catalogue.
        "};
        let conversation = LanguageModelRequest {
            intent: Some(CompletionIntent::UserPrompt),
            messages: vec![
                message(Role::System, vec![system.into()]),
                message(Role::User, vec!["Hello".into()]),
            ],
            ..Default::default()
        };

        let mut request = conversation.clone();
        remove_request_skills(&mut request, &["tdd".into(), "missing".into()]);
        let text = request.messages[0].string_contents();
        assert!(
            !text.contains(
                "<name>tdd</name>
"
            ),
            "{text}"
        );
        assert!(text.contains("<name>exploring-code</name>"), "{text}");
        assert!(
            text.contains(
                "<available_skills>
  <skill>
    <name>exploring-code"
            ),
            "{text}"
        );
        assert!(text.contains("stays outside the catalogue"), "{text}");
        remove_request_skills(&mut request, &["exploring-code".into()]);
        let text = request.messages[0].string_contents();
        assert!(
            text.contains(
                "<available_skills>
</available_skills>"
            ),
            "{text}"
        );

        let mut compaction = conversation.clone();
        compaction.intent = Some(CompletionIntent::ThreadContextSummarization);
        compaction
            .messages
            .push(message(Role::User, vec![COMPACTION_PROMPT.into()]));
        remove_request_skills(&mut compaction, &["tdd".into()]);
        assert_eq!(compaction.messages[0].string_contents(), system);

        assert_eq!(
            catalogue_skill_names(&conversation),
            vec!["exploring-code".to_string(), "tdd".to_string()]
        );
        let mut without_catalogue = conversation.clone();
        without_catalogue.messages[0] = message(Role::System, vec!["You are Zed.".into()]);
        assert_eq!(
            catalogue_skill_names(&without_catalogue),
            Vec::<String>::new()
        );
    }
    #[test]
    fn test_on_failure_takes_continue_or_block_only() {
        let settings = |on_failure: &str| {
            json!({
                "PreToolUse": [{
                    "hooks": [{ "type": "command", "command": "guard", "onFailure": on_failure }],
                }],
            })
        };
        for value in ["continue", "block"] {
            assert!(
                AgentHooks::from_settings(&settings(value), Path::new("/data/hooks")).is_ok(),
                "{value}"
            );
        }
        let error = AgentHooks::from_settings(&settings("sometimes"), Path::new("/data/hooks"))
            .err()
            .expect("an unknown onFailure must be refused")
            .to_string();
        assert!(error.contains("PreToolUse[0].hooks[0]"), "{error}");
    }

    /// Calls the first handler of `event`, declared with `on_failure`.
    async fn call_failing_hook(
        event: &str,
        script: &str,
        on_failure: &str,
        cx: &mut TestAppContext,
    ) -> Result<serde_json::Value> {
        let hooks = AgentHooks::from_settings(
            &json!({
                event: [{
                    "hooks": [{
                        "type": "command",
                        "command": "bash",
                        "args": ["-c", script],
                        "onFailure": on_failure,
                    }],
                }],
            }),
            &std::env::temp_dir().join("zed-agent-hook-on-failure"),
        )?;
        let groups = match event {
            "PreToolUse" => &hooks.pre_tool_use,
            "UserPromptSubmit" => &hooks.user_prompt_submit,
            "Stop" => &hooks.stop,
            _ => unreachable!(),
        };
        let hook = groups[0].hooks[0].clone();
        let input = json!({ "hook_event_name": event, "tool_name": "echo" });
        cx.update(|cx| hook.call(input, cx)).await
    }

    #[gpui::test]
    async fn test_on_failure_block_turns_a_failure_into_a_refusal(cx: &mut TestAppContext) {
        cx.executor().allow_parking();
        for (failure, script) in [
            ("exit code", "cat > /dev/null; echo boom >&2; exit 1"),
            ("invalid JSON", "cat > /dev/null; echo '{not json}'"),
            (
                "another event",
                r#"cat > /dev/null; echo '{"hookSpecificOutput":{"hookEventName":"PostToolUse"}}'"#,
            ),
        ] {
            let output = call_failing_hook("PreToolUse", script, "block", cx)
                .await
                .unwrap_or_else(|error| panic!("{failure}: {error:#}"));
            assert_eq!(
                output["hookSpecificOutput"]["permissionDecision"], "deny",
                "{failure}"
            );
            assert_matches_contract("PreToolUse.output.schema.json", &output);

            // Without `block`, a failure stays a failure: the action goes on.
            assert!(
                call_failing_hook("PreToolUse", script, "continue", cx)
                    .await
                    .is_err(),
                "{failure}"
            );
        }

        let blocked = call_failing_hook("UserPromptSubmit", "exit 1", "block", cx)
            .await
            .unwrap();
        assert_eq!(blocked["decision"], "block");
        // As in Claude Code, `onFailure` has no effect on Stop.
        assert!(
            call_failing_hook("Stop", "exit 1", "block", cx)
                .await
                .is_err()
        );
    }
    #[test]
    fn test_http_hook_settings_are_checked() {
        let http = |handler: serde_json::Value| {
            AgentHooks::from_settings(
                &json!({ "PreToolUse": [{ "hooks": [handler] }] }),
                Path::new("/data/hooks"),
            )
        };
        let hooks = http(json!({
            "type": "http",
            "url": "http://localhost:8787/hooks",
            "headers": { "Authorization": "Bearer $TOKEN" },
            "allowedEnvVars": ["TOKEN"],
            "timeout": 5,
            "onFailure": "block",
        }))
        .unwrap();
        assert_eq!(hooks.pre_tool_use[0].hooks.len(), 1);

        for (handler, reason) in [
            (json!({ "type": "http" }), "url"),
            (json!({ "type": "http", "url": "not a url" }), "url"),
            (
                json!({ "type": "http", "url": "http://x", "id": "guard" }),
                "id",
            ),
            (
                json!({ "type": "http", "url": "http://x", "command": "guard" }),
                "command",
            ),
            (
                json!({ "type": "http", "url": "http://x", "headers": { "A": 1 } }),
                "headers",
            ),
            (
                json!({ "type": "http", "url": "http://x", "allowedEnvVars": "TOKEN" }),
                "allowedEnvVars",
            ),
        ] {
            let error = http(handler.clone())
                .err()
                .unwrap_or_else(|| panic!("must be refused: {handler}"));
            let error = format!("{error:#}");
            assert!(error.contains("PreToolUse[0].hooks[0]"), "{error}");
            assert!(error.contains(reason), "{error}");
        }
    }

    #[test]
    fn test_http_header_values_take_allowed_variables_only() {
        let lookup = |name: &str| match name {
            "TOKEN" => Some("secret".to_string()),
            "OTHER" => Some("leak".to_string()),
            _ => None,
        };
        let allowed = vec!["TOKEN".to_string(), "MISSING".to_string()];
        assert_eq!(
            interpolate_header("Bearer $TOKEN", &allowed, lookup),
            "Bearer secret"
        );
        assert_eq!(
            interpolate_header("x-${TOKEN}-y", &allowed, lookup),
            "x-secret-y"
        );
        assert_eq!(
            interpolate_header("$OTHER|$MISSING|", &allowed, lookup),
            "||"
        );
        assert_eq!(
            interpolate_header("no variable", &allowed, lookup),
            "no variable"
        );
    }

    /// An HTTP hook from settings whose server answers `status` and `body`;
    /// returns the hook output and the requests the server got.
    async fn call_http_hook(
        event: &str,
        status: u16,
        body: &'static str,
        on_failure: &str,
        cx: &mut TestAppContext,
    ) -> (
        Result<serde_json::Value>,
        Arc<std::sync::Mutex<Vec<(String, String, String, String)>>>,
    ) {
        let requests = Arc::new(std::sync::Mutex::new(Vec::new()));
        let client = gpui::http_client::FakeHttpClient::create({
            let requests = requests.clone();
            move |request| {
                let requests = requests.clone();
                async move {
                    let header = |name: &str| {
                        request
                            .headers()
                            .get(name)
                            .and_then(|value| value.to_str().ok())
                            .unwrap_or_default()
                            .to_string()
                    };
                    let (content_type, authorization) =
                        (header("content-type"), header("authorization"));
                    let method = request.method().to_string();
                    let mut sent = String::new();
                    let mut request_body = request.into_body();
                    futures::AsyncReadExt::read_to_string(&mut request_body, &mut sent).await?;
                    requests
                        .lock()
                        .unwrap()
                        .push((method, content_type, authorization, sent));
                    Ok(gpui::http_client::Response::builder()
                        .status(status)
                        .body(body.into())
                        .unwrap())
                }
            }
        });
        cx.update(|cx| cx.set_http_client(client));
        let hooks = AgentHooks::from_settings(
            &json!({
                event: [{
                    "hooks": [{
                        "type": "http",
                        "url": "http://localhost:8787/hooks",
                        "headers": { "Authorization": "Bearer $ZED_TEST_ABSENT_TOKEN" },
                        "allowedEnvVars": ["ZED_TEST_ABSENT_TOKEN"],
                        "onFailure": on_failure,
                    }],
                }],
            }),
            Path::new("/data/hooks"),
        )
        .unwrap();
        let hook = match event {
            "PreToolUse" => hooks.pre_tool_use[0].hooks[0].clone(),
            "Stop" => hooks.stop[0].hooks[0].clone(),
            _ => unreachable!(),
        };
        let input = json!({ "hook_event_name": event, "tool_name": "echo" });
        let output = cx.update(|cx| hook.call(input, cx)).await;
        (output, requests)
    }

    #[gpui::test]
    async fn test_http_hooks_post_the_input_and_read_a_2xx_body(cx: &mut TestAppContext) {
        let deny = r#"{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"No."}}"#;
        let (output, requests) = call_http_hook("PreToolUse", 200, deny, "continue", cx).await;
        assert_eq!(
            output.unwrap()["hookSpecificOutput"]["permissionDecision"],
            "deny"
        );
        let requests = requests.lock().unwrap();
        let (method, content_type, authorization, sent) = &requests[0];
        assert_eq!(method, "POST");
        assert_eq!(content_type, "application/json");
        // An allowed variable that is not set becomes empty.
        assert_eq!(authorization, "Bearer ");
        let sent: serde_json::Value = serde_json::from_str(sent).unwrap();
        assert_eq!(sent["hook_event_name"], "PreToolUse");

        // An empty 2xx body is a success without fields.
        let (output, _) = call_http_hook("PreToolUse", 204, "", "continue", cx).await;
        assert_eq!(output.unwrap(), json!({}));

        // A status outside 2xx, a body that is not an object, or an output
        // for another event fail; the action goes on.
        let other_event = r#"{"hookSpecificOutput":{"hookEventName":"PostToolUse"}}"#;
        for (status, body) in [
            (500, deny),
            (200, "[1]"),
            (200, "not json"),
            (200, other_event),
        ] {
            let (output, _) = call_http_hook("PreToolUse", status, body, "continue", cx).await;
            assert!(output.is_err(), "{status} {body}");
        }

        // With `block`, a failure refuses, as exit code 2 would; not on Stop.
        let (output, _) = call_http_hook("PreToolUse", 503, "", "block", cx).await;
        let output = output.unwrap();
        assert_eq!(output["hookSpecificOutput"]["permissionDecision"], "deny");
        assert_matches_contract("PreToolUse.output.schema.json", &output);
        let (output, _) = call_http_hook("Stop", 503, "", "block", cx).await;
        assert!(output.is_err());
    }
    /// The first hook of `event` declared by the http `handler`, served by
    /// a fake client that answers each request with `respond`.
    fn http_hook_with(
        event: &str,
        handler: serde_json::Value,
        respond: impl Fn(
            gpui::http_client::Request<http_client::AsyncBody>,
        ) -> gpui::http_client::Response<http_client::AsyncBody>
        + Send
        + Sync
        + 'static,
        cx: &mut TestAppContext,
    ) -> Arc<dyn HookHandler> {
        let respond = Arc::new(respond);
        let client = gpui::http_client::FakeHttpClient::create(move |request| {
            let response = respond(request);
            async move { Ok(response) }
        });
        cx.update(|cx| cx.set_http_client(client));
        let hooks = AgentHooks::from_settings(
            &json!({ event: [{ "hooks": [handler] }] }),
            Path::new("/data/hooks"),
        )
        .unwrap();
        match event {
            "PreToolUse" => hooks.pre_tool_use[0].hooks[0].clone(),
            _ => unreachable!(),
        }
    }

    fn respond_with(
        status: u16,
        body: http_client::AsyncBody,
    ) -> gpui::http_client::Response<http_client::AsyncBody> {
        gpui::http_client::Response::builder()
            .status(status)
            .body(body)
            .unwrap()
    }

    /// A body that never ends.
    struct EndlessBody;

    impl futures::AsyncRead for EndlessBody {
        fn poll_read(
            self: std::pin::Pin<&mut Self>,
            _cx: &mut std::task::Context<'_>,
            _buf: &mut [u8],
        ) -> std::task::Poll<std::io::Result<usize>> {
            std::task::Poll::Pending
        }
    }

    #[gpui::test]
    async fn test_http_hook_timeout_covers_the_body(cx: &mut TestAppContext) {
        for on_failure in ["continue", "block"] {
            let hook = http_hook_with(
                "PreToolUse",
                json!({
                    "type": "http",
                    "url": "http://localhost:8787/hooks",
                    "timeout": 1,
                    "onFailure": on_failure,
                }),
                |_| respond_with(200, http_client::AsyncBody::from_reader(EndlessBody)),
                cx,
            );
            let input = json!({ "hook_event_name": "PreToolUse", "tool_name": "echo" });
            let task = cx.update(|cx| hook.call(input, cx));
            cx.run_until_parked();
            cx.executor().advance_clock(Duration::from_secs(2));
            let output = task.await;
            match on_failure {
                "continue" => assert!(output.is_err(), "{output:?}"),
                _ => assert_eq!(
                    output.unwrap()["hookSpecificOutput"]["permissionDecision"],
                    "deny"
                ),
            }
        }
    }

    #[gpui::test]
    async fn test_http_hook_follows_no_redirect(cx: &mut TestAppContext) {
        let policies = Arc::new(std::sync::Mutex::new(Vec::new()));
        let hook = http_hook_with(
            "PreToolUse",
            json!({ "type": "http", "url": "http://localhost:8787/hooks" }),
            {
                let policies = policies.clone();
                move |request| {
                    policies.lock().unwrap().push(
                        request
                            .extensions()
                            .get::<http_client::RedirectPolicy>()
                            .cloned(),
                    );
                    gpui::http_client::Response::builder()
                        .status(307)
                        .header("location", "http://elsewhere.example/steal")
                        .body(http_client::AsyncBody::empty())
                        .unwrap()
                }
            },
            cx,
        );
        let input = json!({ "hook_event_name": "PreToolUse", "tool_name": "echo" });
        assert!(cx.update(|cx| hook.call(input, cx)).await.is_err());
        assert_eq!(
            *policies.lock().unwrap(),
            vec![Some(http_client::RedirectPolicy::NoFollow)]
        );
    }

    #[gpui::test]
    async fn test_http_hook_body_is_capped_and_blank_is_empty(cx: &mut TestAppContext) {
        let input = json!({ "hook_event_name": "PreToolUse", "tool_name": "echo" });
        let too_long = format!(
            "{{\"padding\":\"{}\"}}",
            "x".repeat(MAX_HTTP_HOOK_BODY_BYTES)
        );
        let hook = http_hook_with(
            "PreToolUse",
            json!({ "type": "http", "url": "http://localhost:8787/hooks" }),
            move |_| respond_with(200, too_long.clone().into()),
            cx,
        );
        assert!(cx.update(|cx| hook.call(input.clone(), cx)).await.is_err());

        let hook = http_hook_with(
            "PreToolUse",
            json!({ "type": "http", "url": "http://localhost:8787/hooks" }),
            |_| respond_with(200, " \n\t ".into()),
            cx,
        );
        assert_eq!(
            cx.update(|cx| hook.call(input, cx)).await.unwrap(),
            json!({})
        );
    }

    #[gpui::test]
    async fn test_http_hook_content_type_can_be_replaced(cx: &mut TestAppContext) {
        let content_types = Arc::new(std::sync::Mutex::new(Vec::new()));
        let hook = http_hook_with(
            "PreToolUse",
            json!({
                "type": "http",
                "url": "http://localhost:8787/hooks",
                "headers": { "content-type": "application/vnd.api+json" },
            }),
            {
                let content_types = content_types.clone();
                move |request| {
                    content_types.lock().unwrap().extend(
                        request
                            .headers()
                            .get_all("content-type")
                            .iter()
                            .map(|value| value.to_str().unwrap().to_string()),
                    );
                    respond_with(204, http_client::AsyncBody::empty())
                }
            },
            cx,
        );
        let input = json!({ "hook_event_name": "PreToolUse", "tool_name": "echo" });
        cx.update(|cx| hook.call(input, cx)).await.unwrap();
        assert_eq!(
            *content_types.lock().unwrap(),
            vec!["application/vnd.api+json".to_string()]
        );
    }

    #[test]
    fn test_hook_settings_check_status_messages_and_header_names() {
        let refused = |handler: serde_json::Value| {
            format!(
                "{:#}",
                AgentHooks::from_settings(
                    &json!({ "PreToolUse": [{ "hooks": [handler.clone()] }] }),
                    Path::new("/data/hooks"),
                )
                .err()
                .unwrap_or_else(|| panic!("must be refused: {handler}"))
            )
        };
        for handler in [
            json!({ "type": "http", "url": "http://x", "statusMessage": 42 }),
            json!({ "type": "command", "command": "guard", "statusMessage": ["a"] }),
        ] {
            assert!(refused(handler).contains("statusMessage"));
        }
        let error =
            refused(json!({ "type": "http", "url": "http://x", "headers": { "bad name": "x" } }));
        assert!(error.contains("bad name"), "{error}");
        // An unknown field is named, and only it.
        let error = refused(json!({ "type": "http", "url": "http://x", "command": "guard" }));
        assert!(error.contains("`command`"), "{error}");
        assert!(!error.contains("data directory"), "{error}");
    }

    #[test]
    fn test_hooks_differing_by_on_failure_both_run() {
        for handler in [
            json!({ "type": "command", "command": "guard" }),
            json!({ "type": "http", "url": "http://x" }),
        ] {
            let mut blocking = handler.clone();
            blocking["onFailure"] = json!("block");
            let hooks = AgentHooks::from_settings(
                &json!({ "PreToolUse": [{ "hooks": [handler] }, { "hooks": [blocking] }] }),
                Path::new("/data/hooks"),
            )
            .unwrap();
            assert_eq!(matching_hooks(&hooks.pre_tool_use, "echo").len(), 2);
        }
    }

    #[test]
    fn test_http_header_variables_do_not_nest() {
        let lookup = |name: &str| (name == "TOKEN").then(|| "secret".to_string());
        let allowed = vec!["TOKEN".to_string()];
        assert_eq!(interpolate_header("${A${OTHER}}", &allowed, lookup), "${A}");
        assert_eq!(interpolate_header("$$TOKEN", &allowed, lookup), "$secret");
    }
}
