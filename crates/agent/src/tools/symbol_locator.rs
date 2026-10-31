use std::collections::VecDeque;
use std::fmt;
use std::time::Duration;

use futures::{FutureExt as _, StreamExt as _, channel::mpsc};
use gpui::{App, AsyncApp, Entity};
use language::{Buffer, Location};
use project::{CodeAction, Project, lsp_store::OpenLspBufferHandle};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use text::ToPoint as _;
use text::{Anchor, Point};

/// Identifies a specific symbol (declaration or usage) in the source code.
///
/// Use the file path, line number, and symbol name from file outlines, grep results, or other tool outputs to populate these fields.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct SymbolLocator {
    /// The relative path of the file containing the symbol (e.g. "crates/editor/src/editor.rs").
    pub file_path: String,

    /// The 1-based line number where the symbol appears. Use the line numbers from file outlines or grep results.
    pub line: u32,

    /// The name of the symbol (function name, type name, variable name, etc.)
    pub symbol_name: String,
}

pub struct PendingCodeActions {
    pub actions: Vec<CodeAction>,
    pub buffer: Entity<Buffer>,
}

pub type CodeActionStore = Entity<Option<PendingCodeActions>>;

pub struct ResolvedSymbol {
    pub buffer: Entity<Buffer>,
    pub position: Anchor,
    pub line_text: String,
    pub truncated: bool,
    /// Keeps the buffer open in its language servers while the symbol is in use.
    _lsp_handle: OpenLspBufferHandle,
}

/// How long a tool waits for the language servers of the file it opened.
const LANGUAGE_SERVERS_TIMEOUT: Duration = Duration::from_secs(120);

/// How long a ready server must stay quiet: a server starts loading its project shortly after it
/// opens the file, and answers for that file only until it has loaded it.
const LANGUAGE_SERVERS_SETTLE: Duration = Duration::from_secs(1);

pub const MAX_LINE_DISPLAY_LEN: usize = 200;

pub struct LocationDisplay {
    pub path: String,
    pub start_line: u32,
    pub end_line: u32,
    pub snippet: String,
    pub truncated: bool,
}

impl LocationDisplay {
    pub fn from_location(location: &Location, cx: &App) -> Self {
        let snapshot = location.buffer.read(cx).snapshot();
        let range =
            location.range.start.to_point(&snapshot)..location.range.end.to_point(&snapshot);
        let path = location
            .buffer
            .read(cx)
            .file()
            .map(|f| f.full_path(cx).display().to_string())
            .unwrap_or_else(|| "<untitled>".to_string());

        let start_line = range.start.row + 1;
        let end_line = range.end.row + 1;

        let line_len = snapshot.line_len(range.start.row);
        let truncated = line_len as usize > MAX_LINE_DISPLAY_LEN;
        let snippet: String = snapshot
            .text_for_range(Point::new(range.start.row, 0)..Point::new(range.start.row, line_len))
            .flat_map(|chunk| chunk.chars())
            .skip_while(|c| c.is_whitespace())
            .take(MAX_LINE_DISPLAY_LEN)
            .collect::<String>();
        let snippet = snippet.trim_end().to_string();

        Self {
            path,
            start_line,
            end_line,
            snippet,
            truncated,
        }
    }
}

impl fmt::Display for LocationDisplay {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let truncated_label = if self.truncated { " (truncated)" } else { "" };
        if self.start_line == self.end_line {
            writeln!(f, "{}#L{}{truncated_label}", self.path, self.start_line)?;
        } else {
            writeln!(
                f,
                "{}#L{}-{}{truncated_label}",
                self.path, self.start_line, self.end_line
            )?;
        }
        writeln!(f, "```")?;
        writeln!(f, "{}", self.snippet)?;
        write!(f, "```")
    }
}

/// Searches for `needle` in a char iterator, returning the byte offset of the
/// first occurrence without collecting the full iterator into a string.
///
/// Equivalent to [`str::find`]
fn find_in_char_iter(chars: impl Iterator<Item = char>, needle: &str) -> Option<usize> {
    let needle_chars: Vec<char> = needle.chars().collect();
    if needle_chars.is_empty() {
        return Some(0);
    }

    let mut window: VecDeque<char> = VecDeque::with_capacity(needle_chars.len());
    let mut byte_offsets: VecDeque<usize> = VecDeque::with_capacity(needle_chars.len());
    let mut byte_offset = 0usize;

    for ch in chars {
        window.push_back(ch);
        byte_offsets.push_back(byte_offset);
        byte_offset += ch.len_utf8();

        if window.len() > needle_chars.len() {
            window.pop_front();
            byte_offsets.pop_front();
        }

        if window.len() == needle_chars.len()
            && window.iter().zip(needle_chars.iter()).all(|(a, b)| a == b)
        {
            return byte_offsets.front().copied();
        }
    }

    None
}

impl SymbolLocator {
    /// Resolves this locator into a concrete buffer and position.
    ///
    /// Opens the file at `file_path`, then searches for `symbol_name` on the
    /// specified `line`. Returns an error if the file can't be found, the line
    /// is out of range, or the symbol name doesn't appear on that line.
    /// If the symbol name appears multiple times on the line, uses the first
    /// occurrence.
    pub async fn resolve(
        &self,
        project: &Entity<Project>,
        serves: fn(&lsp::ServerCapabilities) -> bool,
        cx: &mut AsyncApp,
    ) -> Result<ResolvedSymbol, String> {
        let Self {
            file_path,
            line,
            symbol_name,
        } = self;

        let open_buffer_task = project.update(cx, |project, cx| {
            let Some(project_path) = project.find_project_path(file_path, cx) else {
                return Err(format!("Could not find path '{file_path}' in project",));
            };
            Ok(project.open_buffer(project_path, cx))
        })?;

        let buffer = open_buffer_task
            .await
            .map_err(|e| format!("Failed to open '{}': {e}", self.file_path))?;

        // A file no editor has open is known to no language server: register it, and wait for
        // a server that `serves` the request, since a request only reaches the servers already
        // running with the file open.
        let lsp_handle = project.update(cx, |project, cx| {
            project.register_buffer_with_language_servers(&buffer, cx)
        });
        wait_for_language_servers(project, &buffer, serves, cx).await;

        let (position, line_text, truncated) = buffer.read_with(cx, |buffer, _cx| {
            let snapshot = buffer.snapshot();
            let row = line.saturating_sub(1);

            if row > snapshot.max_point().row {
                let line_count = snapshot.max_point().row + 1;
                return Err(format!(
                    "Line {line} is beyond the end of '{file_path}' (file has {line_count} lines)",
                ));
            }

            let line_len = snapshot.line_len(row);
            let truncated = line_len as usize > MAX_LINE_DISPLAY_LEN;
            let line_start = Point::new(row, 0);
            let line_end = Point::new(row, line_len);
            let line_chars = || {
                snapshot
                    .text_for_range(line_start..line_end)
                    .flat_map(|chunk| chunk.chars())
            };

            let byte_offset = find_in_char_iter(line_chars(), symbol_name).ok_or_else(|| {
                let preview: String = line_chars()
                    .skip_while(|c| c.is_whitespace())
                    .take(MAX_LINE_DISPLAY_LEN)
                    .collect();
                format!(
                    "Symbol '{symbol_name}' not found on line {line} of '{file_path}'. \
                     Line content: '{}'",
                    preview.trim_end()
                )
            })?;

            let position = snapshot.anchor_before(Point::new(row, byte_offset as u32));
            let display_text: String = line_chars()
                .skip_while(|c| c.is_whitespace())
                .take(MAX_LINE_DISPLAY_LEN)
                .collect::<String>();
            let display_text = display_text.trim_end().to_string();

            Ok((position, display_text, truncated))
        })?;

        Ok(ResolvedSymbol {
            buffer,
            position,
            line_text,
            truncated,
            _lsp_handle: lsp_handle,
        })
    }
}

/// Waits until the buffer has a language and one of its language servers that `serves` the
/// request runs with the buffer open and no work in progress (a server still loading its project
/// answers for the open files only), and stays so for [`LANGUAGE_SERVERS_SETTLE`]. A server whose
/// startup failed stays "starting", so the wait does not expect all of them. Gives up after
/// [`LANGUAGE_SERVERS_TIMEOUT`].
async fn wait_for_language_servers(
    project: &Entity<Project>,
    buffer: &Entity<Buffer>,
    serves: fn(&lsp::ServerCapabilities) -> bool,
    cx: &mut AsyncApp,
) {
    let lsp_store = project.read_with(cx, |project, _| project.lsp_store());
    let (changed_tx, mut changed_rx) = mpsc::unbounded();
    let _subscriptions = cx.update(|cx| {
        let buffer_changed = changed_tx.clone();
        [
            cx.subscribe(&lsp_store, move |_, _, _| {
                changed_tx.unbounded_send(()).ok();
            }),
            cx.subscribe(buffer, move |_, _, _| {
                buffer_changed.unbounded_send(()).ok();
            }),
        ]
    });
    let mut timeout = cx
        .background_executor()
        .timer(LANGUAGE_SERVERS_TIMEOUT)
        .fuse();
    loop {
        let ready = buffer.update(cx, |buffer, cx| {
            buffer.language().is_some()
                && lsp_store.update(cx, |lsp_store, cx| {
                    if lsp_store
                        .language_servers_for_local_buffer(buffer, cx)
                        .is_empty()
                    {
                        return true;
                    }
                    let opened =
                        lsp_store.language_server_ids_for_opened_buffer(buffer.remote_id());
                    lsp_store
                        .running_language_servers_for_local_buffer(buffer, cx)
                        .any(|(_, server)| {
                            let id = server.server_id();
                            opened.is_some_and(|opened| opened.contains(&id))
                                && serves(&server.capabilities())
                                && lsp_store
                                    .language_server_statuses
                                    .get(&id)
                                    .is_some_and(|status| status.pending_work.is_empty())
                        })
                })
        });
        let mut settled = if ready {
            cx.background_executor()
                .timer(LANGUAGE_SERVERS_SETTLE)
                .fuse()
        } else {
            futures::future::Fuse::terminated()
        };
        futures::select_biased! {
            _ = changed_rx.next() => {}
            _ = settled => return,
            _ = timeout => return,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::proptest::prelude::*;

    #[gpui::property_test]
    fn find_in_char_iter_test(
        // limited character sets to increase odds of finding matches
        #[strategy = "[abcd]{100,1000}"] haystack: String,
        #[strategy = "[abcd]{1,5}"] needle: String,
    ) -> Result<(), TestCaseError> {
        let expected = haystack.find(&needle);
        let actual = find_in_char_iter(haystack.chars(), &needle);
        prop_assert_eq!(actual, expected);
        Ok::<_, TestCaseError>(())
    }
}
