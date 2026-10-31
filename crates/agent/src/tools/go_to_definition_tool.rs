use std::fmt::Write;
use std::sync::Arc;

use super::symbol_locator::{LocationDisplay, SymbolLocator};
use crate::{AgentTool, ToolCallEventStream, ToolInput};
use agent_client_protocol::schema::v2 as acp;
use gpui::{App, Entity, SharedString, Task};
use project::Project;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Jumps to the definition of a symbol using the language server.
///
/// Returns the file path and line number of the symbol's definition, along with a snippet of the source code at that location.
///
/// Before using this tool, use read_file or grep to find the exact symbol name and line number of a usage you want to navigate from.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct GoToDefinitionToolInput {
    /// The symbol to find the definition of.
    pub symbol: SymbolLocator,
}

pub struct GoToDefinitionTool {
    project: Entity<Project>,
}

impl GoToDefinitionTool {
    pub fn new(project: Entity<Project>) -> Self {
        Self { project }
    }
}

impl AgentTool for GoToDefinitionTool {
    type Input = GoToDefinitionToolInput;
    type Output = String;

    const NAME: &'static str = "go_to_definition";

    fn kind() -> acp::ToolKind {
        acp::ToolKind::Search
    }

    fn initial_title(
        &self,
        input: Result<Self::Input, serde_json::Value>,
        _cx: &mut App,
    ) -> SharedString {
        if let Ok(input) = input {
            format!("Go to definition of `{}`", input.symbol.symbol_name).into()
        } else {
            "Go to definition".into()
        }
    }

    fn run(
        self: Arc<Self>,
        input: ToolInput<Self::Input>,
        _event_stream: ToolCallEventStream,
        cx: &mut App,
    ) -> Task<Result<String, String>> {
        let project = self.project.clone();
        cx.spawn(async move |cx| {
            let input = input
                .recv()
                .await
                .map_err(|e| format!("Failed to receive tool input: {e}"))?;

            let resolved = input
                .symbol
                .resolve(&project, |server| server.definition_provider.is_some(), cx)
                .await?;

            let definitions_task = project.update(cx, |project, cx| {
                project.definitions(&resolved.buffer, resolved.position, cx)
            });

            let definitions = definitions_task
                .await
                .map_err(|e| format!("Go to definition failed: {e}"))?
                .unwrap_or_default();

            if definitions.is_empty() {
                return Ok(format!(
                    "No definition found for '{}'.",
                    input.symbol.symbol_name
                ));
            }

            let mut output = String::new();

            if definitions.len() == 1 {
                write!(output, "Definition of `{}`:\n", input.symbol.symbol_name).ok();
            } else {
                write!(
                    output,
                    "Found {} definitions of `{}`:\n",
                    definitions.len(),
                    input.symbol.symbol_name
                )
                .ok();
            }

            for link in &definitions {
                let display = link
                    .target
                    .buffer
                    .read_with(cx, |_, cx| LocationDisplay::from_location(&link.target, cx));
                write!(output, "\n## {display}\n").ok();
            }

            Ok(output)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::symbol_locator::SymbolLocator;
    use gpui::TestAppContext;
    use project::{FakeFs, Project};
    use serde_json::json;
    use settings::SettingsStore;
    use std::time::Duration;
    use util::path;

    #[gpui::test]
    async fn test_go_to_definition_starts_the_language_server(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
        });
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            path!("/root"),
            json!({ "main.rs": "fn main() { helper(); }\nfn helper() {}\n" }),
        )
        .await;
        let project = Project::test(fs, [path!("/root").as_ref()], cx).await;
        let language_registry = project.read_with(cx, |project, _| project.languages().clone());
        language_registry.add(language::rust_lang());
        let mut fake_servers = language_registry.register_fake_lsp(
            "Rust",
            language::FakeLspAdapter {
                capabilities: lsp::ServerCapabilities {
                    definition_provider: Some(lsp::OneOf::Left(true)),
                    ..Default::default()
                },
                initializer: Some(Box::new(|server| {
                    server.set_request_handler::<lsp::request::GotoDefinition, _, _>(
                        |params, _| async move {
                            Ok(Some(lsp::GotoDefinitionResponse::Scalar(
                                lsp::Location::new(
                                    params.text_document_position_params.text_document.uri,
                                    lsp::Range::new(
                                        lsp::Position::new(1, 3),
                                        lsp::Position::new(1, 9),
                                    ),
                                ),
                            )))
                        },
                    );
                })),
                ..Default::default()
            },
        );

        let tool = Arc::new(GoToDefinitionTool::new(project));
        let (event_stream, _events) = ToolCallEventStream::test();
        let input = GoToDefinitionToolInput {
            symbol: SymbolLocator {
                file_path: "root/main.rs".into(),
                line: 1,
                symbol_name: "helper".into(),
            },
        };
        let task = cx.update(|cx| tool.run(ToolInput::resolved(input), event_stream, cx));

        // The file was never opened in an editor: the tool itself must start its server.
        cx.run_until_parked();
        assert!(
            fake_servers.try_recv().is_ok(),
            "no language server started"
        );

        // The tool waits for the server to stay quiet before it asks.
        cx.executor().advance_clock(Duration::from_secs(2));
        let output = task.await.expect("go_to_definition failed");
        assert!(output.contains("Definition of `helper`"), "{output}");
        assert!(output.contains("main.rs#L2"), "{output}");
    }
}
