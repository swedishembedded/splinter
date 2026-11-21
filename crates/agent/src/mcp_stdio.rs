// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for continuing a recorded agent run
// with exactly the tools its environment gave it, for its clients. If your
// team needs expertise in agent tool environments then you can procure our
// services by sending an email to info@swedishembedded.com.

//! The tools of a stdio MCP server, as tools of an engine.
//!
//! A recorded attempt's environment may give its agent its tools as MCP
//! servers (a task image's Sven configuration, `mcp_servers`). A fork of the
//! attempt must offer the same tools, named as Sven names them
//! (`<server>-<tool>`), with the server's own schemas, and run them where the
//! attempt's ran: the server's command is started through the environment's
//! launcher (an examiner's `session-exec --stdin`). The server is asked for
//! its tools once; each call then starts it afresh, initializes it and sends
//! the one call, so a call never holds the environment between calls (an
//! examiner runs one command at a time) and its state is the environment's
//! (the files the call changed), as for a fresh agent process.

use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::Arc;

use serde_json::{json, Value};
use sven_sdk::tool::{ApprovalPolicy, Tool, ToolCall, ToolCapability, ToolOutput};

/// One stdio MCP server: the program and arguments that start it.
#[derive(Clone, Debug)]
pub struct StdioServer {
    /// Its name, which prefixes its tools' names.
    pub name: String,
    /// The program and its arguments.
    pub argv: Vec<String>,
    /// The directory it is started in: the project's root, where the
    /// agent's own process would start it (an examiner's launcher maps it to
    /// the same place in its sandbox).
    pub cwd: Option<std::path::PathBuf>,
}

impl StdioServer {
    /// The server's answer to `request` (its result), the server started for
    /// this one exchange.
    ///
    /// # Errors
    /// The server cannot be started, answers nothing for the request, or
    /// answers with an error.
    pub fn ask(&self, method: &str, params: &Value) -> Result<Value, String> {
        let (program, args) = self
            .argv
            .split_first()
            .ok_or_else(|| format!("the MCP server {} names no command", self.name))?;
        let mut command = Command::new(program);
        if let Some(cwd) = &self.cwd {
            command.current_dir(cwd);
        }
        let mut child = command
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("starting the MCP server {}: {e}", self.name))?;
        let requests = [
            json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
                "protocolVersion": "2024-11-05", "capabilities": {},
                "clientInfo": {"name": "splinter", "version": "1"}}}),
            json!({"jsonrpc": "2.0", "method": "notifications/initialized"}),
            json!({"jsonrpc": "2.0", "id": 2, "method": method, "params": params}),
        ];
        {
            let mut stdin = child.stdin.take().ok_or("the MCP server has no input")?;
            for request in &requests {
                writeln!(stdin, "{request}")
                    .map_err(|e| format!("asking the MCP server {}: {e}", self.name))?;
            }
        }
        let done = child
            .wait_with_output()
            .map_err(|e| format!("the MCP server {}: {e}", self.name))?;
        let reply = String::from_utf8_lossy(&done.stdout)
            .lines()
            .filter_map(|line| serde_json::from_str::<Value>(line).ok())
            .find(|reply| reply.get("id") == Some(&json!(2)))
            .ok_or_else(|| {
                format!(
                    "the MCP server {} gave no answer: {}",
                    self.name,
                    String::from_utf8_lossy(&done.stderr).trim()
                )
            })?;
        if let Some(error) = reply.get("error") {
            return Err(format!(
                "the MCP server {} answered an error: {error}",
                self.name
            ));
        }
        Ok(reply.get("result").cloned().unwrap_or(Value::Null))
    }

    /// The server's tools, as engine tools.
    ///
    /// # Errors
    /// As [`Self::ask`].
    pub fn tools(&self) -> Result<Vec<Arc<dyn Tool>>, String> {
        let listed = self.ask("tools/list", &json!({}))?;
        let server = Arc::new(self.clone());
        Ok(listed
            .get("tools")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|tool| {
                let name = tool.get("name")?.as_str()?.to_string();
                Some(Arc::new(McpTool {
                    name: format!("{}-{name}", self.name),
                    tool: name,
                    description: tool
                        .get("description")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    schema: tool
                        .get("inputSchema")
                        .cloned()
                        .unwrap_or_else(|| json!({"type": "object"})),
                    server: Arc::clone(&server),
                }) as Arc<dyn Tool>)
            })
            .collect())
    }
}

/// The text a tool call's result shows: its text parts, one per line.
#[must_use]
pub fn result_text(result: &Value) -> String {
    result
        .get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|part| part.get("type").and_then(Value::as_str) == Some("text"))
        .filter_map(|part| part.get("text").and_then(Value::as_str))
        .collect::<Vec<_>>()
        .join("\n")
}

struct McpTool {
    /// As Sven names it: `<server>-<tool>`.
    name: String,
    /// As the server names it.
    tool: String,
    description: String,
    schema: Value,
    server: Arc<StdioServer>,
}

#[async_trait::async_trait]
impl Tool for McpTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters_schema(&self) -> Value {
        self.schema.clone()
    }

    fn default_policy(&self) -> ApprovalPolicy {
        ApprovalPolicy::Auto
    }

    fn kernel_capability(&self) -> ToolCapability {
        ToolCapability::ExecuteShell
    }

    async fn execute(&self, call: &ToolCall) -> ToolOutput {
        let server = Arc::clone(&self.server);
        let params = json!({"name": self.tool, "arguments": call.args});
        let answered = tokio::task::spawn_blocking(move || server.ask("tools/call", &params)).await;
        match answered {
            Ok(Ok(result)) => {
                let text = result_text(&result);
                if result.get("isError").and_then(Value::as_bool) == Some(true) {
                    ToolOutput::err(&call.id, text)
                } else {
                    ToolOutput::ok(&call.id, text)
                }
            }
            Ok(Err(why)) => ToolOutput::err(&call.id, why),
            Err(e) => ToolOutput::err(&call.id, format!("{} failed: {e}", self.name)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_results_text_is_its_text_parts() {
        let result = json!({"content": [{"type": "text", "text": "a"}, {"type": "image"}, {"type": "text", "text": "b"}]});
        assert_eq!(result_text(&result), "a\nb");
    }
}
