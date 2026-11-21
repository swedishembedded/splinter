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

use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

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
    /// The longest one exchange may take; past it the server is killed and
    /// the exchange fails as an infrastructure error.
    pub timeout: Duration,
    /// The first infrastructure failure of any exchange, shared by every
    /// copy of the server: a fork that ran its tools is told, and fails
    /// instead of passing a dead tool off as the model's observation.
    pub failure: Arc<Mutex<Option<String>>>,
}

/// How an infrastructure failure of an exchange starts: the server, its
/// launcher or the environment failed, never the model.
pub const INFRASTRUCTURE: &str = "infrastructure: ";

/// The longest one exchange with a server takes by default: far longer than
/// a tool call that runs one program, far shorter than a repair's deadline.
pub const DEFAULT_EXCHANGE_TIMEOUT: Duration = Duration::from_secs(300);

impl StdioServer {
    /// The server's answer to `request` (its result), the server started for
    /// this one exchange.
    ///
    /// # Errors
    /// The server cannot be started, answers nothing for the request, or
    /// answers with an error.
    pub fn ask(&self, method: &str, params: &Value) -> Result<Value, String> {
        self.exchange(method, params).inspect_err(|why| {
            if why.starts_with(INFRASTRUCTURE) {
                if let Ok(mut failure) = self.failure.lock() {
                    failure.get_or_insert_with(|| why.clone());
                }
            }
        })
    }

    /// The first infrastructure failure of an exchange with this server.
    #[must_use]
    pub fn failure(&self) -> Option<String> {
        self.failure.lock().ok().and_then(|f| f.clone())
    }

    fn exchange(&self, method: &str, params: &Value) -> Result<Value, String> {
        let (program, args) = self.argv.split_first().ok_or_else(|| {
            format!(
                "{INFRASTRUCTURE}the MCP server {} names no command",
                self.name
            )
        })?;
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
            .map_err(|e| format!("{INFRASTRUCTURE}starting the MCP server {}: {e}", self.name))?;
        let requests = [
            json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
                "protocolVersion": "2024-11-05", "capabilities": {},
                "clientInfo": {"name": "splinter", "version": "1"}}}),
            json!({"jsonrpc": "2.0", "method": "notifications/initialized"}),
            json!({"jsonrpc": "2.0", "id": 2, "method": method, "params": params}),
        ];
        // Its output is read beside the exchange, so a server that never
        // answers (or never reads) is killed at the bound, not waited for.
        let (sent, received) = mpsc::channel();
        let mut stdout = child.stdout.take();
        let mut stderr = child.stderr.take();
        std::thread::spawn(move || {
            let errors = std::thread::spawn(move || {
                let mut err = Vec::new();
                if let Some(stderr) = stderr.as_mut() {
                    let _ = stderr.read_to_end(&mut err);
                }
                err
            });
            let mut out = Vec::new();
            if let Some(stdout) = stdout.as_mut() {
                let _ = stdout.read_to_end(&mut out);
            }
            let _ = sent.send((out, errors.join().unwrap_or_default()));
        });
        if let Some(mut stdin) = child.stdin.take() {
            let lines: String = requests.iter().map(|r| format!("{r}\n")).collect();
            std::thread::spawn(move || {
                let _ = stdin.write_all(lines.as_bytes());
            });
        }
        let Ok((out, err)) = received.recv_timeout(self.timeout) else {
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!(
                "{INFRASTRUCTURE}the MCP server {} did not answer within {} s and was stopped",
                self.name,
                self.timeout.as_secs()
            ));
        };
        let _ = child.wait();
        let reply = String::from_utf8_lossy(&out)
            .lines()
            .filter_map(|line| serde_json::from_str::<Value>(line).ok())
            .find(|reply| reply.get("id") == Some(&json!(2)))
            .ok_or_else(|| {
                format!(
                    "{INFRASTRUCTURE}the MCP server {} gave no answer: {}",
                    self.name,
                    String::from_utf8_lossy(&err).trim()
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

/// Sven's own argument every tool is offered (what the call is for); it is
/// never the tool's unless the tool declares a parameter of that name.
const INTENT: &str = "intent";

/// `arguments` as the server takes them: without Sven's `intent`, unless the
/// tool's `schema` declares it. A replayed call carries the arguments the
/// model sent, `intent` included.
fn own_arguments(arguments: &Value, schema: &Value) -> Value {
    let declared = schema.pointer("/properties/intent").is_some();
    match arguments {
        Value::Object(map) if !declared && map.contains_key(INTENT) => {
            let mut map = map.clone();
            map.remove(INTENT);
            Value::Object(map)
        }
        other => other.clone(),
    }
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
        let params =
            json!({"name": self.tool, "arguments": own_arguments(&call.args, &self.schema)});
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

    /// Two calls of one turn, run at once, each through an environment that
    /// runs one command at a time (as an examiner's session does): a call's
    /// input must end when its writer is done, whatever else was started
    /// meanwhile, or the first waits for an end of input that the second,
    /// queued behind it, holds open.
    #[test]
    fn two_calls_at_once_through_a_one_at_a_time_environment_both_finish() {
        let scratch = tempfile::tempdir().unwrap();
        let lock = scratch.path().join("session.lock");
        let script = "import json,sys\nlines=sys.stdin.read().splitlines()\nfor l in lines:\n    r=json.loads(l)\n    if 'id' in r: print(json.dumps({'jsonrpc':'2.0','id':r['id'],'result':{'content':[{'type':'text','text':'ok'}]}}),flush=True)\n";
        let server = StdioServer {
            name: "files".into(),
            argv: vec![
                "flock".into(),
                lock.to_string_lossy().into_owned(),
                "python3".into(),
                "-c".into(),
                script.into(),
            ],
            cwd: None,
            timeout: std::time::Duration::from_secs(20),
            failure: Arc::default(),
        };
        let started = std::time::Instant::now();
        let calls: Vec<_> = (0..2)
            .map(|_| {
                let server = server.clone();
                std::thread::spawn(move || {
                    server.ask("tools/call", &json!({"name": "write", "arguments": {}}))
                })
            })
            .collect();
        for call in calls {
            let answered = call.join().unwrap().unwrap();
            assert_eq!(result_text(&answered), "ok");
        }
        assert!(
            started.elapsed() < std::time::Duration::from_secs(10),
            "{:?}",
            started.elapsed()
        );
    }

    #[test]
    fn a_server_that_never_answers_fails_within_its_bound_as_infrastructure() {
        let server = StdioServer {
            name: "silent".into(),
            argv: vec!["sleep".into(), "60".into()],
            cwd: None,
            timeout: std::time::Duration::from_secs(1),
            failure: Arc::default(),
        };
        let started = std::time::Instant::now();
        let refused = server.ask("tools/list", &json!({})).unwrap_err();
        assert!(
            started.elapsed() < std::time::Duration::from_secs(10),
            "{:?}",
            started.elapsed()
        );
        assert!(refused.starts_with(INFRASTRUCTURE), "{refused}");
        assert_eq!(
            server.failure().as_deref(),
            Some(refused.as_str()),
            "the fork is told"
        );
    }

    #[test]
    fn svens_intent_never_reaches_a_tool_that_does_not_declare_it() {
        let call = json!({"commands": ["wq"], "intent": "save"});
        let plain = json!({"type": "object", "properties": {"commands": {"type": "array"}}});
        assert_eq!(own_arguments(&call, &plain), json!({"commands": ["wq"]}));
        let declares = json!({"type": "object", "properties": {"intent": {"type": "string"}}});
        assert_eq!(own_arguments(&call, &declares), call);
    }

    #[test]
    fn a_results_text_is_its_text_parts() {
        let result = json!({"content": [{"type": "text", "text": "a"}, {"type": "image"}, {"type": "text", "text": "b"}]});
        assert_eq!(result_text(&result), "a\nb");
    }
}
