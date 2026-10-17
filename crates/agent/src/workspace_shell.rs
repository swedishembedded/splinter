// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements coding agents whose commands run confined
// to the tree they change, for its clients. If your team needs expertise in
// containing what a language-model agent can run then you can procure our
// services by sending an email to info@swedishembedded.com.

//! `shell` for a workspace whose sandbox confines what it runs: a sven tool
//! that sends the model's command to [`Sandbox::run_command`] in the
//! episode's tree instead of running it on the host.
//!
//! sven's own `shell` starts in the project root but is not confined: the
//! command can read any path the process can, including material a task
//! withholds from its solver. This tool has the same name and takes the
//! same command, so the model sees one `shell`; sven lets a tool given to
//! the engine replace the built-in of that name. It is always used: a
//! container confines the command, and the process sandbox confines it to
//! its tree with the kernel's Landlock (or refuses to run it).

use std::path::PathBuf;

use splinter_sandbox::{CodeResult, CommandCall, WorkspaceEnvironment};
use sven_sdk::tool::{ApprovalPolicy, OutputCategory, Tool, ToolCall, ToolCapability, ToolOutput};

/// The most bytes of a command's output the model is shown. The sandbox may
/// keep far more (a test run's whole report); the model reads the head and
/// the tail of it, which is where a command says what it did and how it ended.
const MODEL_VIEW_BYTES: usize = 32 * 1024;

/// The tool's name: the one of sven's built-in shell, which it replaces.
pub const SHELL: &str = "shell";

/// Runs the model's commands in one tree, in its environment's sandbox.
pub struct SandboxShell {
    environment: WorkspaceEnvironment,
    tree: PathBuf,
}

impl SandboxShell {
    /// The tool running commands in `tree`, which `environment`'s sandbox
    /// mounts as the only place a command can keep files.
    #[must_use]
    pub fn new(environment: WorkspaceEnvironment, tree: PathBuf) -> Self {
        Self { environment, tree }
    }
}

#[async_trait::async_trait]
impl Tool for SandboxShell {
    fn name(&self) -> &str {
        SHELL
    }

    fn description(&self) -> &str {
        "Run a bash command in the project directory; returns stdout, stderr and how \
         the command ended.\n\
         'description' is one short line naming what the command does.\n\
         Non-interactive only. The command runs confined to the project directory (the only place it can \
         write); nothing else of the disk is visible to it; it is stopped at a time limit. Narrow big \
         output at the source (`tail -200`, `grep -E 'error:'`)."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Short description of what this command does."
                },
                "shell_command": {
                    "type": "string",
                    "description": "The complete bash one liner shell command to execute."
                }
            },
            "required": ["shell_command"],
            "additionalProperties": false
        })
    }

    fn default_policy(&self) -> ApprovalPolicy {
        ApprovalPolicy::Ask
    }

    fn output_category(&self) -> OutputCategory {
        OutputCategory::HeadTail
    }

    fn kernel_capability(&self) -> ToolCapability {
        ToolCapability::ExecuteShell
    }

    async fn execute(&self, call: &ToolCall) -> ToolOutput {
        let Some(command) = call.args.get("shell_command").and_then(|v| v.as_str()) else {
            return ToolOutput::err(
                &call.id,
                "Please provide a shell command to execute as the 'shell_command' parameter.",
            );
        };
        let command_call = CommandCall {
            argv: vec!["bash".into(), "-c".into(), command.into()],
            stdin: None,
        };
        let (environment, tree) = (self.environment.clone(), self.tree.clone());
        // The sandbox blocks until the command ends or a limit stops it; if
        // the run is cancelled meanwhile, the command still ends at its own
        // limits.
        let ran = tokio::task::spawn_blocking(move || environment.run(&tree, &command_call)).await;
        match ran {
            Ok(Ok(result)) => ToolOutput::ok(&call.id, render(&result)),
            Ok(Err(e)) => ToolOutput::err(&call.id, format!("{SHELL} could not run: {e}")),
            Err(e) => ToolOutput::err(&call.id, format!("{SHELL} failed: {e}")),
        }
    }
}

/// The result as the model reads it: both streams, then how the command
/// ended when it did not simply succeed.
fn render(result: &CodeResult) -> String {
    let mut out = result.stdout.clone();
    if !result.stderr.is_empty() {
        if !out.is_empty() && !out.ends_with('\n') {
            out.push('\n');
        }
        out.push_str(&result.stderr);
    }
    let mut out = clip_middle(out, MODEL_VIEW_BYTES);
    if result.stdout_truncated || result.stderr_truncated {
        out.push_str("\n[output truncated]");
    }
    if result.timed_out {
        out.push_str("\n[stopped at the time limit]");
    } else if let Some(signal) = result.signal {
        out.push_str(&format!("\n[killed by signal {signal}]"));
    } else if let Some(code) = result.exit_code.filter(|c| *c != 0) {
        out.push_str(&format!("\n[exit code {code}]"));
    }
    out
}

/// `text` cut to about `max` bytes by dropping a middle part, at character
/// boundaries, and saying so.
fn clip_middle(text: String, max: usize) -> String {
    if text.len() <= max {
        return text;
    }
    let mut head = max / 2;
    while !text.is_char_boundary(head) {
        head -= 1;
    }
    let mut tail = text.len() - (max - max / 2);
    while !text.is_char_boundary(tail) {
        tail += 1;
    }
    format!(
        "{}\n[... {} bytes not shown ...]\n{}",
        &text[..head],
        tail - head,
        &text[tail..]
    )
}

#[cfg(test)]
mod tests {
    use super::clip_middle;

    #[test]
    fn a_long_output_shows_its_head_and_tail() {
        let text = format!("first{}last", "é".repeat(100));
        let clipped = clip_middle(text.clone(), 40);
        assert!(clipped.starts_with("first") && clipped.ends_with("last"));
        assert!(clipped.contains("bytes not shown") && clipped.len() < 100);
        assert_eq!(clip_middle("short".into(), 40), "short");
    }
}
