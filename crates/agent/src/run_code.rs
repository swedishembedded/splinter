// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agent tools that run model-written code
// under enforced limits, for its clients. If your team needs expertise in
// agent tooling or sandboxing, you can procure our services by sending an
// email to info@swedishembedded.com.

//! `run_code`: the one tool of a runtime environment, a sven tool over a
//! [`RuntimeEnvironment`].
//!
//! The model sends `{code, stdin?}`; the code runs in the environment's
//! runtime and sandbox, and the model receives the [`CodeResult`] as JSON:
//! standard output and error, the exit code (and signal), whether a limit
//! stopped it, and whether either stream was truncated. A call the sandbox
//! could not carry out is a tool error naming why.
//!
//! The tool declares [`ToolCapability::ExecuteShell`], which is what it
//! does, so sven's permission gate asks before it runs; the solver answers
//! that gate (see [`crate::solve`]).

use sven_sdk::tool::{ApprovalPolicy, Tool, ToolCall, ToolCapability, ToolOutput};

use splinter_sandbox::{CodeCall, CodeResult, RuntimeEnvironment};

/// The tool's name, as the model calls it.
pub const RUN_CODE: &str = "run_code";

/// Runs the model's code in one runtime environment.
pub struct RunCode {
    environment: RuntimeEnvironment,
    description: String,
}

impl RunCode {
    /// The tool over `environment`.
    #[must_use]
    pub fn new(environment: RuntimeEnvironment) -> Self {
        let runtime = environment.runtime();
        let version = runtime
            .version
            .as_deref()
            .map_or_else(String::new, |v| format!(" ({v})"));
        let description = format!(
            "Run a {name}{version} program and return its stdout, stderr, exit_code, \
             signal, timed_out (a time limit stopped it) and stdout_truncated / \
             stderr_truncated. `code` is the whole program; `stdin`, if given, is its \
             standard input. Each call starts from a fresh, empty working directory.",
            name = runtime.spec.name,
        );
        Self {
            environment,
            description,
        }
    }
}

#[async_trait::async_trait]
impl Tool for RunCode {
    fn name(&self) -> &str {
        RUN_CODE
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "The program to run."},
                "stdin": {"type": "string", "description": "Its standard input."}
            },
            "required": ["code"],
            "additionalProperties": false
        })
    }

    fn default_policy(&self) -> ApprovalPolicy {
        ApprovalPolicy::Ask
    }

    fn kernel_capability(&self) -> ToolCapability {
        ToolCapability::ExecuteShell
    }

    async fn execute(&self, call: &ToolCall) -> ToolOutput {
        let code_call: CodeCall = match serde_json::from_value(call.args.clone()) {
            Ok(c) => c,
            Err(e) => {
                return ToolOutput::err(
                    &call.id,
                    format!("{RUN_CODE} takes {{code: string, stdin?: string}}: {e}"),
                )
            }
        };
        let environment = self.environment.clone();
        // The sandbox blocks until the call ends or a limit stops it; if the
        // run is cancelled meanwhile, the call still ends at its own limits.
        let ran = tokio::task::spawn_blocking(move || environment.run(&code_call)).await;
        match ran {
            Ok(Ok(result)) => ToolOutput::ok(&call.id, render(&result)),
            Ok(Err(e)) => ToolOutput::err(&call.id, format!("{RUN_CODE} could not run: {e}")),
            Err(e) => ToolOutput::err(&call.id, format!("{RUN_CODE} failed: {e}")),
        }
    }
}

/// The result as the model reads it.
fn render(result: &CodeResult) -> String {
    serde_json::to_string(result).unwrap_or_else(|e| {
        // Strings, integers and booleans always serialize.
        unreachable!("a code result serializes: {e}")
    })
}
