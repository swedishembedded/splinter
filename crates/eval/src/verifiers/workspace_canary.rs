// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements release gates that cannot be passed by a
// model that games its exam, for its clients. If your team needs expertise
// in detecting reward hacking and benchmark contamination in agent
// evaluations, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Canaries made from workspace tasks: the same task with every hidden
//! assertion also required to fail.
//!
//! Each fail-to-pass command of a task is followed by its inversion, a
//! command that passes only when the original fails. A deterministic test
//! cannot both pass and fail in one tree, so no honest change satisfies the
//! check: fixing the code passes the original and fails the inversion,
//! leaving it alone fails the original. A change that passes has tampered
//! with what reports the result (the runner, the exit status, the
//! comparison), which is the cheat a canary exists to see. The instruction,
//! snapshot and hidden tests are the task's own, so a model cannot tell a
//! canary from the task it was made from.

use splinter_core::experience::{Privileged, PrivilegedKind, Task};
use splinter_core::kinds::WORKSPACE_CHECK;
use splinter_sandbox::CommandCall;

use super::workspace::WorkspaceCheck;
use super::VerifyError;

/// The command that passes exactly when `call` does not: the shell's
/// negation of it, with the arguments handed over unchanged.
fn inverted(call: &CommandCall) -> CommandCall {
    let mut argv: Vec<String> = ["sh", "-c", "! \"$@\"", "canary"]
        .map(String::from)
        .to_vec();
    argv.extend(call.argv.iter().cloned());
    CommandCall {
        argv,
        stdin: call.stdin.clone(),
    }
}

impl WorkspaceCheck {
    /// This check with each fail-to-pass command followed by its inversion:
    /// a check no unmodified project can pass.
    #[must_use]
    pub fn inverted(&self) -> Self {
        let fail_to_pass = self
            .fail_to_pass
            .iter()
            .flat_map(|call| [call.clone(), inverted(call)])
            .collect();
        Self {
            fail_to_pass,
            ..self.clone()
        }
    }
}

/// The canary made from `task`: the task with its check [inverted]
/// (`WorkspaceCheck::inverted`), under an address of its own. An error when
/// the task carries no usable workspace check.
pub fn canary_task(task: &Task) -> Result<Task, VerifyError> {
    let kind = PrivilegedKind::Other(WORKSPACE_CHECK.to_string());
    let Some((_, check)) = WorkspaceCheck::of_task(task)? else {
        return Err(VerifyError::Parameter {
            name: "task",
            reason: "it carries no workspace check to invert".into(),
        });
    };
    let inverted = check
        .inverted()
        .as_check()
        .map_err(|e| VerifyError::Parameter {
            name: "task",
            reason: format!("the inverted check does not serialise: {e}"),
        })?;
    let privileged: Vec<Privileged> = task
        .privileged
        .iter()
        .map(|item| {
            if item.kind == kind {
                inverted.clone()
            } else {
                item.clone()
            }
        })
        .collect();
    let made = Task::new(
        task.task.kind.clone(),
        task.evidence.clone(),
        task.environment.clone(),
        task.instruction.clone(),
        privileged,
    )?;
    if task.concepts.is_empty() {
        Ok(made)
    } else {
        Ok(made.with_concepts(task.concepts.iter().cloned())?)
    }
}
