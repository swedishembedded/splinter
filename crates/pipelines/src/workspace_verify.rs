// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements audits that prove an agent never saw the
// material it is graded against, for its clients. If your team needs
// expertise in contamination-free evaluation of code-editing agents, you
// can procure our services by sending an email to info@swedishembedded.com.

//! The verifiers that grade a change to a repository: the workspace
//! verifier, which applies the experience's patch to a fresh copy of the
//! task's snapshot and runs the project's own checks, and the exposure audit
//! beside it.
//!
//! Where nothing stops a solver's shell from reaching what the grader keeps
//! (the process sandbox is bounded, not isolated), the trajectory is the
//! evidence of whether it did. The audit therefore runs for every
//! experience of a workspace task, whatever the sandbox: it fails an
//! experience whose tool calls or outputs name a hidden path, quote the
//! hidden tests, or name the state root. Such an experience is decided fail
//! whatever the workspace verifier said of its patch, so it is never trained
//! on and is counted by the verify stage.

use std::path::Path;
use std::sync::Arc;

use splinter_core::experience::Task;
use splinter_eval::exposure::{ExposureAudit, HiddenMaterial};
use splinter_eval::verifiers::workspace::{lay_over, WorkspaceCheck, WorkspaceVerifier};
use splinter_eval::verifiers::Verifier;
use splinter_sandbox::ResolvedEnvironment;

use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

/// The largest file whose text is held to audit for: a file larger than this
/// is not source text a solver would be shown whole.
const MAX_AUDITED_FILE_BYTES: u64 = 1024 * 1024;

/// The workspace verifier for `task` and the exposure audit of what its
/// solver was never to see, in the environment the task records.
pub(crate) fn verifiers(
    ctx: &Context,
    task: &Task,
) -> Result<Vec<Box<dyn Verifier>>, OrchestratorError> {
    let refused = |why: String| OrchestratorError::Refused(format!("task {}: {why}", task.task.id));
    let ResolvedEnvironment::Workspace(env) = ctx.environments().for_record(&task.environment)?
    else {
        return Err(refused("it is not in a workspace".into()));
    };
    let hidden = hidden_material(ctx, task, &env)?;
    let scratch = ctx.environments().workspace()?.scratch.join("verify");
    Ok(vec![
        Box::new(WorkspaceVerifier::new(
            vec![env],
            Arc::new(ctx.artifacts()),
            scratch,
        )),
        Box::new(ExposureAudit::new(hidden)),
    ])
}

/// What `task`'s solver must not have seen: the hidden tests' paths and
/// text, set against the text of the tree it was given (so what the two
/// share is not taken for exposure), and the path of the state root.
fn hidden_material(
    ctx: &Context,
    task: &Task,
    env: &splinter_sandbox::WorkspaceEnvironment,
) -> Result<HiddenMaterial, OrchestratorError> {
    let failed = |what: &str, e: &dyn std::fmt::Display| {
        OrchestratorError::Refused(format!("task {}: {what}: {e}", task.task.id))
    };
    let check = WorkspaceCheck::of_task(task)
        .map_err(|e| failed("the workspace check is unusable", &e))?
        .map(|(_, check)| check);
    let work = tempfile::Builder::new()
        .prefix("splinter-exposure-")
        .tempdir_in(ctx.environments().workspace()?.scratch)
        .map_err(|e| failed("no scratch directory for the audit", &e))?;

    let tree = work.path().join("tree");
    let snapshot = &env.snapshot().digest;
    std::fs::create_dir(&tree).map_err(|e| failed("no scratch directory for the audit", &e))?;
    let archive = ctx
        .artifacts()
        .path(snapshot)
        .map_err(|e| failed("the task's snapshot is not kept", &e))?;
    lay_over(&archive, snapshot, &tree).map_err(|e| failed("laying out the snapshot", &e))?;

    let mut material = HiddenMaterial {
        forbidden: vec![ctx.root().path().to_string_lossy().into_owned()],
        public_contents: texts_under(&tree),
        confine_to_workspace: true,
        ..HiddenMaterial::default()
    };
    // The request itself is text the solver is shown by design: it may quote
    // the value a hidden assertion compares (the one disclosure a task makes
    // of its tests), and that is not a leak of the tests' code.
    material.public_contents.push(task.instruction.clone());
    let Some(check) = check else {
        return Ok(material);
    };
    material.paths = check.hidden_paths.clone();
    if let Some(overlay) = &check.hidden {
        let files = work.path().join("hidden");
        std::fs::create_dir(&files)
            .map_err(|e| failed("no scratch directory for the audit", &e))?;
        let archive = ctx
            .artifacts()
            .path(overlay)
            .map_err(|e| failed("the hidden tests are not kept", &e))?;
        lay_over(&archive, overlay, &files)
            .map_err(|e| failed("laying out the hidden tests", &e))?;
        material.contents = texts_under(&files);
    }
    Ok(material)
}

/// The text of every file under `dir` that is UTF-8 and not larger than
/// [`MAX_AUDITED_FILE_BYTES`], in path order. A file that cannot be read has
/// no text to audit against.
fn texts_under(dir: &Path) -> Vec<String> {
    let mut texts = Vec::new();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(next) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(&next) else {
            continue;
        };
        let mut entries: Vec<_> = entries.filter_map(Result::ok).collect();
        entries.sort_by_key(std::fs::DirEntry::file_name);
        for entry in entries {
            let Ok(kind) = entry.file_type() else {
                continue;
            };
            if kind.is_dir() {
                pending.push(entry.path());
            } else if kind.is_file()
                && entry
                    .metadata()
                    .is_ok_and(|m| m.len() <= MAX_AUDITED_FILE_BYTES)
            {
                texts.extend(std::fs::read_to_string(entry.path()).ok());
            }
        }
    }
    texts
}
