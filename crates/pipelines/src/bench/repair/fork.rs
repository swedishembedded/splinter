// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the loop from a failed agent attempt to a
// verified corrected one, for its clients. If your team needs expertise in
// turning agent failures into training data you can procure our services by
// sending an email to info@swedishembedded.com.

//! The fork correction: the examiner lays out the task's workspace, sven's
//! fork replays the student's steps through the last good one, the teacher
//! continues, and the examiner grades the result.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use splinter_agent::fork_at::{continue_from_step, ForkAt, Profile};
use splinter_agent::solve::{Model, SolveOptions};
use splinter_agent::CancelToken;
use splinter_core::experience::{Experience, Provenance};
use splinter_orchestrator::context::Context;

use super::super::artifact::{read_index, read_trial, UnitKind};
use super::super::practice::{BenchCheckout, PRACTICE_FILE};
use super::{commands, compose, Attempted, Case, Intervention};

/// The fork correction: the teacher continues after the last good step.
pub struct ForkCorrection<'a> {
    /// The examiner that lays out and grades the workspace.
    pub checkout: &'a BenchCheckout,
    /// The teacher.
    pub teacher: Model,
    /// The time one continuation may take.
    pub deadline: Duration,
    /// Stops a continuation.
    pub cancel: CancelToken,
}

/// The practice material a practice run attempted: the directory its
/// report's provenance names (the material it was staged from, digest for
/// digest), else the one under its `staged/` that holds `practice.json`.
fn practice_dir(run: &Path) -> Result<PathBuf, String> {
    let report = run.join("report.json");
    let named = std::fs::read(&report)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
        .and_then(|r| {
            r.pointer("/provenance/dir")
                .and_then(|d| d.as_str())
                .map(PathBuf::from)
        })
        .filter(|dir| dir.join(PRACTICE_FILE).is_file());
    if let Some(dir) = named {
        return Ok(dir);
    }
    let staged = run.join("staged");
    let entries = std::fs::read_dir(&staged).map_err(|e| format!("{}: {e}", staged.display()))?;
    entries
        .filter_map(Result::ok)
        .map(|e| e.path())
        .find(|p| p.join(PRACTICE_FILE).is_file())
        .ok_or_else(|| {
            format!(
                "{} names no practice material with {PRACTICE_FILE}",
                report.display()
            )
        })
}

impl Intervention for ForkCorrection<'_> {
    fn name(&self) -> &'static str {
        "fork_correction"
    }

    fn attempt(&self, ctx: &Context, case: &Case<'_>) -> Result<Attempted, String> {
        let keep_through = case
            .diagnosis
            .last_good_step()
            .ok_or("the diagnosis places no last good step")?;
        let folder = case
            .trial
            .staged_task
            .as_deref()
            .ok_or("the practice report names no staged task for the trial")?;
        let practice = practice_dir(case.run)?;
        let export = &case.trial.export;
        let row = read_index(export)?
            .into_iter()
            .filter_map(Result::ok)
            .find(|r| r.trial == case.trial.trial)
            .ok_or_else(|| format!("{} does not list the trial", export.display()))?;
        let artifact = read_trial(export, &row)?;
        let [unit] = artifact.units.as_slice() else {
            return Err("a harness trial has several conversations; none is forked".into());
        };
        let UnitKind::Single(context) = &unit.kind else {
            return Err("a harness conversation is not forked".into());
        };
        // The practice run is read only; the repair works under this run's
        // own state root.
        let shown = shown_observations(&export.join(&row.path))?;
        let work = ctx
            .config()
            .state_root
            .path()
            .join("repairs")
            .join(&case.key[..16.min(case.key.len())]);
        if work.exists() {
            std::fs::remove_dir_all(&work).map_err(|e| format!("{}: {e}", work.display()))?;
        }
        let tree = work.join("tree");
        let scratch = work.join("scratch");
        std::fs::create_dir_all(&scratch).map_err(|e| format!("{}: {e}", scratch.display()))?;
        let result = (|| {
            // The task's own trial sandbox: the student's commands are
            // replayed, and the teacher's run, where a trial's agent runs them.
            let mut session = commands::Session::open(
                self.checkout,
                &practice,
                folder,
                &tree,
                &ctx.config().state_root.path().join("s"),
            )?;
            without_inherited_setgid(&work)?;
            let fork = ForkAt {
                tree: &tree,
                scratch: &scratch,
                student: &unit.trajectory,
                student_model: &case.student.provenance.solver,
                system: &context.system,
                keep_through,
                workspace: None,
                profile: Profile::Compact,
                shown: Some(&shown),
                read_exec: toolchain_dirs(ctx.config().command_env.get("PATH")),
                launcher: Some(session.launcher(self.checkout)),
            };
            let mut options = SolveOptions::new(self.deadline);
            options.cancel = Some(self.cancel.clone());
            let options = self.teacher.solving(options);
            let correction = ctx
                .handle()
                .block_on(continue_from_step(&fork, &self.teacher, options))
                .map_err(|e| format!("the fork correction: {e}"))?;
            let (trajectory, prefix_steps) = compose::onto_stored(
                &case.student.trajectory,
                correction.solution.trajectory,
                keep_through,
            )?;
            let file = work.join("trajectory.json");
            let text = serde_json::to_vec(&trajectory).map_err(|e| e.to_string())?;
            std::fs::write(&file, text).map_err(|e| format!("{}: {e}", file.display()))?;
            let grade = session.grade(&file)?;
            drop(session);
            let lineage = compose::Lineage {
                student: case.student_id,
                trial: &case.trial.trial,
                diagnosis: case.diagnosis,
                keep_through,
                prefix_steps,
                teacher: &self.teacher.identity,
                intervention: self.name(),
                workspace_check: "observations",
                grade: &grade,
            };
            let trajectory = compose::with_lineage(trajectory, &lineage)?;
            let provenance = Provenance {
                teacher: true,
                ..Provenance::new(self.teacher.identity.clone(), ctx.clock())
            };
            let experience = Experience::new(case.student.to_task(), trajectory, None, provenance)
                .map_err(|e| e.to_string())?;
            Ok(Attempted {
                keep_through,
                grade,
                experience,
                prefix_steps,
                teacher_actions: correction.teacher_actions,
            })
        })();
        // The tree is the examiner's copy of the task; what is kept is the
        // experience and the record.
        let _ = std::fs::remove_dir_all(&work);
        result
    }
}

/// The observations the student's model was shown, by call id, from the
/// requests the trial recorded (`model-requests.jsonl`): every tool result a
/// later request carried.
fn shown_observations(trial: &Path) -> Result<BTreeMap<String, String>, String> {
    let path = trial.join("model-requests.jsonl");
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut shown = BTreeMap::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let Ok(entry) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        let messages = entry
            .pointer("/request/messages")
            .and_then(|m| m.as_array());
        for message in messages.into_iter().flatten() {
            if message.get("role").and_then(|r| r.as_str()) != Some("tool") {
                continue;
            }
            let body = message.get("content");
            let id = body
                .and_then(|b| b.get("tool_call_id"))
                .and_then(|i| i.as_str());
            let content = body.and_then(|b| b.get("content")).and_then(|c| c.as_str());
            if let (Some(id), Some(content)) = (id, content) {
                shown.insert(id.to_string(), content.to_string());
            }
        }
    }
    Ok(shown)
}

/// The directories of a `PATH` and their parents, which confined commands
/// may read and run from: an interpreter in a virtual environment
/// (`<venv>/bin/python3`) reads its `<venv>` too.
fn toolchain_dirs(path: Option<&String>) -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = Vec::new();
    for dir in std::env::split_paths(path.map_or("", String::as_str)) {
        let Ok(dir) = dir.canonicalize() else {
            continue;
        };
        if let Some(parent) = dir.parent().filter(|p| p.parent().is_some()) {
            dirs.push(parent.to_path_buf());
        }
        dirs.push(dir);
    }
    dirs.sort();
    dirs.dedup();
    dirs
}

/// Clears the set-group-id bit every directory of `tree` inherited from the
/// directory it was laid out in: the examiner's sandbox makes the workspace
/// without it, and a listing the student recorded shows its modes.
fn without_inherited_setgid(tree: &Path) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;
    const SETGID: u32 = 0o2000;
    let mut pending = vec![tree.to_path_buf()];
    while let Some(dir) = pending.pop() {
        let meta =
            std::fs::symlink_metadata(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let mode = meta.permissions().mode();
        if mode & SETGID != 0 {
            std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(mode & !SETGID))
                .map_err(|e| format!("{}: {e}", dir.display()))?;
        }
        for entry in std::fs::read_dir(&dir).map_err(|e| format!("{}: {e}", dir.display()))? {
            let entry = entry.map_err(|e| format!("{}: {e}", dir.display()))?;
            if entry.file_type().map_err(|e| e.to_string())?.is_dir() {
                pending.push(entry.path());
            }
        }
    }
    Ok(())
}
