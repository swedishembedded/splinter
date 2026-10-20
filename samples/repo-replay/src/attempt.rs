// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pipelines that turn a repository's history
// into verified training data for coding agents, for its clients. If your
// team needs expertise in learning from your own codebase, you can procure
// our services by sending an email to info@swedishembedded.com.

//! One attempt: an agent replays one commit in a tree that holds nothing but
//! the commit's parent.
//!
//! The tree is the parent's files in a fresh single-commit repository
//! ([`crate::git::single_commit_repo`]): no later history, no refs, no
//! remote. The agent runs under the grounding prompt with the plan as its
//! first user message; afterwards the gate judges the tree it left and the
//! calls it made. Attempts share nothing, so they run in parallel; the
//! pipeline orders their results.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use splinter_sdk::agent::solve::SolveOptions;
use splinter_sdk::agent::sven::atif::Trajectory;
use splinter_sdk::agent::sven::model::ModelProvider;
use splinter_sdk::agent::work::{WorkOptions, Worker};

use crate::gate::{evaluate, GateConfig, GateInput, Verdict};
use crate::generated::is_generated_path;
use crate::git::{single_commit_repo, Change, Git};
use crate::grounding_prompt::grounding_prompt;
use crate::trace::Trace;
use crate::tree::{compare, read_tree, Residual, Tree};

/// Everything known about one replayable commit, computed once and shared by
/// its attempts.
#[derive(Clone, Debug)]
pub struct CommitPlan {
    /// The commit's id.
    pub sha: String,
    /// Its parent's id.
    pub parent: String,
    /// The commit message, scrubbed of people and secrets.
    pub message: String,
    /// What it changes, generated paths removed.
    pub changes: Vec<Change>,
    /// The unified diff of those changes.
    pub diff: String,
    /// The commit's tree (what the attempt must end as).
    pub target_tree: Tree,
    /// Words of the parent's text files.
    pub parent_words: std::collections::HashSet<String>,
}

/// How an attempt is run.
#[derive(Clone)]
pub struct AttemptSettings {
    /// The model the agent runs on.
    pub model: Arc<dyn ModelProvider>,
    /// Wall-clock limit of one attempt.
    pub deadline: Duration,
    /// Output tokens an attempt may generate.
    pub max_output_tokens: Option<u64>,
    /// The gate's bounds.
    pub gate: GateConfig,
    /// Paths of the source repository and the run's state, which the shell may not name.
    pub forbidden_locations: Vec<String>,
    /// Where attempt trees are laid out.
    pub scratch: PathBuf,
    /// Project checks run in the attempt tree afterwards (`sh -c`), outcome
    /// recorded and never gated; empty runs none.
    pub checks: Vec<String>,
    /// Longest a check may run.
    pub check_timeout: Duration,
}

/// The outcome of one project check.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CheckOutcome {
    /// The command line.
    pub command: String,
    /// Its exit code; `None` when it was killed.
    pub exit_code: Option<i32>,
    /// Whether the limit stopped it.
    pub timed_out: bool,
    /// How long it ran.
    pub seconds: f64,
}

/// What an attempt produced.
#[derive(Debug)]
pub struct AttemptOutcome {
    /// The gate's decision.
    pub verdict: Verdict,
    /// The run as sven recorded it.
    pub trajectory: Trajectory,
    /// The calls and words the gate read.
    pub trace: Trace,
    /// Where the final tree differed from the commit's.
    pub residual: Vec<Residual>,
    /// Project checks after the edits.
    pub checks: Vec<CheckOutcome>,
    /// The attempt's wall time.
    pub seconds: f64,
    /// The agent's last words.
    pub final_message: Option<String>,
}

/// Runs `command` through `sh -c` in `dir`, killed after `timeout`.
fn run_check(dir: &Path, command: &str, timeout: Duration) -> CheckOutcome {
    let started = Instant::now();
    let child = std::process::Command::new("sh")
        .arg("-c")
        .arg(command)
        .current_dir(dir)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn();
    let Ok(mut child) = child else {
        return CheckOutcome {
            command: command.into(),
            exit_code: None,
            timed_out: false,
            seconds: 0.0,
        };
    };
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                return CheckOutcome {
                    command: command.into(),
                    exit_code: status.code(),
                    timed_out: false,
                    seconds: started.elapsed().as_secs_f64(),
                }
            }
            Ok(None) if started.elapsed() < timeout => {
                std::thread::sleep(Duration::from_millis(50))
            }
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return CheckOutcome {
                    command: command.into(),
                    exit_code: None,
                    timed_out: true,
                    seconds: started.elapsed().as_secs_f64(),
                };
            }
        }
    }
}

/// Replays `commit` with the task `plan`.
pub async fn attempt(
    git: &Git,
    settings: &AttemptSettings,
    commit: &CommitPlan,
    plan: &str,
) -> Result<AttemptOutcome> {
    std::fs::create_dir_all(&settings.scratch)?;
    let scratch = tempfile::Builder::new()
        .prefix("attempt-")
        .tempdir_in(&settings.scratch)
        .context("creating the attempt directory")?;
    let tree_dir = scratch.path().join("tree");
    std::fs::create_dir(&tree_dir)?;
    git.export_tree(&commit.parent, &tree_dir)?;
    single_commit_repo(&tree_dir)?;
    let roots: Vec<String> = [Some(tree_dir.clone()), tree_dir.canonicalize().ok()]
        .into_iter()
        .flatten()
        .map(|p| p.to_string_lossy().into_owned())
        .collect();

    let mut solve = SolveOptions::new(settings.deadline);
    solve.system = Some(grounding_prompt(&commit.diff));
    solve.max_output_tokens = settings.max_output_tokens;
    let mut options = WorkOptions::new(solve);
    options.max_tool_calls = Some(u32::try_from(settings.gate.max_tool_calls).unwrap_or(u32::MAX));
    options.max_repeated_results = Some(4);
    let mut worker = Worker::start(&tree_dir, Arc::clone(&settings.model), options)
        .map_err(|e| anyhow::anyhow!("starting the agent: {e}"))?;
    let started = Instant::now();
    let report = worker
        .send(plan)
        .await
        .map_err(|e| anyhow::anyhow!("the agent run failed: {e}"))?;
    let seconds = started.elapsed().as_secs_f64();

    let checks = settings
        .checks
        .iter()
        .map(|c| run_check(&tree_dir, c, settings.check_timeout))
        .collect();
    let actual = read_tree(&tree_dir)?;
    let residual = compare(&commit.target_tree, &actual, &|p| is_generated_path(p));
    let trajectory = report.solution.trajectory.clone();
    let trace = Trace::of(&trajectory);
    let stopped = if report.stopped_by_tool_cap {
        Some("the tool-call cap stopped the run".to_string())
    } else if report.stopped_by_repetition {
        Some(format!(
            "a call kept returning the same answer: {}",
            report.repeated_call.clone().unwrap_or_default()
        ))
    } else if report.solution.conclusion != splinter_sdk::agent::RunConclusion::Success {
        Some(format!("the run ended as {:?}", report.solution.conclusion))
    } else {
        None
    };
    let added_text = crate::plan::changed_lines(&commit.diff).join("\n");
    let verdict = evaluate(
        &GateInput {
            changes: &commit.changes,
            plan,
            parent_words: &commit.parent_words,
            added_text: &added_text,
            trace: &trace,
            residual: &residual,
            forbidden_locations: &settings.forbidden_locations,
            attempt_roots: &roots,
            stopped: stopped.as_deref(),
        },
        &settings.gate,
    );
    Ok(AttemptOutcome {
        verdict,
        trajectory,
        final_message: trace.said.last().cloned(),
        trace,
        residual,
        checks,
        seconds,
    })
}
