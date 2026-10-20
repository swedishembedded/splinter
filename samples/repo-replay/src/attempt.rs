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
use splinter_sdk::agent::RunConclusion;

use crate::control::Control;
use crate::gate::{evaluate, GateConfig, GateInput, Verdict};
use crate::generated::is_generated_path;
use crate::git::{single_commit_repo, Change, Git};
use crate::grounding_prompt::{grounding_prompt, CONTINUE_NOTICE};
use crate::plan_only_prompt::plan_only_prompt;
use crate::trace::Trace;
use crate::tree::{compare, read_tree, vocabulary, Residual};

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
    /// The same diff as the agent sees it.
    pub masked: crate::mask::MaskedDiff,
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
    /// Longest a request may wait for the server's first byte and token.
    pub first_response: Option<Duration>,
    /// Times an attempt that stopped announcing an unmade call is told to go on.
    pub follow_ups: usize,
    /// The gate's bounds (its `max_tool_calls` is set per commit from the budget).
    pub gate: GateConfig,
    /// Tool calls by commit size.
    pub budget: crate::budget::Budget,
    /// Paths of the source repository and the run's state, which the shell may not name.
    pub forbidden_locations: Vec<String>,
    /// Where attempt trees are laid out.
    pub scratch: PathBuf,
    /// Whether the agent is shown the change.
    pub control: Control,
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
    /// What the requests of the attempt waited and took.
    pub stats: crate::timed::Stats,
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

    // What the parent's code already says, read before the agent touches the tree.
    let parent_tree = read_tree(&tree_dir)?;
    let parent_words = vocabulary(&parent_tree);
    let parent_paths: std::collections::HashSet<String> = parent_tree.keys().cloned().collect();

    let mut solve = SolveOptions::new(settings.deadline);
    // The budget follows the commit's size; the gate and the agent's own cap agree.
    let calls = settings
        .budget
        .calls(commit.changes.len(), crate::budget::hunks_of(&commit.diff));
    let gate = GateConfig {
        max_tool_calls: calls,
        ..settings.gate
    };
    solve.system = Some(match settings.control {
        Control::Grounded => grounding_prompt(&commit.masked, calls),
        Control::PlanOnly => plan_only_prompt(calls),
    });
    solve.max_output_tokens = settings.max_output_tokens;
    solve.first_response = settings.first_response;
    let mut options = WorkOptions::new(solve);
    options.max_tool_calls = Some(u32::try_from(gate.max_tool_calls).unwrap_or(u32::MAX));
    options.max_repeated_results = Some(4);
    let stats = Arc::new(std::sync::Mutex::new(crate::timed::Stats::default()));
    let model = crate::timed::Timed::wrap(Arc::clone(&settings.model), Arc::clone(&stats));
    let mut worker = Worker::start(&tree_dir, model, options)
        .map_err(|e| anyhow::anyhow!("starting the agent: {e}"))?;
    let started = Instant::now();
    let mut report = worker
        .send(plan)
        .await
        .map_err(|e| anyhow::anyhow!("the agent run failed: {e}"))?;
    // A turn that ends announcing a call it never made is not a finished
    // attempt: say so, a bounded number of times, within what is left of the
    // time and the tool calls.
    let mut calls_used = report.tool_calls;
    for _ in 0..settings.follow_ups {
        let announced = report.solution.conclusion == RunConclusion::StoppedWithoutConclusion
            && !report.stopped_by_tool_cap
            && !report.stopped_by_repetition;
        let left = settings.deadline.saturating_sub(started.elapsed());
        if !announced || left.is_zero() {
            break;
        }
        let bounds = worker.options_mut();
        bounds.solve.deadline = left;
        bounds.max_tool_calls = Some(
            u32::try_from(gate.max_tool_calls)
                .unwrap_or(u32::MAX)
                .saturating_sub(calls_used),
        );
        report = worker
            .send(CONTINUE_NOTICE)
            .await
            .map_err(|e| anyhow::anyhow!("the agent run failed: {e}"))?;
        calls_used += report.tool_calls;
    }
    report.tool_calls = calls_used;
    let seconds = started.elapsed().as_secs_f64();

    let checks = settings
        .checks
        .iter()
        .map(|c| run_check(&tree_dir, c, settings.check_timeout))
        .collect();
    let actual = read_tree(&tree_dir)?;
    let target_tree = {
        let dir = tempfile::Builder::new()
            .prefix("target-")
            .tempdir_in(&settings.scratch)?;
        git.export_tree(&commit.sha, dir.path())?;
        read_tree(dir.path())?
    };
    let mut residual = compare(&target_tree, &actual, &|p| is_generated_path(p));
    for r in &mut residual {
        if let Residual::Extra { path } = r {
            if parent_paths.contains(path.as_str()) {
                *r = Residual::Stale {
                    path: std::mem::take(path),
                };
            }
        }
    }
    let trajectory = report.solution.trajectory.clone();
    let trace = Trace::of(&trajectory);
    let stopped = if report.stopped_by_tool_cap {
        Some("the tool-call cap stopped the run".to_string())
    } else if report.stopped_by_repetition {
        Some(format!(
            "a call kept returning the same answer: {}",
            report.repeated_call.clone().unwrap_or_default()
        ))
    } else if report.solution.conclusion != RunConclusion::Success {
        Some(format!("the run ended as {:?}", report.solution.conclusion))
    } else {
        None
    };
    let added_text = crate::plan::changed_lines(&commit.diff).join("\n");
    let verdict = evaluate(
        &GateInput {
            changes: &commit.changes,
            plan,
            parent_words: &parent_words,
            diff: &commit.diff,
            added_text: &added_text,
            trace: &trace,
            residual: &residual,
            forbidden_locations: &settings.forbidden_locations,
            attempt_roots: &roots,
            stopped: stopped.as_deref(),
        },
        &gate,
    );
    Ok(AttemptOutcome {
        verdict,
        trajectory,
        final_message: trace.said.last().cloned(),
        trace,
        residual,
        checks,
        stats: stats.lock().map(|s| s.clone()).unwrap_or_default(),
        seconds,
    })
}
