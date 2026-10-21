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
//! remote. The agent runs under the system prompt of the harness document with the
//! plan as its first user message, and is corrected by a supervisor when the
//! document asks for it; afterwards the gate judges the tree it left and the
//! calls it made. Attempts share nothing, so they run in parallel; the
//! pipeline orders their results.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::scratch::Claim;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use splinter_sdk::agent::solve::Model;
use splinter_sdk::agent::sven::atif::{StepOrigin, Trajectory};
use splinter_sdk::agent::sven::model::ModelProvider;

use crate::case::Case;
use crate::control::Control;
use crate::gate::{GateConfig, Verdict};
use crate::git::{single_commit_repo, Change, Git};
use crate::supervise::{Correction, Refused};
use crate::trace::Trace;
use crate::tree::{read_tree, vocabulary, Residual};

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
    /// Correction messages a wrong attempt may be sent (0: none).
    pub corrections: usize,
    /// The harness documents that run the work.
    pub harness: crate::harness::HarnessConfig,
    /// Writes the corrections.
    pub supervisor: Model,
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
    /// The correction messages the agent was sent, in order.
    pub corrections: Vec<Correction>,
    /// Supervisor messages that were refused, over all rounds.
    pub refused: Vec<Refused>,
    /// Why the supervisor could not be reached, if it could not.
    pub supervisor_error: Option<String>,
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

/// Marks the agent's steps before the last correction as context: they stay
/// in the conversation a record shows, and no dataset view trains on them.
/// `sent` says, for each user step in order, whether it was a correction.
fn mark_context_before_last_correction(trajectory: &mut Trajectory, sent: &[bool]) {
    let Some(last) = sent.iter().rposition(|c| *c) else {
        return;
    };
    let mut users = 0;
    let mut cut = None;
    for (position, step) in trajectory.steps.iter().enumerate() {
        if step.source == StepOrigin::User {
            if users == last {
                cut = Some(position);
                break;
            }
            users += 1;
        }
    }
    let Some(cut) = cut else { return };
    for step in &mut trajectory.steps[..cut] {
        if step.source == StepOrigin::Agent {
            step.is_copied_context = Some(true);
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
    // Dropped when this function returns, i.e. before the verdict is stored.
    let scratch =
        Claim::create(&settings.scratch, "attempt-").context("creating the attempt directory")?;
    let tree_dir = scratch.path().join("tree");
    std::fs::create_dir(&tree_dir)?;
    git.export_tree(&commit.parent, &tree_dir)?;
    single_commit_repo(&tree_dir)?;
    let roots: Vec<String> = [Some(tree_dir.clone()), tree_dir.canonicalize().ok()]
        .into_iter()
        .flatten()
        .map(|p| p.to_string_lossy().into_owned())
        .collect();

    // The hidden side of the attempt, outside the agent's tree: the commit's
    // tree and what the judging reads. The parent's code is read before the
    // agent touches the tree.
    let parent_tree = read_tree(&tree_dir)?;
    let case_dir = Claim::create(&settings.scratch, "target-")?;
    // The tree is in a subdirectory: the claim's directory also holds its marker file.
    let target_dir = Case::target_dir(case_dir.path());
    std::fs::create_dir(&target_dir)?;
    git.export_tree(&commit.sha, &target_dir)?;
    // The budget follows the commit's size; the gate and the agent's own cap agree.
    let calls = settings
        .budget
        .calls(commit.changes.len(), crate::budget::hunks_of(&commit.diff));
    let case = Case {
        plan: plan.to_string(),
        diff: commit.diff.clone(),
        changes: commit.changes.clone(),
        parent_words: vocabulary(&parent_tree).into_iter().collect(),
        parent_paths: parent_tree.keys().cloned().collect(),
        forbidden_locations: settings.forbidden_locations.clone(),
        attempt_roots: roots,
        gate: settings.gate,
        calls,
        per_correction: settings.budget.per_correction,
    };
    case.write(case_dir.path())?;

    let stats = Arc::new(std::sync::Mutex::new(crate::timed::Stats::default()));
    let model = crate::timed::Timed::wrap(Arc::clone(&settings.model), Arc::clone(&stats));
    let started = Instant::now();
    let worked = crate::harness::work(
        &settings.harness,
        crate::harness::WorkSetup {
            agent: model,
            supervisor: Arc::clone(&settings.supervisor.provider),
            tree: &tree_dir,
            case_dir: case_dir.path(),
            control: settings.control,
            plan,
            masked: &commit.masked,
            diff: &commit.diff,
            calls,
            per_correction: settings.budget.per_correction,
            follow_ups: settings.follow_ups,
            corrections: settings.corrections,
            first_response: settings.first_response,
            max_output_tokens: settings.max_output_tokens,
            deadline: settings.deadline,
        },
    )
    .await?;
    let seconds = started.elapsed().as_secs_f64();
    let checks = settings
        .checks
        .iter()
        .map(|c| run_check(&tree_dir, c, settings.check_timeout))
        .collect();
    let residual = case.residual(case_dir.path(), &tree_dir)?;
    let mut trajectory = worked.trajectory;
    let sent = crate::case::user_turns_are_corrections(&trajectory);
    mark_context_before_last_correction(&mut trajectory, &sent);
    let trace = Trace::of(&trajectory);
    let guidance: Vec<(usize, String)> = worked
        .corrections
        .iter()
        .map(|c| (c.calls_before, c.message.clone()))
        .collect();
    let verdict = case.judge(&trace, &residual, worked.stopped.as_deref(), &guidance);
    Ok(AttemptOutcome {
        verdict,
        trajectory,
        final_message: trace.said.last().cloned(),
        trace,
        residual,
        checks,
        stats: stats.lock().map(|s| s.clone()).unwrap_or_default(),
        seconds,
        corrections: worked.corrections,
        refused: worked.refused,
        supervisor_error: worked.supervisor_error,
    })
}
