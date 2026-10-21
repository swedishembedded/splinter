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

use crate::scratch::Claim;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use splinter_sdk::agent::solve::{Model, SolveOptions};
use splinter_sdk::agent::sven::atif::{StepOrigin, Trajectory};
use splinter_sdk::agent::sven::model::ModelProvider;
use splinter_sdk::agent::work::{WorkOptions, WorkReport, Worker};
use splinter_sdk::agent::RunConclusion;

use crate::case::{describe, Case, CORRECTABLE, MAX_AGENT_CHANGES};
use crate::control::Control;
use crate::gate::{GateConfig, Verdict};
use crate::git::{single_commit_repo, Change, Git};
use crate::grounding_prompt::{grounding_prompt, CONTINUE_NOTICE};
use crate::plan_only_prompt::plan_only_prompt;
use crate::supervise::{write_correction, Correction, MessageContext, Refused};
use crate::supervisor_prompt::supervisor_request;
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
    /// The harness document that runs the work, when the run uses one.
    pub harness: Option<crate::harness::HarnessConfig>,
    /// Writes the corrections.
    pub supervisor: Model,
    /// Times the supervisor may rewrite a refused message.
    pub supervisor_tries: usize,
    /// Wall-clock limit of one correction message.
    pub supervisor_deadline: Duration,
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

fn failed(e: impl std::fmt::Display) -> anyhow::Error {
    anyhow::anyhow!("the agent run failed: {e}")
}

/// Why `report` ended the run early, if it did.
fn stop_reason(report: &WorkReport) -> Option<String> {
    if report.stopped_by_tool_cap {
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
    }
}

/// How one conversation with the agent is bounded.
struct Round<'a> {
    settings: &'a AttemptSettings,
    started: Instant,
    /// Tool calls made so far over all rounds.
    calls_used: u32,
    /// For each user message sent, in order, whether it is a correction.
    sent: Vec<bool>,
}

impl Round<'_> {
    /// Sends `text`, then (a bounded number of times) tells the agent to go on
    /// when it ended its turn announcing a call it never made, all within
    /// `allowed` tool calls and what is left of the time.
    async fn send(
        &mut self,
        worker: &mut Worker,
        text: &str,
        correction: bool,
        allowed: u32,
    ) -> Result<WorkReport> {
        let mut used = 0u32;
        let mut text = text;
        let mut is_correction = correction;
        let mut follow_ups = 0;
        loop {
            let left = self
                .settings
                .deadline
                .saturating_sub(self.started.elapsed());
            let bounds = worker.options_mut();
            bounds.solve.deadline = left;
            bounds.max_tool_calls = Some(allowed.saturating_sub(used));
            self.sent.push(is_correction);
            let report = worker.send(text).await.map_err(failed)?;
            used += report.tool_calls;
            self.calls_used += report.tool_calls;
            // A turn that ends announcing a call it never made is not a
            // finished attempt: say so, a bounded number of times.
            let announced = report.solution.conclusion == RunConclusion::StoppedWithoutConclusion
                && !report.stopped_by_tool_cap
                && !report.stopped_by_repetition;
            if !announced || follow_ups >= self.settings.follow_ups || left.is_zero() {
                return Ok(report);
            }
            follow_ups += 1;
            text = CONTINUE_NOTICE;
            is_correction = false;
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
    let parent_words_set = vocabulary(&parent_tree);
    let case = Case {
        plan: plan.to_string(),
        diff: commit.diff.clone(),
        changes: commit.changes.clone(),
        parent_words: parent_words_set.iter().cloned().collect(),
        parent_paths: parent_tree.keys().cloned().collect(),
        forbidden_locations: settings.forbidden_locations.clone(),
        attempt_roots: roots,
        gate: settings.gate,
        calls,
        per_correction: settings.budget.per_correction,
    };
    case.write(case_dir.path())?;
    // Where the tree now differs from the commit's.
    let residual_now = || case.residual(case_dir.path(), &tree_dir);

    let mut solve = SolveOptions::new(settings.deadline);
    solve.system = Some(match settings.control {
        Control::Grounded => grounding_prompt(&commit.masked, calls),
        Control::PlanOnly => plan_only_prompt(calls),
    });
    solve.max_output_tokens = settings.max_output_tokens;
    solve.first_response = settings.first_response;
    let stats = Arc::new(std::sync::Mutex::new(crate::timed::Stats::default()));
    let model = crate::timed::Timed::wrap(Arc::clone(&settings.model), Arc::clone(&stats));
    let started = Instant::now();
    let judge = |trace: &Trace,
                 residual: &[Residual],
                 stopped: Option<&str>,
                 corrections: &[Correction]| {
        let guidance: Vec<(usize, String)> = corrections
            .iter()
            .map(|c| (c.calls_before, c.message.clone()))
            .collect();
        case.judge(trace, residual, stopped, &guidance)
    };

    // The harness document runs the work, the follow-ups and the corrections;
    // the judging below is the same either way.
    if let Some(config) = &settings.harness {
        let worked = crate::harness::work(
            config,
            crate::harness::WorkSetup {
                agent: Arc::clone(&model),
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
        let residual = residual_now()?;
        let mut trajectory = worked.trajectory;
        let sent = crate::case::user_turns_are_corrections(&trajectory);
        mark_context_before_last_correction(&mut trajectory, &sent);
        let trace = Trace::of(&trajectory);
        let verdict = judge(
            &trace,
            &residual,
            worked.stopped.as_deref(),
            &worked.corrections,
        );
        return Ok(AttemptOutcome {
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
        });
    }

    let mut options = WorkOptions::new(solve);
    options.max_repeated_results = Some(4);
    let mut worker = Worker::start(&tree_dir, model, options)
        .map_err(|e| anyhow::anyhow!("starting the agent: {e}"))?;
    let mut round = Round {
        settings,
        started,
        calls_used: 0,
        sent: Vec::new(),
    };
    let mut report = round
        .send(
            &mut worker,
            plan,
            false,
            u32::try_from(calls).unwrap_or(u32::MAX),
        )
        .await?;
    let mut corrections: Vec<Correction> = Vec::new();
    let mut refused: Vec<Refused> = Vec::new();
    let mut supervisor_error = None;
    // A wrong tree is told so, in a user's words, a bounded number of times.
    while corrections.len() < settings.corrections {
        let residual = residual_now()?;
        let trace = Trace::of(&report.solution.trajectory);
        let stopped = stop_reason(&report);
        let verdict = judge(&trace, &residual, stopped.as_deref(), &corrections);
        let repairable = !residual.is_empty()
            && verdict
                .reasons
                .iter()
                .all(|r| CORRECTABLE.contains(&r.class.as_str()));
        if !repairable {
            break;
        }
        let agent_changes = Git::at(&tree_dir).working_diff().unwrap_or_default();
        let agent_changes: String = agent_changes.chars().take(MAX_AGENT_CHANGES).collect();
        let differences: Vec<String> = residual.iter().map(describe).collect();
        let request = supervisor_request(plan, &agent_changes, &commit.diff, &differences);
        let guidance: Vec<(usize, String)> = corrections
            .iter()
            .map(|c| (c.calls_before, c.message.clone()))
            .collect();
        let support = crate::justify::support_of_run(plan, &trace.calls, &guidance);
        let ctx = MessageContext::new(plan, &commit.diff, &parent_words_set, &support);
        match write_correction(
            &settings.supervisor,
            &request,
            &ctx,
            settings.supervisor_tries,
            settings.supervisor_deadline,
        )
        .await
        {
            Ok((message, before)) => {
                refused.extend(before);
                corrections.push(Correction {
                    message: message.clone(),
                    calls_before: trace.calls.len(),
                });
                let allowed = u32::try_from(settings.budget.per_correction).unwrap_or(u32::MAX);
                report = round.send(&mut worker, &message, true, allowed).await?;
            }
            Err(none) => {
                refused.extend(none.refused);
                supervisor_error = none.model_error;
                break;
            }
        }
    }
    let seconds = round.started.elapsed().as_secs_f64();

    let checks = settings
        .checks
        .iter()
        .map(|c| run_check(&tree_dir, c, settings.check_timeout))
        .collect();
    let residual = residual_now()?;
    let mut trajectory = report.solution.trajectory.clone();
    mark_context_before_last_correction(&mut trajectory, &round.sent);
    let trace = Trace::of(&trajectory);
    let stopped = stop_reason(&report);
    let verdict = judge(&trace, &residual, stopped.as_deref(), &corrections);
    Ok(AttemptOutcome {
        verdict,
        trajectory,
        final_message: trace.said.last().cloned(),
        trace,
        residual,
        checks,
        stats: stats.lock().map(|s| s.clone()).unwrap_or_default(),
        seconds,
        corrections,
        refused,
        supervisor_error,
    })
}
