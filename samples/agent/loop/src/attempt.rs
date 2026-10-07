// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements self-improving coding agents whose every
// step is auditable, for its clients. If your team needs expertise in agent
// evaluation or locally operated coding agents, you can procure our
// services by sending an email to info@swedishembedded.com.

//! One attempt: a fresh worker in a clean checkout, judged afterwards.
//!
//! The attempt resets the checkout to the baseline, lets the worker work
//! within its limits, then reads back from git what changed, applies the
//! protected-path rule and runs the supervisor's acceptance checks. How the
//! attempt ended is classified, and a failed one yields diagnostic feedback
//! for the next: the failure and the tail of the failing output, never a
//! patch.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result};
use serde_json::json;
use splinter_sdk::agent::solve::{Model, SolveError, SolveOptions};
use splinter_sdk::agent::work::{WorkOptions, Worker};
use splinter_sdk::agent::CancelToken;
use splinter_sdk::Context;

use crate::acceptance::{run_all, tail_lines, CheckResult};
use crate::contract::Contract;
use crate::observe::Observation;
use crate::outcome::AttemptSummary;
use crate::repo::{changes_since, reset_to, touched_protected, FileChange};
use crate::trace::Tracer;

/// Lines of a failing check's output the next attempt is shown.
const FEEDBACK_LINES: usize = 40;

/// How an attempt turned out, beyond its summary.
pub struct Judged {
    /// The summary recorded in the outcome.
    pub summary: AttemptSummary,
    /// Whether the candidate is accepted.
    pub accepted: bool,
    /// Whether a person or process asked the run to stop.
    pub cancelled: bool,
    /// Whether the run's cost cap ended it (no retry can help).
    pub over_budget: bool,
    /// Diagnostic feedback for the next attempt.
    pub feedback: Option<String>,
    /// Whether any model call failed in transit and used up its retries.
    pub provider_failed: bool,
    /// The patch of the candidate.
    pub patch: String,
}

/// What an attempt needs from the run around it.
pub struct Setup<'a> {
    /// The run's model context.
    pub ctx: &'a Context,
    /// The model.
    pub model: &'a Model,
    /// Whether the model is reached over an API.
    pub remote: bool,
    /// The contract.
    pub contract: &'a Contract,
    /// The system prompt in effect.
    pub system: &'a str,
    /// The trace.
    pub tracer: &'a Arc<Tracer>,
    /// The checkout.
    pub work: &'a Path,
    /// The run's directory.
    pub run_dir: &'a Path,
    /// Cancels when the run is cancelled from outside.
    pub run_cancel: CancelToken,
    /// Seconds of the total budget left.
    pub remaining_secs: u64,
}

/// The prompt of an attempt: the task, how it will be judged, and what the
/// previous attempt's failure looked like.
#[must_use]
pub fn compose_prompt(contract: &Contract, feedback: Option<&str>) -> String {
    let mut prompt = format!("Task:\n{}\n", contract.task.trim());
    let shown: Vec<_> = contract.acceptance.iter().filter(|c| c.visible).collect();
    let hidden = contract.acceptance.len() - shown.len();
    if !shown.is_empty() {
        prompt.push_str("\nYour change will be judged by these checks (run them yourself):\n");
        for check in shown {
            prompt.push_str(&format!("- {}: `{}`\n", check.name, check.command));
        }
    }
    if hidden > 0 {
        prompt.push_str(&format!(
            "\nThe supervisor also runs {hidden} check{} you cannot see. Make the behaviour the task and the documentation describe correct in general, not just for the cases you try.\n",
            if hidden == 1 { "" } else { "s" }
        ));
    }
    if !contract.protected.is_empty() {
        prompt.push_str(&format!(
            "\nDo not modify: {}.\n",
            contract.protected.join(", ")
        ));
    }
    for hint in &contract.hints {
        prompt.push_str(&format!("\nSupervisor note: {hint}\n"));
    }
    if let Some(feedback) = feedback {
        prompt.push_str(&format!(
            "\nA previous attempt failed and its work was discarded. What went wrong:\n{feedback}\n"
        ));
    }
    prompt
}

/// Forwards a cancel from `from` to `to` until `done` is set.
fn forward_cancel(
    from: CancelToken,
    to: CancelToken,
    done: Arc<AtomicBool>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        while !done.load(Ordering::Relaxed) {
            if from.is_cancelled() {
                to.cancel();
                return;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    })
}

/// What a candidate looked like when it was inspected.
struct Inspection {
    patch: String,
    changes: Vec<FileChange>,
    checks: Vec<CheckResult>,
    touched: Vec<String>,
}

/// Reads the candidate back from git and judges it with the checks (unless
/// the run was cancelled, when nothing more is run).
fn inspect(setup: &Setup<'_>, number: u32, round: u32, cancelled: bool) -> Result<Inspection> {
    let contract = setup.contract;
    let (patch, changes) = changes_since(setup.work, &contract.baseline_rev)?;
    let suffix = if round == 0 {
        String::new()
    } else {
        format!("-round{round}")
    };
    let patch_path = setup.run_dir.join(format!("patch-{number}{suffix}.diff"));
    crate::store::write_atomic(&patch_path, crate::redact::redact(&patch).as_bytes())?;
    emit_changes(setup, &changes, &patch_path)?;
    let touched = touched_protected(&changes, &contract.protected);
    let checks = if cancelled {
        Vec::new()
    } else {
        run_all(&contract.acceptance, setup.work)
    };
    emit_checks(setup, &checks)?;
    Ok(Inspection {
        patch,
        changes,
        checks,
        touched,
    })
}

/// Whether a failure is one a further round in the same conversation can
/// help with: the worker stopped with the checks still red. A limit, a
/// cancel, a provider failure or a protected path ends the attempt.
fn worth_a_follow_up(failure: &str) -> bool {
    failure.starts_with("acceptance_failed") || failure == "no_acceptance_checks_ran"
}

/// The seconds an attempt must have left for another round to be worth
/// starting: less than a model call takes.
const MIN_ROUND_SECS: u64 = 20;

/// What the sends of one attempt add up to.
struct Sends {
    tool_calls: u32,
    capped: bool,
    conclusion: String,
    final_output: Option<String>,
    provider_failed: bool,
}

/// Runs attempt `number` and judges it.
///
/// The worker works until it stops; the candidate is then judged, and while
/// the checks are red and the attempt has time, tool calls and rounds left
/// the worker is told what failed, in the same conversation and on the same
/// checkout, and works again. A worker that announces its next step and
/// stops, a habit of small models, is brought back this way instead of
/// costing a whole attempt.
pub fn run_attempt(setup: &Setup<'_>, number: u32, feedback: Option<&str>) -> Result<Judged> {
    let contract = setup.contract;
    reset_to(setup.work, &contract.baseline_rev)?;
    let prompt = compose_prompt(contract, feedback);
    let attempt_secs = contract
        .limits
        .attempt_secs
        .min(setup.remaining_secs)
        .max(1);
    setup.tracer.set_attempt(number);
    setup.tracer.emit(
        "attempt_started",
        None,
        json!({
            "attempt": number,
            "model": setup.model.identity,
            "prompt": prompt,
            "prompt_digest": splinter_sdk::vocabulary::digest::Digest::sha256_of(prompt.as_bytes()).to_string(),
            "deadline_secs": attempt_secs,
            "max_output_tokens": contract.limits.max_output_tokens,
            "max_tool_calls": contract.limits.max_tool_calls,
            "follow_ups": contract.limits.follow_ups,
            "has_feedback": feedback.is_some(),
        }),
    )?;

    let attempt_cancel = CancelToken::new();
    let done = Arc::new(AtomicBool::new(false));
    let forwarder = forward_cancel(
        setup.run_cancel.clone(),
        attempt_cancel.clone(),
        done.clone(),
    );
    let observation = Observation::new(
        setup.tracer.clone(),
        setup.work.to_path_buf(),
        contract.baseline_rev.clone(),
        setup.remote,
        contract.limits.max_cost_usd.filter(|_| setup.remote),
        attempt_cancel.clone(),
    );
    let started = Instant::now();
    let mut solve = setup
        .model
        .solving(SolveOptions::new(Duration::from_secs(attempt_secs)));
    solve.max_output_tokens = Some(contract.limits.max_output_tokens);
    solve.cancel = Some(attempt_cancel.clone());
    solve.system = Some(setup.system.to_string());
    let mut options = WorkOptions::new(solve);
    options.observer = Some(observation.observer());

    let mut sends = Sends {
        tool_calls: 0,
        capped: false,
        conclusion: "ProviderError".into(),
        final_output: None,
        provider_failed: false,
    };
    let mut worker: Option<Worker> = None;
    let mut message = prompt;
    let mut round = 0u32;
    let (inspection, failure) = loop {
        let left_secs = attempt_secs.saturating_sub(started.elapsed().as_secs());
        let left_calls = contract
            .limits
            .max_tool_calls
            .saturating_sub(sends.tool_calls);
        options.max_tool_calls = Some(left_calls);
        options.solve.deadline = Duration::from_secs(left_secs.max(1));
        send_round(setup, &mut worker, &options, &message, &mut sends)?;
        let cancelled = setup.run_cancel.is_cancelled();
        let gathered = observation.gathered();
        let inspection = inspect(setup, number, round, cancelled)?;
        let all_pass = !inspection.checks.is_empty() && inspection.checks.iter().all(|c| c.passed);
        let failure = classify(
            &sends.conclusion,
            sends.capped,
            cancelled,
            gathered.over_budget,
            sends.provider_failed,
            &inspection.touched,
            &inspection.checks,
            all_pass,
        );
        let again = failure.as_deref().is_some_and(worth_a_follow_up)
            && round < contract.limits.follow_ups
            && attempt_secs.saturating_sub(started.elapsed().as_secs()) >= MIN_ROUND_SECS
            && sends.tool_calls < contract.limits.max_tool_calls;
        if !again {
            break (inspection, failure);
        }
        round += 1;
        message = diagnostic(
            failure.as_deref().unwrap_or_default(),
            &inspection.touched,
            &inspection.checks,
            sends.final_output.as_deref(),
        );
        setup.tracer.emit(
            "follow_up",
            None,
            json!({"attempt": number, "round": round, "after": failure, "message": message}),
        )?;
    };
    done.store(true, Ordering::Relaxed);
    let _ = forwarder.join();
    let seconds = started.elapsed().as_secs_f64();

    let trajectory_path = match &worker {
        Some(worker) => Some(save_trajectory(setup, number, worker)?),
        None => {
            setup.tracer.emit(
                "provider_error",
                None,
                json!({"retries": contract.limits.provider_retries}),
            )?;
            None
        }
    };
    save_audit_log(setup);

    let gathered = observation.gathered();
    let cancelled = setup.run_cancel.is_cancelled();
    let accepted = failure.is_none();
    let feedback = failure.as_ref().map(|why| {
        diagnostic(
            why,
            &inspection.touched,
            &inspection.checks,
            sends.final_output.as_deref(),
        )
    });
    let summary = AttemptSummary {
        attempt: number,
        conclusion: sends.conclusion.clone(),
        failure,
        tool_calls: sends.tool_calls,
        stopped_by_tool_cap: sends.capped,
        changed_files: inspection.changes,
        checks: inspection.checks,
        final_message: sends.final_output.clone().or(gathered.last_message.clone()),
        seconds,
        input_tokens: Some(gathered.input_tokens),
        output_tokens: Some(gathered.output_tokens),
    };
    setup.tracer.emit(
        "attempt_finished",
        None,
        json!({"attempt": number, "accepted": accepted, "conclusion": summary.conclusion,
               "failure": summary.failure, "tool_calls": sends.tool_calls, "rounds": round + 1,
               "seconds": seconds, "trajectory": trajectory_path, "dropped_events": gathered.dropped,
               "cost_usd": gathered.cost_usd, "calls_without_price": gathered.calls_without_price}),
    )?;
    Ok(Judged {
        summary,
        accepted,
        cancelled,
        over_budget: gathered.over_budget,
        feedback,
        provider_failed: sends.provider_failed,
        patch: inspection.patch,
    })
}

/// One send of the attempt: the first starts the worker (retrying a model
/// call that failed in transit), a later one continues its conversation.
fn send_round(
    setup: &Setup<'_>,
    worker: &mut Option<Worker>,
    options: &WorkOptions,
    message: &str,
    sends: &mut Sends,
) -> Result<()> {
    let report = match worker.as_mut() {
        None => match send_with_retries(setup, options, message) {
            Ok((report, started)) => {
                *worker = Some(started);
                Some(report)
            }
            Err(_) => None,
        },
        Some(w) => {
            *w.options_mut() = options.clone();
            match setup.ctx.block_on(w.send(message)) {
                Ok(report) => Some(report),
                Err(e) => {
                    setup.tracer.emit(
                        "provider_error",
                        None,
                        json!({"error": e.to_string(), "during": "follow-up"}),
                    )?;
                    None
                }
            }
        }
    };
    match report {
        Some(report) => {
            sends.tool_calls += report.tool_calls;
            sends.capped |= report.stopped_by_tool_cap;
            sends.conclusion = format!("{:?}", report.solution.conclusion);
            sends.final_output = report.solution.final_output.clone();
        }
        None => sends.provider_failed = true,
    }
    Ok(())
}

/// Why an attempt was not accepted, if it was not.
#[allow(clippy::too_many_arguments)]
fn classify(
    conclusion: &str,
    capped: bool,
    cancelled: bool,
    over_budget: bool,
    provider_failed: bool,
    touched: &[String],
    checks: &[CheckResult],
    all_pass: bool,
) -> Option<String> {
    if cancelled {
        return Some("cancelled".into());
    }
    if over_budget {
        return Some("cost_limit".into());
    }
    if provider_failed {
        return Some("provider_error".into());
    }
    if !touched.is_empty() {
        return Some(format!("protected_path_modified: {}", touched.join(", ")));
    }
    if all_pass {
        return None;
    }
    if capped {
        return Some("tool_call_limit".into());
    }
    match conclusion {
        "Timeout" => return Some("time_limit".into()),
        "BudgetExhausted" => return Some("token_limit".into()),
        "AgentError" => return Some("agent_error".into()),
        _ => {}
    }
    if checks.is_empty() {
        return Some("no_acceptance_checks_ran".into());
    }
    let failed: Vec<&str> = checks
        .iter()
        .filter(|c| !c.passed)
        .map(|c| c.name.as_str())
        .collect();
    Some(format!("acceptance_failed: {}", failed.join(", ")))
}

/// The feedback the next attempt is shown: what failed and the output that
/// shows it; never a patch.
fn diagnostic(why: &str, touched: &[String], checks: &[CheckResult], said: Option<&str>) -> String {
    let mut text = format!("Failure: {why}.\n");
    if !touched.is_empty() {
        text.push_str("You changed files you were told not to; leave them as they are.\n");
    }
    for check in checks.iter().filter(|c| !c.passed) {
        let output = if check.stderr_tail.trim().is_empty() {
            &check.stdout_tail
        } else {
            &check.stderr_tail
        };
        text.push_str(&format!(
            "Check `{}` ({}) failed{}:\n{}\n",
            check.name,
            check.command,
            check
                .exit_code
                .map_or(String::new(), |c| format!(" with exit code {c}")),
            tail_lines(output, FEEDBACK_LINES)
        ));
    }
    if let Some(said) = said.filter(|s| !s.trim().is_empty()) {
        text.push_str(&format!("You ended by saying: {}\n", tail_lines(said, 6)));
    }
    text
}

/// Sends `prompt`, retrying a model call that failed in transit a bounded
/// number of times with a growing pause; each retry starts a new worker so
/// no half-finished state is carried over.
fn send_with_retries(
    setup: &Setup<'_>,
    options: &WorkOptions,
    prompt: &str,
) -> Result<(splinter_sdk::agent::work::WorkReport, Worker)> {
    let retries = setup.contract.limits.provider_retries;
    let mut last: Option<anyhow::Error> = None;
    for retry in 0..=retries {
        if retry > 0 {
            let pause = Duration::from_secs(1u64 << retry.min(5));
            setup.tracer.emit(
                "provider_retry",
                None,
                json!({"retry": retry, "pause_secs": pause.as_secs(), "after": last.as_ref().map(|e| format!("{e:#}"))}),
            )?;
            std::thread::sleep(pause);
            reset_to(setup.work, &setup.contract.baseline_rev)?;
        }
        let mut worker = Worker::start(setup.work, setup.model.provider.clone(), options.clone())
            .map_err(|e| anyhow::anyhow!("the worker could not start: {e}"))?;
        match setup.ctx.block_on(worker.send(prompt)) {
            Ok(report) => return Ok((report, worker)),
            Err(SolveError::Engine(e)) => last = Some(anyhow::anyhow!("{e}")),
            Err(e) => return Err(anyhow::anyhow!("{e}")),
        }
    }
    Err(last.unwrap_or_else(|| anyhow::anyhow!("no model call was made")))
}

/// Persists sven's own record of the attempt (ATIF) in the run's directory.
fn save_trajectory(setup: &Setup<'_>, number: u32, worker: &Worker) -> Result<PathBuf> {
    let path = setup
        .run_dir
        .join("trajectories")
        .join(format!("attempt-{number}.atif.json"));
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    splinter_sdk::agent::sven::atif::persist::write_trajectory_atomic(
        &path,
        &worker.trajectory(),
        None,
    )
    .with_context(|| format!("writing {}", path.display()))?;
    Ok(path)
}

/// Keeps sven's tool audit log of the attempt as an artifact.
fn save_audit_log(setup: &Setup<'_>) {
    let log = setup.work.join(".sven").join("audit.jsonl");
    if let Ok(bytes) = std::fs::read(&log) {
        let text = crate::redact::redact(&String::from_utf8_lossy(&bytes));
        if let Ok(address) = setup.tracer.store_artifact(text.as_bytes()) {
            let _ = setup.tracer.emit(
                "audit_log",
                None,
                json!({"artifact": address, "bytes": bytes.len()}),
            );
        }
    }
}

fn emit_changes(setup: &Setup<'_>, changes: &[FileChange], patch: &Path) -> Result<()> {
    setup.tracer.emit(
        "candidate",
        None,
        json!({"changed_files": changes, "patch": patch,
               "baseline": setup.contract.baseline_rev}),
    )?;
    Ok(())
}

fn emit_checks(setup: &Setup<'_>, checks: &[CheckResult]) -> Result<()> {
    for check in checks {
        setup.tracer.emit(
            "validation",
            None,
            json!({"name": check.name, "command": check.command, "passed": check.passed,
                   "exit_code": check.exit_code, "timed_out": check.timed_out,
                   "duration_ms": check.duration_ms, "stdout_tail": check.stdout_tail,
                   "stderr_tail": check.stderr_tail,
                   "covers": "the whole candidate checkout, at the revision of this attempt's patch"}),
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acceptance::Check;
    use crate::contract::{Limits, CONTRACT_SCHEMA};

    fn contract() -> Contract {
        Contract {
            schema: CONTRACT_SCHEMA,
            task: "fix the bug".into(),
            repository: "/r".into(),
            baseline_rev: "abc".into(),
            setup: Vec::new(),
            acceptance: vec![Check {
                name: "unit".into(),
                command: "make test".into(),
                timeout_secs: 5,
                visible: true,
            }],
            protected: vec!["tests/".into()],
            limits: Limits {
                attempt_secs: 1,
                total_secs: 1,
                max_output_tokens: 1,
                max_tool_calls: 1,
                max_attempts: 1,
                provider_retries: 0,
                follow_ups: 0,
                max_cost_usd: None,
            },
            model: "local:x".into(),
            allow_api_models: false,
            system_prompt_digest: String::new(),
            system_prompt_source: "built-in".into(),
            hints: vec![],
        }
    }

    fn failed(name: &str, stderr: &str) -> CheckResult {
        CheckResult {
            name: name.into(),
            command: "make test".into(),
            passed: false,
            exit_code: Some(1),
            timed_out: false,
            duration_ms: 1,
            stdout_tail: String::new(),
            stderr_tail: stderr.into(),
        }
    }

    #[test]
    fn the_prompt_names_the_checks_the_protected_paths_and_any_hint() {
        let mut c = contract();
        c.hints.push("look at parse()".into());
        let prompt = compose_prompt(&c, None);
        assert!(prompt.contains("make test") && prompt.contains("tests/"));
        c.acceptance[0].visible = false;
        let hidden = compose_prompt(&c, None);
        assert!(
            !hidden.contains("make test"),
            "a hidden check's command is not shown"
        );
        assert!(hidden.contains("1 check you cannot see"));
        assert!(prompt.contains("Supervisor note: look at parse()"));
        assert!(!prompt.contains("previous attempt"));
        assert!(compose_prompt(&c, Some("boom")).contains("previous attempt failed"));
    }

    #[test]
    fn failures_are_classified_by_their_cause_in_a_fixed_order() {
        let f = [failed("unit", "E")];
        let why = |capped, cancelled, over, provider, touched: &[String], all| {
            classify(
                "Success", capped, cancelled, over, provider, touched, &f, all,
            )
        };
        assert_eq!(
            why(false, true, false, false, &[], false).unwrap(),
            "cancelled"
        );
        assert_eq!(
            why(false, false, true, false, &[], false).unwrap(),
            "cost_limit"
        );
        assert_eq!(
            why(false, false, false, true, &[], false).unwrap(),
            "provider_error"
        );
        assert!(why(false, false, false, false, &["tests/a".into()], true)
            .unwrap()
            .starts_with("protected_path_modified"));
        assert_eq!(
            why(true, false, false, false, &[], false).unwrap(),
            "tool_call_limit"
        );
        assert_eq!(
            why(false, false, false, false, &[], false).unwrap(),
            "acceptance_failed: unit"
        );
        assert!(why(false, false, false, false, &[], true).is_none());
        assert_eq!(
            classify("Timeout", false, false, false, false, &[], &f, false).unwrap(),
            "time_limit"
        );
    }

    #[test]
    fn feedback_shows_the_failing_output_and_never_a_patch() {
        let text = diagnostic(
            "acceptance_failed: unit",
            &[],
            &[failed("unit", "AssertionError: 1 != 2")],
            Some("done"),
        );
        assert!(text.contains("AssertionError: 1 != 2") && text.contains("exit code 1"));
        assert!(!text.contains("diff --git"));
    }
}
