// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements self-improving coding agents whose every
// step is auditable, for its clients. If your team needs expertise in agent
// evaluation or locally operated coding agents, you can procure our
// services by sending an email to info@swedishembedded.com.

//! One attempt: a fresh worker in a fresh copy of the snapshot, judged
//! afterwards.
//!
//! The worker is the shared one (`solve_in_workspace`): its tree is laid out
//! from the snapshot and thrown away, the change is read back from git as a
//! patch, and the checks the worker is shown earn it follow-up turns inside
//! the attempt. The loop then judges the patch by the contract's checks in
//! another fresh copy (the workspace grading), audits the worker's
//! trajectory for any sight of the supervisor's material, classifies how the
//! attempt ended and, for a failed one, writes diagnostic feedback for the
//! next: the failure and the output of the checks the worker may see, never
//! a patch, never what a hidden check printed.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context as _, Result};
use serde_json::json;
use splinter_sdk::agent::solve::{Model, SolveError, SolveOptions};
use splinter_sdk::agent::sven::atif::StepOrigin;
use splinter_sdk::agent::work::{
    solve_in_workspace, tail, WorkOptions, WorkProtocol, WorkspaceSolution,
};
use splinter_sdk::agent::CancelToken;
use splinter_sdk::measure::exposure::{audit_exposure, HiddenMaterial};
use splinter_sdk::sandbox::{CommandCall, Patch};
use splinter_sdk::vocabulary::annotation::Outcome;
use splinter_sdk::vocabulary::experience::Provenance;
use splinter_sdk::Context;

use crate::contract::Contract;
use crate::observe::Observation;
use crate::outcome::{AttemptSummary, CheckResult};
use crate::store::LoopHome;
use crate::trace::Tracer;
use crate::workspace::{workspace_check, Judgement, Workspace};
use splinter_sdk::measure::verifiers::workspace::Refusal;

/// The name experiences of the loop's attempts carry as their solver.
const SOLVER: &str = "agent-loop";

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
    /// The repository at the baseline.
    pub workspace: &'a Workspace,
    /// The system prompt in effect.
    pub system: &'a str,
    /// The trace.
    pub tracer: &'a Arc<Tracer>,
    /// The loop's namespace, which holds every run's contract.
    pub home: &'a LoopHome,
    /// The run's directory.
    pub run_dir: &'a Path,
    /// Cancels when the run is cancelled from outside.
    pub run_cancel: CancelToken,
    /// Seconds of the total budget left.
    pub remaining_secs: u64,
}

/// How an attempt is asked to work. A model left to itself may probe one
/// question again and again and never write the change; a first complete
/// edit that the checks then judge is a better use of a bounded attempt.
const WORKING_STYLE: &str = "\nHow to work: read what you need, then make a first complete edit early and run the checks on it; refine from what they print. Look each thing up once. Do not repeat a command whose answer you already have: if you are about to, write the change instead.\n";

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
    prompt.push_str(WORKING_STYLE);
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

/// What the episodes of one attempt add up to.
struct Episode {
    solved: Option<WorkspaceSolution>,
}

impl Episode {
    fn tool_calls(&self) -> u32 {
        self.solved.as_ref().map_or(0, |s| s.report.tool_calls)
    }

    fn capped(&self) -> bool {
        self.solved
            .as_ref()
            .is_some_and(|s| s.report.stopped_by_tool_cap)
    }

    fn repeated(&self) -> Option<&str> {
        self.solved.as_ref()?.report.repeated_call.as_deref()
    }

    fn conclusion(&self) -> String {
        self.solved.as_ref().map_or_else(
            || "ProviderError".into(),
            |s| format!("{:?}", s.solution.conclusion),
        )
    }

    fn patch(&self) -> Patch {
        self.solved.as_ref().map_or(
            Patch {
                diff: Vec::new(),
                files: Vec::new(),
            },
            |s| s.patch.clone(),
        )
    }
}

/// Runs attempt `number` and judges it.
///
/// The worker works until it stops; while the checks it is shown are red and
/// the attempt has time, tool calls and rounds left it is told what failed,
/// in the same conversation, and works again. A worker that announces its
/// next step and stops, a habit of small models, is brought back this way
/// instead of costing a whole attempt.
pub fn run_attempt(setup: &Setup<'_>, number: u32, feedback: Option<&str>) -> Result<Judged> {
    let contract = setup.contract;
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
            "max_repeated_results": contract.limits.max_repeated_results,
            "follow_ups": contract.limits.follow_ups,
            "has_feedback": feedback.is_some(),
        }),
    )?;
    let check = workspace_check(contract);
    let task = setup.workspace.task(&prompt, &check)?;

    let attempt_cancel = CancelToken::new();
    let done = Arc::new(AtomicBool::new(false));
    let forwarder = forward_cancel(
        setup.run_cancel.clone(),
        attempt_cancel.clone(),
        done.clone(),
    );
    let observation = Observation::new(
        setup.tracer.clone(),
        setup.workspace.env.clone(),
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
    options.tree_observer = Some(observation.tree_observer());
    let protocol = WorkProtocol {
        follow_ups: contract.limits.follow_ups,
        max_tool_calls: Some(contract.limits.max_tool_calls),
        max_repeated_results: Some(contract.limits.max_repeated_results).filter(|n| *n > 0),
    };
    let shown: Vec<CommandCall> = contract
        .acceptance
        .iter()
        .filter(|c| c.visible)
        .map(|c| c.call())
        .collect();

    let episode = Episode {
        solved: solve_with_retries(
            setup,
            &task,
            &shown,
            options,
            &protocol,
            started,
            attempt_secs,
        )?,
    };
    done.store(true, Ordering::Relaxed);
    let _ = forwarder.join();
    let seconds = started.elapsed().as_secs_f64();
    let cancelled = setup.run_cancel.is_cancelled();
    let gathered = observation.gathered();
    let provider_failed = episode.solved.is_none();

    let patch = episode.patch();
    let patch_text = String::from_utf8_lossy(&patch.diff).into_owned();
    let judgement = if cancelled || provider_failed {
        Judgement {
            refusal: None,
            checks: Vec::new(),
            changes: Vec::new(),
        }
    } else {
        setup.workspace.judge(contract, &patch)?
    };
    let patch_path = setup.run_dir.join(format!("patch-{number}.diff"));
    // Exactly the candidate: `apply` applies this file, and a redacted patch
    // is not the change that was judged.
    crate::store::write_atomic(&patch_path, &patch.diff)?;
    emit_judgement(setup, &judgement, &patch_path)?;
    let exposed = match &episode.solved {
        Some(solved) => exposed_material(setup, &task, solved)?,
        None => false,
    };
    emit_follow_ups(setup, number, &episode)?;
    let trajectory_path = match &episode.solved {
        Some(solved) => Some(save_trajectory(setup, number, solved)?),
        None => None,
    };

    let failure = classify(&Facts {
        conclusion: &episode.conclusion(),
        capped: episode.capped(),
        repeated: episode.repeated(),
        cancelled,
        over_budget: gathered.over_budget,
        provider_failed,
        exposed,
        judgement: &judgement,
    });
    let accepted = failure.is_none();
    let feedback = failure
        .as_ref()
        .map(|why| diagnostic(contract, why, &judgement, gathered.last_message.as_deref()));
    let summary = AttemptSummary {
        attempt: number,
        conclusion: episode.conclusion(),
        failure,
        tool_calls: episode.tool_calls(),
        stopped_by_tool_cap: episode.capped(),
        stopped_by_repetition: episode.repeated().is_some(),
        changed_files: judgement.changes,
        checks: judgement.checks,
        final_message: gathered.last_message.clone(),
        seconds,
        input_tokens: Some(gathered.input_tokens),
        output_tokens: Some(gathered.output_tokens),
    };
    let rounds = episode.solved.as_ref().map_or(1, |s| s.feedback_rounds + 1);
    setup.tracer.emit(
        "attempt_finished",
        None,
        json!({"attempt": number, "accepted": accepted, "conclusion": summary.conclusion,
               "failure": summary.failure, "tool_calls": summary.tool_calls, "rounds": rounds,
               "seconds": seconds, "trajectory": trajectory_path, "dropped_events": gathered.dropped,
               "cost_usd": gathered.cost_usd, "calls_without_price": gathered.calls_without_price}),
    )?;
    Ok(Judged {
        summary,
        accepted,
        cancelled,
        over_budget: gathered.over_budget,
        feedback,
        provider_failed,
        patch: patch_text,
    })
}

/// The worker's episode, retrying a model call that failed in transit a
/// bounded number of times with a growing pause; each retry is a whole new
/// episode in a new copy of the snapshot, so no half-finished state is
/// carried over. `None` when the provider kept failing.
fn solve_with_retries(
    setup: &Setup<'_>,
    task: &splinter_sdk::vocabulary::experience::Task,
    shown: &[CommandCall],
    options: WorkOptions,
    protocol: &WorkProtocol,
    started: Instant,
    attempt_secs: u64,
) -> Result<Option<WorkspaceSolution>> {
    let retries = setup.contract.limits.provider_retries;
    let mut last: Option<String> = None;
    for retry in 0..=retries {
        if retry > 0 {
            let pause = Duration::from_secs(1u64 << retry.min(5));
            setup.tracer.emit(
                "provider_retry",
                None,
                json!({"retry": retry, "pause_secs": pause.as_secs(), "after": last}),
            )?;
            std::thread::sleep(pause);
        }
        let mut bounds = options.clone();
        bounds.solve.deadline = Duration::from_secs(
            attempt_secs
                .saturating_sub(started.elapsed().as_secs())
                .max(1),
        );
        let solved = setup.ctx.block_on(solve_in_workspace(
            task,
            &setup.workspace.env,
            &setup.workspace.archive,
            shown,
            setup.model.provider.clone(),
            bounds,
            protocol,
        ));
        match solved {
            Ok(solved) => return Ok(Some(solved)),
            Err(SolveError::Engine(e)) => last = Some(format!("{e}")),
            Err(e) => return Err(anyhow!("{e}")),
        }
    }
    setup.tracer.emit(
        "provider_error",
        None,
        json!({"retries": retries, "error": last}),
    )?;
    Ok(None)
}

/// Whether the worker's trajectory shows any sight of the supervisor's
/// material: a hidden check's command or the script it names, or the
/// directories that hold the loop's records (every contract in them names
/// the checks). The audit is the shared exposure audit; an attempt that
/// failed it is rejected whatever the checks said, and the trace keeps what
/// it found (where, never the material).
fn exposed_material(
    setup: &Setup<'_>,
    task: &splinter_sdk::vocabulary::experience::Task,
    solved: &WorkspaceSolution,
) -> Result<bool> {
    let experience = solved
        .solution
        .clone()
        .into_experience(task.clone(), Provenance::new(SOLVER, setup.ctx.clock()))
        .context("the attempt's trajectory is not an experience")?;
    let finding = audit_exposure(task, &experience, &hidden_material(setup));
    let exposed = finding.outcome == Outcome::Fail;
    if exposed {
        setup
            .tracer
            .emit("exposure", None, finding.evidence.clone())?;
    }
    Ok(exposed)
}

/// What the worker must not name: the commands of the hidden checks and the
/// scripts they point at, the loop's records and Splinter's state.
fn hidden_material(setup: &Setup<'_>) -> HiddenMaterial {
    let mut paths = Vec::new();
    for check in setup.contract.acceptance.iter().filter(|c| !c.visible) {
        paths.push(check.command.clone());
        let words: Vec<&str> = check.command.split_whitespace().collect();
        // The first word is the program; a lone word is the script itself.
        let candidates = if words.len() == 1 {
            &words[..]
        } else {
            &words[1..]
        };
        paths.extend(
            candidates
                .iter()
                .filter(|w| w.starts_with('/'))
                .map(|w| (*w).to_string()),
        );
    }
    HiddenMaterial {
        paths,
        forbidden: vec![
            setup.home.root().display().to_string(),
            setup.ctx.root().path().display().to_string(),
        ],
        ..HiddenMaterial::default()
    }
}

/// The follow-up turns the worker was sent inside the attempt, from its
/// trajectory: every user turn after the first.
fn emit_follow_ups(setup: &Setup<'_>, number: u32, episode: &Episode) -> Result<()> {
    let Some(solved) = &episode.solved else {
        return Ok(());
    };
    let turns = solved
        .solution
        .trajectory
        .steps
        .iter()
        .filter(|step| step.source == StepOrigin::User)
        .skip(1);
    for (round, step) in (1u32..).zip(turns) {
        setup.tracer.emit(
            "follow_up",
            None,
            json!({"attempt": number, "round": round,
                   "message": step.message.as_text().unwrap_or_default()}),
        )?;
    }
    Ok(())
}

/// Persists sven's own record of the attempt (ATIF) in the run's directory.
fn save_trajectory(setup: &Setup<'_>, number: u32, solved: &WorkspaceSolution) -> Result<PathBuf> {
    let path = setup
        .run_dir
        .join("trajectories")
        .join(format!("attempt-{number}.atif.json"));
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    splinter_sdk::agent::sven::atif::persist::write_trajectory_atomic(
        &path,
        &solved.solution.trajectory,
        None,
    )
    .with_context(|| format!("writing {}", path.display()))?;
    Ok(path)
}

fn emit_judgement(setup: &Setup<'_>, judgement: &Judgement, patch: &Path) -> Result<()> {
    setup.tracer.emit(
        "candidate",
        None,
        json!({"changed_files": judgement.changes, "patch": patch,
               "baseline": setup.contract.baseline_rev}),
    )?;
    for check in &judgement.checks {
        setup.tracer.emit(
            "validation",
            None,
            json!({"name": check.name, "command": check.command, "passed": check.passed,
                   "exit_code": check.exit_code, "timed_out": check.timed_out,
                   "duration_ms": check.duration_ms, "stdout_tail": check.stdout_tail,
                   "stderr_tail": check.stderr_tail,
                   "covers": "a fresh copy of the snapshot with this attempt's patch applied"}),
        )?;
    }
    Ok(())
}

/// What decides how an attempt is classified.
struct Facts<'a> {
    conclusion: &'a str,
    capped: bool,
    repeated: Option<&'a str>,
    cancelled: bool,
    over_budget: bool,
    provider_failed: bool,
    exposed: bool,
    judgement: &'a Judgement,
}

/// Why an attempt was not accepted, if it was not. The causes are checked in
/// a fixed order: what stopped the work comes before how the work was
/// judged, and a worker that saw the supervisor's material is rejected
/// whatever its checks said.
fn classify(facts: &Facts<'_>) -> Option<String> {
    if facts.cancelled {
        return Some("cancelled".into());
    }
    if facts.over_budget {
        return Some("cost_limit".into());
    }
    if facts.provider_failed {
        return Some("provider_error".into());
    }
    if facts.exposed {
        return Some("hidden_material_exposed".into());
    }
    match &facts.judgement.refusal {
        Some(Refusal::OffLimits(paths)) => {
            return Some(format!("protected_path_modified: {}", paths.join(", ")))
        }
        Some(Refusal::RunnerConfigurationChanged(paths)) => {
            return Some(format!("runner_configuration_changed: {}", paths.join(", ")))
        }
        Some(Refusal::DoesNotApply) => return Some("patch_does_not_apply".into()),
        None => {}
    }
    if facts.judgement.all_passed() {
        return None;
    }
    if facts.capped {
        return Some("tool_call_limit".into());
    }
    if let Some(call) = facts.repeated {
        return Some(format!(
            "stagnation: the call `{call}` returned the same answer again and again, so the \
             attempt was stopped; do not repeat it, make the change itself"
        ));
    }
    match facts.conclusion {
        "Timeout" => return Some("time_limit".into()),
        "BudgetExhausted" => return Some("token_limit".into()),
        "AgentError" => return Some("agent_error".into()),
        _ => {}
    }
    if facts.judgement.checks.is_empty() {
        return Some("no_acceptance_checks_ran".into());
    }
    let failed: Vec<&str> = facts
        .judgement
        .checks
        .iter()
        .filter(|c| !c.passed)
        .map(|c| c.name.as_str())
        .collect();
    Some(format!("acceptance_failed: {}", failed.join(", ")))
}

/// The feedback the next attempt is shown: what failed and, for a check the
/// worker may see, the output that shows it. A hidden check is named and
/// nothing more: neither its command nor what it printed. Never a patch.
fn diagnostic(contract: &Contract, why: &str, judgement: &Judgement, said: Option<&str>) -> String {
    let mut text = format!("Failure: {why}.\n");
    if why.starts_with("protected_path_modified") {
        text.push_str("You changed files you were told not to; leave them as they are.\n");
    }
    for check in judgement.checks.iter().filter(|c| !c.passed) {
        let shown = contract
            .acceptance
            .iter()
            .any(|c| c.name == check.name && c.visible);
        text.push_str(&failed_check(check, shown));
    }
    if let Some(said) = said.filter(|s| !s.trim().is_empty()) {
        text.push_str(&format!("You ended by saying: {}\n", tail(said)));
    }
    text
}

/// One failed check as feedback.
fn failed_check(check: &CheckResult, shown: bool) -> String {
    if !shown {
        return format!("Check `{}` failed.\n", check.name);
    }
    let output = if check.stderr_tail.trim().is_empty() {
        &check.stdout_tail
    } else {
        &check.stderr_tail
    };
    format!(
        "Check `{}` ({}) failed{}:\n{}\n",
        check.name,
        check.command,
        check
            .exit_code
            .map_or(String::new(), |c| format!(" with exit code {c}")),
        tail(output)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::{Check, Limits, CONTRACT_SCHEMA};

    fn contract() -> Contract {
        Contract {
            schema: CONTRACT_SCHEMA,
            task: "fix the bug".into(),
            repository: "/r".into(),
            baseline_rev: "abc".into(),
            acceptance: vec![Check {
                name: "unit".into(),
                command: "make test".into(),
                visible: true,
            }],
            protected: vec!["tests/".into()],
            limits: Limits {
                attempt_secs: 1,
                total_secs: 1,
                max_output_tokens: 1,
                max_tool_calls: 1,
                max_repeated_results: 0,
                max_attempts: 1,
                provider_retries: 0,
                follow_ups: 0,
                max_cost_usd: None,
            },
            model: "local:x".into(),
            allow_api_models: false,
            system_prompt_digest: String::new(),
            system_prompt_source: "built-in".into(),
            definitions_digest: String::new(),
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

    fn judged(checks: Vec<CheckResult>, refusal: Option<Refusal>) -> Judgement {
        Judgement {
            refusal,
            checks,
            changes: Vec::new(),
        }
    }

    fn why(judgement: &Judgement, tweak: impl FnOnce(&mut Facts<'_>)) -> Option<String> {
        let mut facts = Facts {
            conclusion: "Success",
            capped: false,
            repeated: None,
            cancelled: false,
            over_budget: false,
            provider_failed: false,
            exposed: false,
            judgement,
        };
        tweak(&mut facts);
        classify(&facts)
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
        assert!(
            compose_prompt(&c, None).contains("make a first complete edit early"),
            "the working style is part of every attempt's prompt"
        );
    }

    #[test]
    fn failures_are_classified_by_their_cause_in_a_fixed_order() {
        let red = judged(vec![failed("unit", "E")], None);
        assert_eq!(why(&red, |f| f.cancelled = true).unwrap(), "cancelled");
        assert_eq!(why(&red, |f| f.over_budget = true).unwrap(), "cost_limit");
        assert_eq!(
            why(&red, |f| f.provider_failed = true).unwrap(),
            "provider_error"
        );
        assert_eq!(why(&red, |f| f.capped = true).unwrap(), "tool_call_limit");
        assert_eq!(why(&red, |_| {}).unwrap(), "acceptance_failed: unit");
        assert_eq!(
            why(&red, |f| f.conclusion = "Timeout").unwrap(),
            "time_limit"
        );
        let fenced = judged(Vec::new(), Some(Refusal::OffLimits(vec!["tests/a".into()])));
        assert!(why(&fenced, |_| {})
            .unwrap()
            .starts_with("protected_path_modified"));
        let green = judged(
            vec![CheckResult {
                passed: true,
                ..failed("unit", "")
            }],
            None,
        );
        assert!(why(&green, |_| {}).is_none());
        assert_eq!(
            why(&green, |f| f.exposed = true).unwrap(),
            "hidden_material_exposed",
            "a worker that saw the supervisor's material is rejected whatever its checks said"
        );
    }

    #[test]
    fn feedback_shows_the_output_of_a_shown_check_and_never_a_patch() {
        let c = contract();
        let judgement = judged(vec![failed("unit", "AssertionError: 1 != 2")], None);
        let text = diagnostic(&c, "acceptance_failed: unit", &judgement, Some("done"));
        assert!(text.contains("AssertionError: 1 != 2") && text.contains("exit code 1"));
        assert!(!text.contains("diff --git"));
    }

    #[test]
    fn feedback_names_a_hidden_check_and_shows_neither_its_command_nor_its_output() {
        let mut c = contract();
        c.acceptance[0].visible = false;
        let judgement = judged(vec![failed("unit", "AssertionError: 1 != 2")], None);
        let text = diagnostic(&c, "acceptance_failed: unit", &judgement, None);
        assert!(text.contains("Check `unit` failed."));
        assert!(!text.contains("AssertionError") && !text.contains("make test"));
    }
}
