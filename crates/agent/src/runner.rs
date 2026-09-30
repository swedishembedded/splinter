// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements delegated-task coding agents whose every
// observable action is traced and handed back as checkable evidence. If your
// team needs expertise in agent orchestration, budget enforcement, or audit
// evidence, you can procure our services by sending an email to
// info@swedishembedded.com.

//! One delegated attempt, end to end.
//!
//! The flow is: capture the task contract, snapshot the workspace, give the
//! task to sven's engine and let its tool loop run, trace everything the
//! kernel reports, enforce the configured limits, execute the completion
//! checks, and write the structured outcome. The engine is sven's own - there
//! is no second agent loop here; this file is the harness around it.
//!
//! Interruption semantics, stated because they decide what a resume means:
//! a timeout or Ctrl-C drops the in-flight turn. Effects of tool calls the
//! kernel already executed are done - that is what "executed" means - and
//! they are in the trace. A resumed attempt rebuilds its continuation prompt
//! from that trace and is told to reconcile the workspace against it before
//! acting, which is why recovery never repeats a non-idempotent action
//! blindly.

use crate::events::{collect, Tally};
use crate::outcome::{
    capture_changed_files, capture_tool_evidence, ChangedFile, Check, Outcome, Status,
};
use splinter_policy::{local_model_name, panic_message, LocalQwen, ModelSelection};
use splinter_store::clock::utc_now;
use splinter_store::runs::{Limits, RunManifest};
use splinter_store::trace::Trace;
use splinter_store::{write_atomic, StateRoot};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use sven_sdk::{config, ApprovalPolicy, Engine};

/// Everything one attempt needs, as configured by the caller.
#[derive(Clone, Debug)]
pub struct AttemptOptions {
    /// Directory the agent works in; it becomes the process working
    /// directory for the attempt, and its git state is the changed-file
    /// evidence when it is a repository.
    pub workspace: PathBuf,
    /// The task in plain language, as the agent receives it. A resume wraps
    /// it in a continuation prompt naming the tool calls already completed.
    pub task: String,
    /// Completion checks the attempt executes itself after the turn, as its
    /// own validation evidence. Repeatable.
    pub checks: Vec<String>,
    /// The model the agent runs on: local weights served in-process by
    /// default, a model reached over an API when the caller opted in.
    pub model: ModelSelection,
    /// Wall-clock seconds the turn may run before it is dropped and the
    /// attempt ends as `timeout`.
    pub timeout_secs: u64,
    /// Tool-call rounds the agent may make in the turn; `None` keeps sven's
    /// configured default.
    pub max_tool_rounds: Option<u32>,
    /// Attempts the run may make, resumes included; recorded at `run` and
    /// enforced by `resume` from the manifest.
    pub max_attempts: u32,
    /// Usage limits, raced against the turn.
    pub budget: splinter_store::runs::Budget,
    /// Whether the caller permitted a model reached over an API - recorded
    /// with the task contract, so a remote call is attributable to an
    /// explicit opt-in.
    pub allow_api_models: bool,
    /// Capture the exact model input at the wire (needs a proxied upstream).
    pub record_input: bool,
}

/// Runs one attempt of `task` in `workspace`, from a fresh conversation.
pub fn run(root: &StateRoot, options: AttemptOptions) -> anyhow::Result<(Outcome, RunManifest)> {
    let workspace = options.workspace.canonicalize()?;
    let run_id = splinter_store::new_run_id();
    let dir = root.run_dir(&run_id);
    let trace = Arc::new(Trace::open(&dir, &run_id, 1)?);

    let head = git_head(&workspace).unwrap_or_else(|| "not a git repository".into());
    let mut manifest = RunManifest {
        schema: 1,
        run_id: run_id.clone(),
        workspace: workspace.display().to_string(),
        task: options.task.clone(),
        status: "pending".into(),
        attempts: 1,
        started_ts: utc_now(),
        updated_ts: utc_now(),
        model: options.model.identity(),
        base_url: remote_base_url(&options.model),
        local_adapter: options.model.local().and_then(|w| w.adapter.clone()),
        limits: Limits {
            timeout_secs: options.timeout_secs,
            max_tool_rounds: options.max_tool_rounds,
            max_attempts: Some(options.max_attempts),
            budget: options.budget,
        },
    };
    save_manifest(&dir, &manifest)?;

    let mut contract = serde_json::json!({
        "workspace": manifest.workspace,
        "head": head,
        "task": options.task,
        "checks": options.checks,
        "limits": manifest.limits,
        "model": manifest.model,
        "allow_api_models": options.allow_api_models,
        "splinter_version": env!("CARGO_PKG_VERSION"),
    });
    trace.event("task_received", &mut contract)?;
    record_workspace_baseline(&trace, &workspace)?;

    finish(
        &dir,
        &run_id,
        &mut manifest,
        options,
        None,
        trace,
        Instant::now(),
    )
}

/// Resumes a run from its last valid checkpoint, as a further attempt on the
/// same trace.
pub fn resume(
    root: &StateRoot,
    run_id: &str,
    options: AttemptOptions,
) -> anyhow::Result<(Outcome, RunManifest)> {
    let dir = root.run_dir(run_id);
    let manifest = splinter_store::runs::read_manifest(root, run_id)?;
    splinter_store::runs::ensure_attempt_allowed(&manifest)?;
    // A cancel aimed at the previous attempt must not stop this one.
    splinter_store::runs::clear_cancel(&dir)?;
    let checkpoint_path = dir.join("checkpoint").join("state.json");
    let text = std::fs::read_to_string(&checkpoint_path).map_err(|e| {
        anyhow::anyhow!(
            "{}: {e}\nThis run has no resumable checkpoint yet - a checkpoint is written when an \
             attempt's turn ends, whichever way it ends.",
            checkpoint_path.display()
        )
    })?;
    let state: sven_sdk::AgentState = serde_json::from_str(&text)
        .map_err(|e| anyhow::anyhow!("{}: {e}", checkpoint_path.display()))?;

    let trace = Arc::new(Trace::open(&dir, run_id, manifest.attempts + 1)?);
    let prior = summarize_prior(&dir)?;
    trace.event(
        "resumed",
        &mut serde_json::json!({
            "attempt": manifest.attempts + 1,
            "prior_status": manifest.status,
            "completed_tool_calls_before": prior.len(),
            "budget": options.budget,
            "allow_api_models": options.allow_api_models,
        }),
    )?;
    let mut manifest = manifest;
    manifest.attempts += 1;
    manifest.status = "pending".into();
    manifest.updated_ts = utc_now();
    // Manifests written before local_adapter existed read as None; a resume
    // that does know the adapter backfills it, so every later resume
    // inherits the configuration this attempt actually serves from.
    if manifest.local_adapter.is_none() {
        manifest.local_adapter = options.model.local().and_then(|w| w.adapter.clone());
    }
    save_manifest(&dir, &manifest)?;

    let mut opts = options;
    opts.task = continuation_prompt(&opts.task, &prior);
    finish(
        &dir,
        run_id,
        &mut manifest,
        opts,
        Some(state),
        trace,
        Instant::now(),
    )
}

/// The tool calls the previous attempt completed, as the trace recorded them
/// - the evidence recovery reconciles against.
fn summarize_prior(dir: &std::path::Path) -> anyhow::Result<Vec<String>> {
    let events = splinter_store::trace::read_events(dir)?;
    let mut completed = Vec::new();
    for event in events {
        if event.get("type").and_then(|v| v.as_str()) != Some("tool_finished") {
            continue;
        }
        let payload = event.get("payload").cloned().unwrap_or_default();
        let name = payload.get("name").and_then(|v| v.as_str()).unwrap_or("?");
        let is_error = payload
            .get("is_error")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        completed.push(format!(
            "{name} ({})",
            if is_error { "errored" } else { "completed" }
        ));
    }
    Ok(completed)
}

fn continuation_prompt(task: &str, prior: &[String]) -> String {
    if prior.is_empty() {
        return format!(
            "{task}\n\nA previous attempt at this task was interrupted before it made any \
             recorded tool call. The workspace may have been modified anyway; check its actual \
             state before acting, and continue the task."
        );
    }
    format!(
        "{task}\n\nA previous attempt at this task was interrupted. These are the tool calls it \
         completed, from its trace:\n{}.\nAny of them may have taken effect partially or fully; \
         reconcile the workspace's actual state against that record before acting, and continue \
         the task from there.",
        prior.join("; ")
    )
}

/// The attempt itself, shared by `run` and `resume`. Runs the async engine
/// work to completion, then writes the outcome and closes the manifest.
fn finish(
    dir: &Path,
    run_id: &str,
    manifest: &mut RunManifest,
    options: AttemptOptions,
    prior_state: Option<sven_sdk::AgentState>,
    trace: Arc<Trace>,
    started: Instant,
) -> anyhow::Result<(Outcome, RunManifest)> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    // The agent's tools run relative to the process working directory, so
    // the workspace has to be it. Attempts are sequential for this reason.
    let previous_cwd = std::env::current_dir()?;
    std::env::set_current_dir(&options.workspace)?;
    let result = runtime.block_on(drive(
        dir,
        run_id,
        options,
        prior_state,
        Arc::clone(&trace),
        started,
    ));
    std::env::set_current_dir(previous_cwd)?;

    // Dropping a multi-thread runtime waits for every blocking task it ever
    // spawned; one wedged backend thread would then hang the process after
    // the outcome is already on disk. Bound the wait instead - past this
    // point everything the attempt owed the store has been written, so a
    // straggler is safe to abandon.
    runtime.shutdown_timeout(Duration::from_secs(10));

    let outcome = result?;
    outcome.save(dir)?;
    manifest.status = outcome.status.as_str().into();
    manifest.updated_ts = utc_now();
    save_manifest(dir, manifest)?;
    Ok((outcome, manifest.clone()))
}

/// Records where the workspace stood before acting, so a dirty file the
/// caller already had is never attributed to the attempt.
fn record_workspace_baseline(trace: &Trace, workspace: &std::path::Path) -> anyhow::Result<()> {
    let out = std::process::Command::new("git")
        .args(["status", "--porcelain"])
        .current_dir(workspace)
        .output();
    let mut payload = match out {
        Ok(o) if o.status.success() => {
            let text = String::from_utf8_lossy(&o.stdout).into_owned();
            serde_json::json!({ "dirty_before": text.lines().count(), "status": text })
        }
        _ => serde_json::json!({ "dirty_before": 0, "status": "not a git repository" }),
    };
    trace.event("workspace_baseline", &mut payload).map(|_| ())
}

/// The endpoint a remote model is reached at, as the manifest records it.
fn remote_base_url(model: &ModelSelection) -> Option<String> {
    match model {
        ModelSelection::Remote(remote) => remote.base_url.clone(),
        ModelSelection::Local(_) => None,
    }
}

fn git_head(workspace: &std::path::Path) -> Option<String> {
    let out = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(workspace)
        .output()
        .ok()?;
    if out.status.success() {
        Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
    } else {
        None
    }
}

fn save_manifest(dir: &Path, manifest: &RunManifest) -> anyhow::Result<()> {
    let path = dir.join("run.json");
    write_atomic(&path, &serde_json::to_string_pretty(manifest)?)
        .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))
}

async fn drive(
    dir: &Path,
    run_id: &str,
    options: AttemptOptions,
    prior_state: Option<sven_sdk::AgentState>,
    trace: Arc<Trace>,
    started: Instant,
) -> anyhow::Result<Outcome> {
    let mut settings = config::load(None)?;
    if let ModelSelection::Remote(remote) = &options.model {
        remote.apply_to(&mut settings)?;
    }
    if let Some(rounds) = options.max_tool_rounds {
        settings.agent.max_tool_rounds = rounds;
    }

    // Through the recorder when one is asked for, so the captured request is
    // the one the server really answered.
    let recorder = if options.record_input {
        match remote_base_url(&options.model)
            .as_deref()
            .and_then(splinter_lab::upstream_of)
        {
            Some(upstream) => match splinter_lab::Recorder::start(&upstream).await {
                Ok(recorder) => {
                    settings.model.base_url = Some(recorder.base_url().to_string());
                    Some(recorder)
                }
                Err(e) => {
                    let mut note =
                        serde_json::json!({ "note": format!("not recording model input: {e}") });
                    trace.event("record_input_skipped", &mut note)?;
                    None
                }
            },
            None => {
                let mut note = serde_json::json!({
                    "note": "--record-input needs --base-url to name a proxied upstream",
                });
                trace.event("record_input_skipped", &mut note)?;
                None
            }
        }
    } else {
        None
    };

    // The engine, with its model: the local one loaded in-process when this
    // attempt is a local one, else the provider the configuration names. The
    // load is the expensive step (weights to device), so it is traced as its
    // own event - a run that died during model load must be distinguishable
    // from one that never had a model.
    let engine = {
        let builder = Engine::builder().config(settings);
        // Kept outside the engine: when the attempt ends before the turn
        // does (timeout, interrupt) the runner stops the in-flight local
        // generation through it, so the device is quiet before the process
        // exits.
        let local: Option<Arc<LocalQwen>> = if let Some(weights) = options.model.local() {
            let t0 = std::time::Instant::now();
            // A device that cannot hold the requested context surfaces as a
            // wgpu error, which brain's backend reports by panicking. Load
            // under catch_unwind so that becomes this attempt's recorded,
            // non-crashing failure - the trace keeps its model_load_failed
            // event and the CLI exits with an error instead of a signal.
            let name = local_model_name(&weights.base);
            let load = || LocalQwen::load(weights, &name);
            let provider = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(load)) {
                Ok(Ok(provider)) => Ok(provider),
                Ok(Err(e)) => Err(format!("{e:#}")),
                Err(panic) => Err(format!(
                    "device panic during model load: {}",
                    panic_message(&panic)
                )),
            }
            .map_err(|error| {
                let mut note = serde_json::json!({
                    "error": error,
                    "weights": weights.base.display().to_string(),
                    "adapter": weights.adapter.as_deref().map(|a| a.display().to_string()),
                    "context_tokens": weights.context_tokens,
                });
                let _ = trace.event("model_load_failed", &mut note);
                anyhow::anyhow!("loading the local model: {error}")
            })?;
            let provider = Arc::new(provider);
            let mut note = serde_json::json!({
                "model": name,
                "weights": weights.base.display().to_string(),
                "adapter": weights.adapter.as_deref().map(|a| a.display().to_string()),
                "context_tokens": weights.context_tokens,
                "load_secs": t0.elapsed().as_secs(),
            });
            trace.event("model_loaded", &mut note)?;
            Some(Arc::clone(&provider))
        } else {
            None
        };
        let builder = match &local {
            Some(provider) => builder
                .model_provider(Arc::clone(provider) as Arc<dyn sven_sdk::model::ModelProvider>),
            None => builder,
        };
        let engine = builder
            .approvals(ApprovalPolicy::AutoApprove)
            .build()
            .map_err(|e| anyhow::anyhow!("building the engine: {e}"))?;
        (engine, local)
    };
    let (engine, local_provider) = engine;

    let mut agent = match prior_state {
        Some(state) => engine
            .resume(state)
            .map_err(|e| anyhow::anyhow!("resuming: {e}"))?,
        None => engine.agent("agent"),
    };
    let events = agent.events();
    let tally = Arc::new(Tally::default());

    let collector = {
        let tally = Arc::clone(&tally);
        let trace = trace.clone();
        tokio::spawn(collect(events, trace, tally))
    };

    // The limits race the turn. `select!` owns each future it is given, so
    // on a raced arm the in-flight turn is dropped when the select ends -
    // which is what makes the interruption real: nothing further runs, and
    // what already ran is in the trace.
    let task = options.task.clone();
    let sent: Option<Result<String, sven_sdk::CallError>>;
    let raced: Option<Status>;
    let mut stop_reason: Option<String> = None;
    tokio::select! {
        res = agent.send(&task) => { sent = Some(res.map(|outcome| outcome.reply)); raced = None; }
        _ = tokio::time::sleep(Duration::from_secs(options.timeout_secs)) => { sent = None; raced = Some(Status::Timeout); }
        _ = tokio::signal::ctrl_c() => { sent = None; raced = Some(Status::Cancelled); }
        (status, reason) = watch_limits(dir, options.budget, &tally, started) => {
            sent = None;
            raced = Some(status);
            stop_reason = Some(reason);
        }
    }
    if let (Some(status), Some(reason)) = (raced, &stop_reason) {
        let mut note = serde_json::json!({ "status": status.as_str(), "reason": reason });
        trace.event("limit_fired", &mut note)?;
    }

    // Write the turn's durable state before the agent goes away - whichever
    // way the turn ended, because a failed episode is exactly the one whose
    // record explains why.
    let transcript = agent.state().transcript();
    let transcript_path = dir.join("transcript.json");
    write_atomic(
        &transcript_path,
        &serde_json::to_string_pretty(&transcript)?,
    )?;
    let checkpoint_text = serde_json::to_string(&agent.state().clone())?;
    std::fs::create_dir_all(dir.join("checkpoint"))?;
    write_atomic(&dir.join("checkpoint").join("state.json"), &checkpoint_text)?;

    // On a raced end the turn was dropped mid-generation; on an errored end
    // (the engine's own stream watchdog, say) the turn future returned while
    // the generation thread is still running. Either way the process must
    // not tear the device down under it - that is the crash this call
    // exists to prevent - so stop unconditionally: with nothing in flight
    // it is a no-op.
    if let Some(provider) = &local_provider {
        let stopped = provider.stop_generation(Duration::from_secs(60));
        let mut note = serde_json::json!({ "stopped": stopped });
        let _ = trace.event("generation_stopped", &mut note);
    }

    // Close the agent so the broadcast channel drains, then wait (bounded)
    // for the collector to write everything still buffered. A trace that
    // ends before its last tool call would not be evidence.
    drop(agent);
    let _ = tokio::time::timeout(Duration::from_secs(5), collector).await;

    // The exact model input, when it was being captured.
    if let Some(recorder) = &recorder {
        let path = splinter_lab::capture_path(dir);
        if let Err(e) = recorder.dump(&path).await {
            let mut note = serde_json::json!({ "note": format!("could not write the captured model input: {e}") });
            trace.event("record_input_failed", &mut note)?;
        }
    }

    build_outcome(
        run_id,
        dir,
        &options,
        raced,
        stop_reason,
        sent,
        &tally,
        &trace,
        started,
    )
    .await
}

/// Resolves when the attempt must stop for a reason other than its own
/// turn, the timeout or Ctrl-C: a cancel requested from another process, or
/// a spent usage budget. Polled, because both are read off shared state
/// (the run directory, the tally) that no one signals.
async fn watch_limits(
    dir: &std::path::Path,
    budget: splinter_store::runs::Budget,
    tally: &Tally,
    started: Instant,
) -> (Status, String) {
    let mut tick = tokio::time::interval(Duration::from_millis(500));
    loop {
        tick.tick().await;
        if splinter_store::runs::cancel_requested(dir) {
            return (Status::Cancelled, "cancel requested".into());
        }
        let usage = tally.usage(started.elapsed().as_secs());
        if let Some(reason) = crate::budget::exhausted(&budget, &usage, tally.usage_reports()) {
            return (Status::BudgetExhausted, reason);
        }
    }
}

/// Builds the structured outcome from what the run observed: kernel events,
/// completion checks, and workspace git state. Nothing here reads the
/// model's own claims about its work except as the verbatim reply.
#[allow(clippy::too_many_arguments)]
async fn build_outcome(
    run_id: &str,
    dir: &std::path::Path,
    options: &AttemptOptions,
    raced: Option<Status>,
    stop_reason: Option<String>,
    sent: Option<Result<String, sven_sdk::CallError>>,
    tally: &Tally,
    trace: &Trace,
    started: Instant,
) -> anyhow::Result<Outcome> {
    let status = raced.unwrap_or(match &sent {
        Some(Ok(_)) => Status::Completed,
        _ => Status::Errored,
    });

    let mut unresolved = Vec::new();
    let reply_text = sent
        .as_ref()
        .and_then(|r| r.as_ref().ok())
        .map(String::as_str);
    if status == Status::Completed && crate::outcome::reply_is_blank(reply_text) {
        unresolved.push(
            "the turn ended with an empty final reply; whatever the model meant to say about \
             its work is not recorded here"
                .into(),
        );
    }
    let mut diff_artifacts: Vec<String> = Vec::new();
    if let Some(reason) = stop_reason {
        unresolved.push(format!("the attempt was stopped: {reason}"));
    }
    if matches!(
        status,
        Status::Timeout | Status::Cancelled | Status::BudgetExhausted
    ) {
        unresolved.push(
            "the attempt's turn was interrupted; tool calls completed before the interrupt took \
             effect and are in the trace - use `resume` to reconcile and continue"
                .into(),
        );
    }
    if let Some(Err(e)) = &sent {
        unresolved.push(format!("the turn errored: {e}"));
    }
    if tally.asked_questions() > 0 {
        unresolved.push(
            "the agent asked a question; a headless run cannot answer it, so any work that \
             depended on the answer did not happen"
                .into(),
        );
    }

    let mut checks = Vec::new();
    if status == Status::Completed || status == Status::Failed {
        for (n, command) in options.checks.iter().enumerate() {
            let check = run_check(command, &options.workspace, dir, n).await?;
            let mut payload = serde_json::json!({
                "command": check.command,
                "exit": check.exit,
                "output_ref": check.output_ref,
            });
            trace.event("check", &mut payload)?;
            if !check.passed {
                unresolved.push(format!("completion check failed: {command}"));
            }
            checks.push(check);
        }
    } else if !options.checks.is_empty() {
        unresolved.push(format!(
            "{} completion check(s) did not run: the turn did not complete",
            options.checks.len()
        ));
    }

    // Changed files: git when the workspace is a repository, else the file
    // paths the kernel's mutating tools acted on. The basis is named in the
    // outcome either way, so no reader mistakes one for the other.
    let (changed, basis): (Vec<ChangedFile>, String) =
        match capture_changed_files(&options.workspace) {
            Ok((files, basis)) if basis == "git" => {
                let mut artifacts = vec!["events.jsonl".to_string(), "transcript.json".to_string()];
                if let Ok(out) = std::process::Command::new("git")
                    .args(["diff", "HEAD"])
                    .current_dir(&options.workspace)
                    .output()
                {
                    if out.status.success() {
                        let text = String::from_utf8_lossy(&out.stdout).into_owned();
                        let _ = write_atomic(&dir.join("workspace.diff"), &text);
                        artifacts.push("workspace.diff".into());
                    }
                }
                diff_artifacts = artifacts;
                (files, basis)
            }
            Ok((_, _)) | Err(_) => {
                let paths = tally.mutated_paths();
                (
                    capture_tool_evidence(&options.workspace, &paths)?,
                    "tool_evidence".into(),
                )
            }
        };

    let usage = tally.usage(started.elapsed().as_secs());
    let artifacts = if basis == "git" {
        diff_artifacts
    } else {
        vec!["events.jsonl".into(), "transcript.json".into()]
    };
    if usage.cost_usd.is_none() {
        unresolved.push("usage cost was not measured (the provider reported no price)".into());
    }

    Ok(Outcome {
        schema: Outcome::SCHEMA,
        run_id: run_id.to_string(),
        status: crate::outcome::verdict(status, &checks),
        reply: sent.and_then(|r| r.ok()),
        changed_files: changed,
        changed_files_basis: basis,
        checks,
        tool_failures: tally.tool_failures(),
        usage,
        unresolved,
        artifacts,
    })
}

/// One completion check, executed by the attempt itself. The full output
/// always goes to its own artifact file - a passing build's output is
/// evidence too.
async fn run_check(
    command: &str,
    workspace: &std::path::Path,
    dir: &std::path::Path,
    n: usize,
) -> anyhow::Result<Check> {
    let out = tokio::process::Command::new("sh")
        .arg("-c")
        .arg(command)
        .current_dir(workspace)
        .output()
        .await
        .map_err(|e| anyhow::anyhow!("running {command:?}: {e}"))?;
    let exit = out.status.code().unwrap_or(-1);
    let mut output = String::from_utf8_lossy(&out.stdout).into_owned();
    output.push_str(&String::from_utf8_lossy(&out.stderr));
    let reference = dir.join("artifacts").join(format!("check-{n}.txt"));
    let _ = write_atomic(&reference, &output);
    Ok(Check {
        command: command.to_string(),
        exit,
        passed: exit == 0,
        output_ref: Some(reference.display().to_string()),
    })
}
