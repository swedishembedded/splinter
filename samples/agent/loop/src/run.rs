// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements self-improving coding agents whose every
// step is auditable, for its clients. If your team needs expertise in agent
// evaluation or locally operated coding agents, you can procure our
// services by sending an email to info@swedishembedded.com.

//! A whole run: contract, attempts, checkpoints, outcome.
//!
//! A run is recorded by Splinter's own run machinery (its id, its stages,
//! its cross-process cancel); the loop adds the contract, the event stream,
//! the checkpoints and the outcome under its own directory. After every
//! attempt a checkpoint is written. No tree outlives an attempt (each is a
//! worker in a fresh copy of the baseline's snapshot), so a resumed run has
//! no unknown partial state to distrust: it says so in the trace and goes on
//! with the next attempt.

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context as _, Result};
use serde_json::json;
use splinter_sdk::agent::solve::Model;
use splinter_sdk::knowledge::capture::DirtyPolicy;
use splinter_sdk::runs::Recorder;
use splinter_sdk::sandbox::Patch;
use splinter_sdk::vocabulary::digest::Digest;
use splinter_sdk::vocabulary::model_ref::ModelRef;
use splinter_sdk::{Context, Error as SplinterError, Splinter};

use crate::attempt::{run_attempt, Judged, Setup};
use crate::contract::{Contract, CONTRACT_SCHEMA};
use crate::outcome::{
    Artifacts, AttemptSummary, Checkpoint, Outcome, Status, Usage, OUTCOME_SCHEMA,
};
use crate::store::{
    read_json, write_atomic, write_json, LoopHome, CHECKPOINT_FILE, CONTRACT_FILE, OUTCOME_FILE,
    PATCH_FILE,
};
use crate::trace::{Tracer, EVENTS_FILE};
use crate::workspace::Workspace;

/// The provider name of a brain server on this machine.
const LOCAL_SERVER: &str = "brain";

/// What to do: a new run under a contract, or a continuation.
pub enum Request {
    /// A run under `Contract`.
    New(Box<Contract>),
    /// The run with this id, from its last checkpoint.
    Resume(String),
}

/// Runs `request` as a recorded run and returns its outcome.
pub fn execute(splinter: &Splinter, home: &LoopHome, request: Request) -> Result<Outcome> {
    let (command, arguments) = match &request {
        Request::New(contract) => (
            "agent-loop",
            json!({"task": contract.task, "model": contract.model}),
        ),
        Request::Resume(id) => ("agent-loop-resume", json!({"run": id})),
    };
    let recorded = splinter
        .run(command, &arguments, |ctx, rec| {
            drive(ctx, rec, home, request).map_err(|e| SplinterError::Refused(format!("{e:#}")))
        })
        .map_err(|e| anyhow!("{e}"))?;
    Ok(recorded.report)
}

/// The system prompt in effect and where it came from.
pub fn system_prompt(home: &LoopHome) -> Result<(String, String)> {
    let path = home.system_prompt_override();
    if path.exists() {
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        return Ok((text, path.display().to_string()));
    }
    Ok((
        include_str!("../prompts/system.md").to_string(),
        "built-in".into(),
    ))
}

fn drive(
    ctx: &Context,
    rec: &mut Recorder<'_>,
    home: &LoopHome,
    request: Request,
) -> Result<Outcome> {
    let run = rec.id().to_string();
    let run_dir = home.run_dir(&run);
    let (contract, resumed_from) = match request {
        Request::New(contract) => (*contract, None),
        Request::Resume(old) => {
            let contract = read_json::<Contract>(&home.run_dir(&old).join(CONTRACT_FILE))
                .with_context(|| format!("run {old} has no contract to resume"))?;
            if contract.schema != CONTRACT_SCHEMA {
                bail!(
                    "run {old} was started under contract schema {}, this loop reads {CONTRACT_SCHEMA}; start a new run",
                    contract.schema
                );
            }
            (contract, Some(old))
        }
    };
    // A resumed run continues the original's directory, so its events,
    // checkpoint and checkout stay together.
    let run_id = resumed_from.clone().unwrap_or_else(|| run.clone());
    let run_dir = if resumed_from.is_some() {
        home.run_dir(&run_id)
    } else {
        run_dir
    };
    std::fs::create_dir_all(&run_dir)?;
    if resumed_from.is_none() {
        write_json(&run_dir.join(CONTRACT_FILE), &contract)?;
    }
    let tracer = Arc::new(Tracer::open(&run_dir, &run_id)?);
    tracer.emit(
        if resumed_from.is_some() { "run_resumed" } else { "run_started" },
        None,
        json!({"contract": contract, "splinter_run": run, "assistance": if contract.assisted() { "assisted" } else { "unaided" },
               "loop_version": env!("CARGO_PKG_VERSION"), "schema": contract.schema}),
    )?;

    let reference: ModelRef = contract
        .model
        .parse()
        .map_err(|e| anyhow!("the model reference {:?} is not valid: {e}", contract.model))?;
    // A model served by brain on this machine is reached over a socket but
    // is local inference with no price; every other `remote:` model is a
    // third party's, priced and subject to the cost cap.
    let served_here =
        matches!(&reference, ModelRef::Remote { provider, .. } if provider == LOCAL_SERVER);
    let remote = matches!(reference, ModelRef::Remote { .. }) && !served_here;
    let model = ctx.model(&reference).map_err(|e| anyhow!("{e}"))?;
    tracer.emit(
        "model_selected",
        None,
        json!({"requested": contract.model, "identity": model.identity, "remote": remote,
               "class": if remote { "remote" } else if served_here { "local_served" } else { "local" },
               "system_prompt_digest": contract.system_prompt_digest,
               "system_prompt_source": contract.system_prompt_source,
               "thinking": ctx.config().thinking,
               "temperature": ctx.config().served_temperature}),
    )?;
    let definitions = crate::definitions::effective(&contract.repository);
    tracer.emit(
        "definitions",
        None,
        json!({"digest": definitions.digest, "entries": definitions.entries}),
    )?;
    if !contract.definitions_digest.is_empty() && definitions.digest != contract.definitions_digest
    {
        tracer.emit(
            "definition_changed",
            None,
            json!({"what": "agent, skill, command or project definitions",
                   "note": "the definitions in effect are not the ones the contract recorded"}),
        )?;
    }
    let (system, _) = system_prompt(home)?;
    if Digest::sha256_of(system.as_bytes()).to_string() != contract.system_prompt_digest {
        tracer.emit(
            "definition_changed",
            None,
            json!({"what": "system prompt", "note": "the prompt in effect is not the one the contract recorded"}),
        )?;
    }

    // The repository is read here, once, at the baseline the contract names;
    // from now on every worker and every judgement starts from this snapshot.
    let workspace = Workspace::capture(
        ctx,
        &contract.repository,
        &contract.baseline_rev,
        DirtyPolicy::Allow,
    )?;
    tracer.emit(
        if resumed_from.is_some() {
            "reconcile"
        } else {
            "workspace_prepared"
        },
        None,
        json!({"repository": contract.repository, "baseline": contract.baseline_rev,
               "snapshot": workspace.snapshot_digest(),
               "note": "each attempt works in a fresh copy of the snapshot; nothing carries over from an interrupted attempt"}),
    )?;
    baseline_validation(&tracer, &contract, &workspace)?;

    let mut checkpoint = if resumed_from.is_some() {
        read_json::<Checkpoint>(&run_dir.join(CHECKPOINT_FILE))
            .unwrap_or_else(|_| fresh_checkpoint())
    } else {
        fresh_checkpoint()
    };
    let started = Instant::now()
        .checked_sub(Duration::from_secs_f64(checkpoint.elapsed_seconds))
        .unwrap_or_else(Instant::now);
    let mut feedback: Option<String> = None;
    let mut status = Status::Rejected;
    let mut termination = format!(
        "{} attempts ended without an accepted candidate",
        contract.limits.max_attempts
    );
    let mut last_patch = String::new();

    for number in checkpoint.attempts_done + 1..=contract.limits.max_attempts {
        if rec.check_cancelled().is_err() {
            status = Status::Cancelled;
            termination = "cancelled before an attempt began".into();
            break;
        }
        let elapsed = started.elapsed().as_secs();
        if elapsed >= contract.limits.total_secs {
            status = Status::LimitExhausted;
            termination = format!(
                "the total time limit of {}s was reached",
                contract.limits.total_secs
            );
            tracer.emit(
                "limit_reached",
                None,
                json!({"limit": "total_time", "secs": contract.limits.total_secs}),
            )?;
            break;
        }
        let setup = Setup {
            ctx,
            model: &model,
            remote,
            contract: &contract,
            workspace: &workspace,
            system: &system,
            tracer: &tracer,
            home,
            run_dir: &run_dir,
            run_cancel: rec.cancel_token(),
            remaining_secs: contract.limits.total_secs - elapsed,
        };
        // A failure of the loop itself is a recorded outcome, not a crash
        // that leaves a run with no result to read.
        let judged = match run_attempt(&setup, number, feedback.as_deref()) {
            Ok(judged) => judged,
            Err(e) => {
                tracer.emit(
                    "loop_error",
                    None,
                    json!({"attempt": number, "error": format!("{e:#}")}),
                )?;
                status = Status::Error;
                termination = format!("the loop failed during attempt {number}: {e:#}");
                break;
            }
        };
        let Judged {
            summary,
            accepted,
            cancelled,
            over_budget,
            feedback: next,
            provider_failed,
            patch,
        } = judged;
        absorb(&mut checkpoint.usage, &summary);
        checkpoint.attempts.push(summary.clone());
        checkpoint.attempts_done = number;
        checkpoint.elapsed_seconds = started.elapsed().as_secs_f64();
        write_json(&run_dir.join(CHECKPOINT_FILE), &checkpoint)?;
        rec.stage(
            &format!("attempt-{number}"),
            &json!({"accepted": accepted, "failure": summary.failure,
            "tool_calls": summary.tool_calls, "seconds": summary.seconds}),
        )
        .map_err(|e| anyhow!("{e}"))?;
        last_patch = patch;
        feedback = next;
        if accepted {
            status = Status::Accepted;
            termination = format!("attempt {number} passed every acceptance check");
            break;
        }
        if cancelled {
            status = Status::Cancelled;
            termination = format!("cancelled during attempt {number}");
            break;
        }
        if over_budget {
            status = Status::LimitExhausted;
            termination = "the cost limit was reached".into();
            break;
        }
        if provider_failed {
            status = Status::Error;
            termination = format!("the model's provider failed during attempt {number}");
            break;
        }
    }
    if status == Status::Rejected && started.elapsed().as_secs() >= contract.limits.total_secs {
        status = Status::LimitExhausted;
    }

    finish(FinishArgs {
        home,
        run: &run_id,
        run_dir: &run_dir,
        contract: &contract,
        model: &model,
        checkpoint: &mut checkpoint,
        status,
        termination,
        patch: &last_patch,
        snapshot: workspace.snapshot_digest(),
        wall: started.elapsed().as_secs_f64(),
        tracer: &tracer,
    })
}

struct FinishArgs<'a> {
    home: &'a LoopHome,
    run: &'a str,
    run_dir: &'a Path,
    contract: &'a Contract,
    model: &'a Model,
    checkpoint: &'a mut Checkpoint,
    status: Status,
    termination: String,
    patch: &'a str,
    snapshot: String,
    wall: f64,
    tracer: &'a Tracer,
}

fn finish(a: FinishArgs<'_>) -> Result<Outcome> {
    let last = a.checkpoint.attempts.last().cloned();
    let unresolved = unresolved_of(a.status, last.as_ref(), &a.termination);
    let patch_path = (!a.patch.is_empty()).then(|| a.run_dir.join(PATCH_FILE));
    if let Some(path) = &patch_path {
        write_atomic(path, a.patch.as_bytes())?;
    }
    let mut usage = a.checkpoint.usage.clone();
    usage.wall_seconds = a.wall;
    let trajectories = std::fs::read_dir(a.run_dir.join("trajectories"))
        .map(|d| {
            d.filter_map(|e| e.ok().map(|e| e.path()))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let outcome = Outcome {
        schema: OUTCOME_SCHEMA,
        run: a.run.to_string(),
        status: a.status,
        assistance: if a.contract.assisted() {
            "assisted".into()
        } else {
            "unaided".into()
        },
        model_requested: a.contract.model.clone(),
        model_identity: a.model.identity.clone(),
        attempts: a.checkpoint.attempts.clone(),
        changed_files: last
            .as_ref()
            .map(|s| s.changed_files.clone())
            .unwrap_or_default(),
        validation: last.as_ref().map(|s| s.checks.clone()).unwrap_or_default(),
        unresolved,
        usage,
        artifacts: Artifacts {
            run_dir: a.run_dir.to_path_buf(),
            events: a.run_dir.join(EVENTS_FILE),
            patch: patch_path,
            snapshot: a.snapshot,
            trajectories,
        },
        termination: a.termination,
    };
    a.tracer.emit(
        "run_finished",
        None,
        json!({"status": outcome.status, "termination": outcome.termination,
        "assistance": outcome.assistance, "usage": outcome.usage}),
    )?;
    a.checkpoint.finished = true;
    write_json(&a.run_dir.join(CHECKPOINT_FILE), a.checkpoint)?;
    write_json(&a.run_dir.join(OUTCOME_FILE), &outcome)?;
    let _ = a.home;
    Ok(outcome)
}

fn unresolved_of(status: Status, last: Option<&AttemptSummary>, termination: &str) -> Vec<String> {
    let mut unresolved = Vec::new();
    if status != Status::Accepted {
        unresolved.push(format!("not accepted: {termination}"));
    }
    if let Some(last) = last {
        if let Some(failure) = &last.failure {
            unresolved.push(format!("last attempt: {failure}"));
        }
        for check in last.checks.iter().filter(|c| !c.passed) {
            unresolved.push(format!("check {} failed", check.name));
        }
        if status == Status::Accepted && last.changed_files.is_empty() {
            unresolved.push(
                "accepted with no file changed: the checks already passed at the baseline".into(),
            );
        }
    }
    unresolved
}

fn fresh_checkpoint() -> Checkpoint {
    Checkpoint {
        attempts_done: 0,
        attempts: Vec::new(),
        usage: Usage::default(),
        elapsed_seconds: 0.0,
        finished: false,
    }
}

fn absorb(usage: &mut Usage, summary: &AttemptSummary) {
    usage.input_tokens += summary.input_tokens.unwrap_or(0);
    usage.output_tokens += summary.output_tokens.unwrap_or(0);
}

/// Judges the untouched baseline by the contract's checks and records the
/// results, so the record shows which of them the task has to turn from red
/// to green.
fn baseline_validation(tracer: &Tracer, contract: &Contract, workspace: &Workspace) -> Result<()> {
    let judged = workspace.judge(
        contract,
        &Patch {
            diff: Vec::new(),
            files: Vec::new(),
        },
    )?;
    for r in &judged.checks {
        tracer.emit(
            "baseline_validation",
            None,
            json!({"name": r.name, "passed": r.passed, "exit_code": r.exit_code,
            "stdout_tail": r.stdout_tail, "stderr_tail": r.stderr_tail}),
        )?;
    }
    if judged.all_passed() {
        tracer.emit("warning", None, json!({"message": "every acceptance check already passes at the baseline: the task may need no change"}))?;
    }
    Ok(())
}

/// Applies the patch of an accepted run to the repository the run started
/// on, which must have no uncommitted change to a tracked file and still be
/// at the baseline. The patch applies whole or not at all.
pub fn apply(splinter: &Splinter, home: &LoopHome, run: &str) -> Result<Vec<std::path::PathBuf>> {
    let dir = home.run_dir(run);
    let outcome: Outcome = read_json(&dir.join(OUTCOME_FILE))?;
    if outcome.status != Status::Accepted {
        bail!(
            "run {run} was not accepted ({:?}); nothing is applied",
            outcome.status
        );
    }
    let contract: Contract = read_json(&dir.join(CONTRACT_FILE))?;
    let patch_file = dir.join(PATCH_FILE);
    let ctx = splinter.context();
    let now = Workspace::capture(&ctx, &contract.repository, "HEAD", DirtyPolicy::Refuse)
        .with_context(|| format!("the patch {} was not applied", patch_file.display()))?;
    if now.commit != contract.baseline_rev {
        bail!(
            "{} is at {}, not the baseline {}; apply the patch {} by hand",
            contract.repository.display(),
            now.commit,
            contract.baseline_rev,
            patch_file.display()
        );
    }
    let diff =
        std::fs::read(&patch_file).with_context(|| format!("reading {}", patch_file.display()))?;
    now.env
        .apply(
            &contract.repository,
            &Patch {
                diff,
                files: Vec::new(),
            },
        )
        .map_err(|e| anyhow!("{e}"))
}
