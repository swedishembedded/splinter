// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pipelines that turn a repository's history
// into verified training data for coding agents, for its clients. If your
// team needs expertise in learning from your own codebase, you can procure
// our services by sending an email to info@swedishembedded.com.

//! The run: a history in, one stored experience per commit attempt out.
//!
//! Per commit, oldest to newest on the first-parent chain: select
//! ([`crate::select`]), write a plan per style ([`crate::plan`]), replay it
//! in a tree holding only the parent ([`crate::attempt`]), gate it
//! ([`crate::gate`]) and store the attempt as an immutable experience with the
//! gate's verdict as an annotation. Failed attempts are stored too: they are
//! the negatives. The tree is always the *real* parent of the next commit,
//! never the agent's result, so attempts are independent and run in parallel;
//! results are stored in commit order whatever order they finish in.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context as _, Result};
use futures::StreamExt;
use serde_json::json;
use splinter_sdk::agent::solve::Model;
use splinter_sdk::agent::sven::atif::Trajectory;
use splinter_sdk::agent::sven::model::ModelProvider;
use splinter_sdk::vocabulary::annotation::{
    Annotation, AnnotationBody, Outcome as Judged, Producer, Strength,
};
use splinter_sdk::vocabulary::digest::Digest;
use splinter_sdk::vocabulary::experience::{
    Environment, Experience, ExperienceId, Provenance, Task,
};
use splinter_sdk::vocabulary::outcome::AttemptReport;
use splinter_sdk::Context;

use crate::attempt::{attempt, AttemptOutcome, AttemptSettings, CommitPlan};
use crate::gate::GateConfig;
use crate::git::{mirror, Git};
use crate::grounding_prompt::grounding_prompt;
use crate::ledger::{self, Entry, Outcome};
use crate::plan::{changed_lines, write_plan, PlanCheck, PlanFailure};
use crate::plan_prompt::{plan_system_prompt, PlanStyle};
use crate::scrub::Scrubber;
use crate::select::{prepare, SelectConfig, Selected};

/// The task kind every experience of this sample carries.
pub const TASK_KIND: &str = "repo-replay";

/// What the run is asked to do.
#[derive(Clone, Debug)]
pub struct RunConfig {
    /// The repository to replay (read, never written).
    pub source: PathBuf,
    /// Where the run keeps its mirror, scratch trees, ledger and store.
    pub state: PathBuf,
    /// Replay the commits after this revision.
    pub from: Option<String>,
    /// Replay the history up to this revision (default `HEAD`).
    pub to: Option<String>,
    /// Replay at most this many commits: the most recent replayable ones.
    pub max_commits: Option<usize>,
    /// The plan styles to attempt per commit.
    pub styles: Vec<PlanStyle>,
    /// Attempts run at once.
    pub jobs: usize,
    /// Selection bounds.
    pub select: SelectConfig,
    /// Gate bounds.
    pub gate: GateConfig,
    /// Plans written per attempt before giving up.
    pub plan_tries: usize,
    /// A diff line of this many words may not be repeated in a plan.
    pub min_line_tokens: usize,
    /// Wall-clock limit of one attempt.
    pub deadline: Duration,
    /// Wall-clock limit of one plan.
    pub plan_deadline: Duration,
    /// Output tokens one attempt may generate.
    pub max_output_tokens: Option<u64>,
    /// Longest a request may wait for the server's first byte and token.
    pub first_response: Option<Duration>,
    /// Project checks run in the attempt tree afterwards, never gated.
    pub checks: Vec<String>,
    /// Longest a project check may run.
    pub check_timeout: Duration,
    /// Passes that retry the attempts lost to the model or the disk, after the first.
    pub error_retries: usize,
    /// How long to wait before a retry pass.
    pub retry_wait: Duration,
}

/// The models of a run.
#[derive(Clone)]
pub struct Models {
    /// Writes plans.
    pub planner: Model,
    /// Replays commits.
    pub agent: Arc<dyn ModelProvider>,
    /// The agent's identity, recorded as the experiences' solver.
    pub identity: String,
}

/// What a run produced.
#[derive(Debug, Default)]
pub struct RunReport {
    /// Commits of the history that were replayable.
    pub replayable: usize,
    /// The ledger lines this run wrote, in order.
    pub entries: Vec<Entry>,
}

/// The state directory's parts.
pub struct Layout {
    /// The bare mirror of the source.
    pub mirror: PathBuf,
    /// Attempt trees live here, and are removed with their attempt.
    pub scratch: PathBuf,
}

/// Where a run keeps its parts under `state`.
#[must_use]
pub fn layout(state: &Path) -> Layout {
    Layout {
        mirror: state.join("mirror.git"),
        scratch: state.join("scratch"),
    }
}

/// Every string of `trajectory` scrubbed (messages, arguments, results).
pub fn scrub_trajectory(trajectory: &Trajectory, scrubber: &Scrubber) -> Result<Trajectory> {
    fn walk(v: &mut serde_json::Value, scrubber: &Scrubber) {
        match v {
            serde_json::Value::String(s) => *s = scrubber.scrub(s),
            serde_json::Value::Array(a) => a.iter_mut().for_each(|x| walk(x, scrubber)),
            serde_json::Value::Object(o) => o.values_mut().for_each(|x| walk(x, scrubber)),
            _ => {}
        }
    }
    let mut value = serde_json::to_value(trajectory)?;
    walk(&mut value, scrubber);
    Ok(serde_json::from_value(value)?)
}

/// The people of the chain: every author name and address.
fn people(chain: &[crate::git::CommitMeta]) -> Vec<String> {
    chain
        .iter()
        .flat_map(|c| [c.author_name.clone(), c.author_email.clone()])
        .collect()
}

fn producer() -> Producer {
    Producer {
        name: "repo-replay/gate".into(),
        version: "1".into(),
    }
}

/// What one unit of work (commit and style) came to.
enum Unit {
    Refused(PlanFailure),
    Attempted(Box<(String, AttemptOutcome)>),
    Error(String),
}

fn digest(text: &str) -> Digest {
    Digest::of(text.as_bytes())
}

/// Runs the replay described by `config`; see the module documentation.
pub async fn run(ctx: &Context, models: &Models, config: &RunConfig) -> Result<RunReport> {
    std::fs::create_dir_all(&config.state)?;
    // Absolute, so the paths the agent is told and the paths the gate compares are the same.
    let state = config
        .state
        .canonicalize()
        .context("resolving the state directory")?;
    let parts = layout(&state);
    let git = mirror(&config.source, &parts.mirror)?;
    ledger::note_run(
        &state,
        &crate::git::version()?,
        &config.source,
        config.from.as_deref(),
        config.to.as_deref(),
    )?;
    let chain = git.first_parent_chain(config.from.as_deref(), config.to.as_deref())?;
    let scrubber = Scrubber::new(people(&chain).iter().map(String::as_str));

    // Selection: a pure function of the history and the bounds.
    let mut replayable = Vec::new();
    let mut skips = Vec::new();
    for meta in &chain {
        match prepare(&git, meta, &scrubber, &config.select) {
            Ok(Selected::Replay(plan)) => replayable.push(*plan),
            Ok(Selected::Skipped(meta, why)) => skips.push((meta.sha, why)),
            // One commit git cannot be read for must not stop a run of a thousand.
            Err(e) => {
                eprintln!("skipping {}: {e:#}", meta.sha);
                skips.push((meta.sha.clone(), crate::select::Skip::Unreadable));
            }
        }
    }
    if let Some(max) = config.max_commits {
        let drop = replayable.len().saturating_sub(max);
        replayable.drain(..drop);
    }

    let done = ledger::read(&state)?;
    let mut report = RunReport {
        replayable: replayable.len(),
        ..RunReport::default()
    };
    for (sha, why) in skips {
        let seen = done
            .iter()
            .any(|e| e.commit == sha && e.outcome == Outcome::Skipped);
        if !seen {
            let entry = skip_entry(&sha, why.class());
            ledger::append(&state, &entry)?;
            report.entries.push(entry);
        }
    }

    let settings = AttemptSettings {
        model: Arc::clone(&models.agent),
        deadline: config.deadline,
        max_output_tokens: config.max_output_tokens,
        first_response: config.first_response,
        gate: config.gate,
        forbidden_locations: [&config.source, &parts.mirror]
            .into_iter()
            .flat_map(|p| [Some(p.clone()), p.canonicalize().ok()])
            .flatten()
            .map(|p| p.to_string_lossy().into_owned())
            .collect(),
        scratch: parts.scratch.clone(),
        checks: config.checks.clone(),
        check_timeout: config.check_timeout,
    };

    for pass in 0..=config.error_retries {
        let done = ledger::read(&state)?;
        let verdict_of = |c: &CommitPlan, s: PlanStyle| {
            done.iter()
                .rfind(|e| e.commit == c.sha && e.style == Some(s))
                .map(|e| e.outcome)
        };
        // The first pass does everything without a verdict; a retry pass
        // only what failed for want of infrastructure.
        let units: Vec<(&CommitPlan, PlanStyle)> = replayable
            .iter()
            .flat_map(|c| config.styles.iter().map(move |s| (c, *s)))
            .filter(|(c, s)| {
                let last = verdict_of(c, *s);
                let settled = done.iter().any(|e| {
                    e.commit == c.sha
                        && e.style == Some(*s)
                        && matches!(e.outcome, Outcome::Passed | Outcome::Failed)
                });
                !settled && (pass == 0 || last == Some(Outcome::Error))
            })
            .collect();
        if units.is_empty() {
            break;
        }
        let total = units.len();
        eprintln!("pass {pass}: {total} attempts");
        let errors = process(
            ctx,
            models,
            config,
            &git,
            &settings,
            &scrubber,
            &state,
            &units,
            &mut report,
        )
        .await?;
        if errors == 0 || pass == config.error_retries {
            break;
        }
        eprintln!(
            "{errors} attempts failed for want of the model or the disk; retrying them in {:?}",
            config.retry_wait
        );
        tokio::time::sleep(config.retry_wait).await;
    }

    Ok(report)
}

/// Runs `units` (at most `config.jobs` at once, results taken in order),
/// appending each outcome to the ledger and logging a progress line. Returns
/// how many ended as infrastructure errors.
#[allow(clippy::too_many_arguments)]
async fn process(
    ctx: &Context,
    models: &Models,
    config: &RunConfig,
    git: &Git,
    settings: &AttemptSettings,
    scrubber: &Scrubber,
    state: &Path,
    units: &[(&CommitPlan, PlanStyle)],
    report: &mut RunReport,
) -> Result<usize> {
    let mut results = futures::stream::iter(units.iter().map(|(commit, style)| {
        let (git, settings) = (git, settings);
        async move {
            (
                *commit,
                *style,
                work_unit(git, settings, models, config, commit, *style).await,
            )
        }
    }))
    .buffered(config.jobs.max(1));
    let mut errors = 0;
    let mut finished = 0;

    while let Some((commit, style, unit)) = results.next().await {
        let size = (
            commit.changes.iter().map(|c| c.lines()).sum::<usize>(),
            commit.changes.len(),
        );
        let entry = match unit {
            Unit::Error(detail) => Entry {
                commit: commit.sha.clone(),
                style: Some(style),
                outcome: Outcome::Error,
                classes: vec![],
                changed_lines: size.0,
                files: size.1,
                tool_calls: 0,
                seconds: 0.0,
                experience: None,
                checks: vec![],
                plan_words: None,
                detail: Some(detail),
            },
            Unit::Refused(failure) => Entry {
                commit: commit.sha.clone(),
                style: Some(style),
                outcome: Outcome::PlanRefused,
                classes: vec![failure.class().into()],
                changed_lines: size.0,
                files: size.1,
                tool_calls: 0,
                seconds: 0.0,
                experience: None,
                checks: vec![],
                plan_words: None,
                detail: Some(failure.to_string()),
            },
            Unit::Attempted(boxed) => {
                let (plan, outcome) = *boxed;
                match store_attempt(ctx, models, scrubber, commit, style, &plan, &outcome) {
                    Err(e) => Entry {
                        commit: commit.sha.clone(),
                        style: Some(style),
                        outcome: Outcome::Error,
                        classes: vec![],
                        changed_lines: size.0,
                        files: size.1,
                        tool_calls: 0,
                        seconds: 0.0,
                        experience: None,
                        checks: vec![],
                        plan_words: None,
                        detail: Some(format!("storing the attempt: {e:#}")),
                    },
                    Ok(id) => Entry {
                        commit: commit.sha.clone(),
                        style: Some(style),
                        outcome: if outcome.verdict.passed {
                            Outcome::Passed
                        } else {
                            Outcome::Failed
                        },
                        classes: outcome
                            .verdict
                            .classes()
                            .into_iter()
                            .map(String::from)
                            .collect(),
                        changed_lines: size.0,
                        files: size.1,
                        tool_calls: outcome.trace.calls.len(),
                        seconds: outcome.seconds,
                        experience: Some(id.hex().to_string()),
                        checks: outcome.checks.clone(),
                        plan_words: Some(plan.split_whitespace().count()),
                        detail: outcome.verdict.reasons.first().map(|r| r.detail.clone()),
                    },
                }
            }
        };
        ledger::append(state, &entry)?;
        errors += usize::from(entry.outcome == Outcome::Error);
        finished += 1;
        eprintln!(
            "[{finished}/{}] {} {} {:?} {:?} calls {} {:.0}s{}",
            units.len(),
            &entry.commit[..10],
            entry.style.map_or("-", PlanStyle::name),
            entry.outcome,
            entry.classes,
            entry.tool_calls,
            entry.seconds,
            entry
                .detail
                .as_deref()
                .filter(|_| entry.outcome == Outcome::Error)
                .map_or(String::new(), |d| format!(" ({d})")),
        );
        report.entries.push(entry);
    }
    Ok(errors)
}

fn skip_entry(sha: &str, class: &str) -> Entry {
    Entry {
        commit: sha.into(),
        style: None,
        outcome: Outcome::Skipped,
        classes: vec![class.into()],
        changed_lines: 0,
        files: 0,
        tool_calls: 0,
        seconds: 0.0,
        experience: None,
        checks: vec![],
        plan_words: None,
        detail: None,
    }
}

async fn work_unit(
    git: &Git,
    settings: &AttemptSettings,
    models: &Models,
    config: &RunConfig,
    commit: &CommitPlan,
    style: PlanStyle,
) -> Unit {
    let lines = changed_lines(&commit.diff);
    let check = PlanCheck {
        diff_lines: &lines,
        commit: &commit.sha,
        min_line_tokens: config.min_line_tokens,
    };
    let plan = match write_plan(
        &models.planner,
        style,
        &commit.message,
        &commit.diff,
        &check,
        config.plan_tries,
        config.plan_deadline,
    )
    .await
    {
        Ok(plan) => plan,
        Err(PlanFailure::Model(e)) => return Unit::Error(format!("planning: {e}")),
        Err(refused) => return Unit::Refused(refused),
    };
    match attempt(git, settings, commit, &plan).await {
        Ok(outcome) => Unit::Attempted(Box::new((plan, outcome))),
        Err(e) => Unit::Error(format!("{e:#}")),
    }
}

/// Stores the attempt as an experience of the plan, with the gate's verdict.
fn store_attempt(
    ctx: &Context,
    models: &Models,
    scrubber: &Scrubber,
    commit: &CommitPlan,
    style: PlanStyle,
    plan: &str,
    outcome: &AttemptOutcome,
) -> Result<ExperienceId> {
    let environment = Environment::new(
        Environment::WORKSPACE,
        json!({"snapshot": commit.parent, "commit": commit.sha, "plan_style": style.name()}),
    );
    let task = Task::new(
        TASK_KIND,
        Vec::new(),
        environment,
        scrubber.scrub(plan),
        Vec::new(),
    )?;
    let mut provenance = Provenance::new(&models.identity, ctx.runtime().clock());
    provenance.generator = Some("splinter-repo-replay".into());
    provenance.prompt_digests = vec![
        digest(&plan_system_prompt(style)),
        digest(&grounding_prompt("")),
    ];
    let trajectory = scrub_trajectory(&outcome.trajectory, scrubber)?;
    let final_output = outcome.final_message.as_deref().map(|m| scrubber.scrub(m));
    let experience = Experience::new(task, trajectory, final_output, provenance)?;
    let store = ctx.experiences();
    let id = store.put(&experience).context("storing the experience")?;
    let evidence = json!({
        "classes": outcome.verdict.classes(),
        "reasons": outcome.verdict.reasons,
        "residual": outcome.residual,
        "checks": outcome.checks,
        "plan_style": style.name(),
        "commit": commit.sha,
    });
    store.annotate(&Annotation {
        experience: id.clone(),
        producer: producer(),
        body: AnnotationBody::Verdict {
            outcome: if outcome.verdict.passed {
                Judged::Pass
            } else {
                Judged::Fail
            },
            strength: Strength::Executable,
            evidence,
        },
    })?;
    store.annotate(&Annotation {
        experience: id.clone(),
        producer: producer(),
        body: AnnotationBody::AttemptReport(AttemptReport {
            wall_seconds: Some(outcome.seconds),
            tool_calls: u32::try_from(outcome.trace.calls.len()).ok(),
            ..AttemptReport::default()
        }),
    })?;
    Ok(id)
}
