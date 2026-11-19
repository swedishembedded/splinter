// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for measuring what an agent can
// and cannot do, capability by capability, from graded practice, for its
// clients. If your team needs expertise in competence tracking for learning
// agents, you can procure our services by sending an email to
// info@swedishembedded.com.

//! A policy's competence per capability, from the examiner's exports.
//!
//! Every trial of the given exports or practice runs is read and diagnosed
//! as [`super::diagnose`] reads it. Its task's capability ids come, in this
//! order, from:
//!
//! 1. the export itself (its index row, its manifest, or its line of
//!    `skipped.jsonl`): the examiner labelled the task when it exported it;
//! 2. the practice run's own material (the `practice.json` its report names,
//!    held to the sha256 the report recorded), by the staged task's name;
//! 3. a practice directory given as labels, by the staged task's
//!    `source_digest`: the same task, byte for byte, as the examiner staged
//!    it again with a taxonomy that names its capabilities. A task matched
//!    by name only is never labelled.
//!
//! Beside competence, each policy's graded attempts are measured for what
//! they can teach a group-relative policy gradient per token
//! ([`PolicySignal`]): the learning signal per task and per family (the
//! lineage the staged task's `task.toml` names), the informative-group share
//! and the reward variance per million completion tokens, uniform and under
//! the variance-per-cost draw, for the strict reward and for `shaped-partial`.
//!
//! A task labelled by none of them counts as
//! [`crate::curriculum::capability::UNLABELLED`]. A passed trial is a graded
//! pass; a failed one whose diagnosis blames the infrastructure or the task
//! is ungraded (never a `0`); every other failure is a graded fail with its
//! class. Competence and weights are
//! [`crate::curriculum::capability`]'s, per policy: a practice run's report
//! names its policy, and an export without one counts under the policy the
//! caller names (`unknown` when none). A sealed export is refused whole, as
//! the diagnosis refuses it.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use splinter_core::digest::Digest;
use splinter_core::outcome::Attribution;
use splinter_eval::campaign::pass_rate::PassRates;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

use super::artifact::sha256_hex;
use super::diagnose::{diagnose_dirs, DiagnosedTrial, DiagnosisRefused};
use super::practice::{Practice, PracticeFacts};
use crate::curriculum::capability::{competence, Attempt, CapabilityCompetence};
use crate::rl_build::{DEFAULT_FAIL_CEILING, PARTIAL_METRIC};
use splinter_core::usage::TokenUsage;
use splinter_data::PartialCredit;
use splinter_eval::campaign::learning_signal::{
    Attempt as SignalAttempt, SignalReport, TaskSignal,
};

/// The format tag of a [`CapabilityReport`].
pub const CAPABILITY_FORMAT: &str = "splinter.capability-competence.v1";

/// The store's document class a [`CapabilityReport`] is kept under.
pub const CAPABILITY_DOCUMENT: &str = "capability_competence";

/// The policy name trials of an export that names none count under, when
/// the caller names none either.
pub const UNKNOWN_POLICY: &str = "unknown";

/// Where the trials' capability ids came from, counted.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Labelled {
    /// From the export.
    pub export: usize,
    /// From the practice run's own material.
    pub material: usize,
    /// From a labels directory, by source digest.
    pub labels: usize,
    /// From nowhere: counted as unlabelled.
    pub none: usize,
}

/// One directory read, and what identifies its content.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Source {
    /// The export or practice run directory.
    pub dir: PathBuf,
    /// The sha256 of its export's `index.jsonl`, when it has one.
    pub index_sha256: Option<String>,
    /// The policy its trials count under.
    pub policy: String,
    /// The practice material it ran, by `practice.json`'s sha256, when a
    /// practice run's report names it.
    pub practice_sha256: Option<String>,
}

/// One policy's competence.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PolicyCompetence {
    /// The policy: `<model>+<adapter digest>` as a practice run names it,
    /// or the caller's name.
    pub policy: String,
    /// Trials counted.
    pub trials: usize,
    /// Every capability its trials exercise, by id.
    pub capabilities: Vec<CapabilityCompetence>,
    /// What its graded attempts can teach per token, per task and per
    /// family, under the strict reward and under `shaped-partial`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signal: Option<PolicySignal>,
}

/// The learning signal of a policy's graded attempts
/// ([`splinter_eval::campaign::learning_signal`]), each task's attempts one
/// group, its family the lineage its staged `task.toml` names (else the task).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PolicySignal {
    /// Under the strict reward: a pass earns one, a failure nothing.
    pub verdict: SignalReport,
    /// Under `shaped-partial` at the default credit
    /// ([`crate::datasets::DEFAULT_FAIL_CEILING`]), the floors from the
    /// material.
    pub shaped_partial: SignalReport,
    /// Graded failures that recorded no `functional_correctness`, left out
    /// of `shaped_partial` (an unmeasured failure is not a zero).
    pub unmeasured_failures: usize,
    /// Tasks whose naive floor the material states.
    pub floored_tasks: usize,
}

/// What a capability report found: per policy, competence per capability,
/// with where every number came from.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CapabilityReport {
    /// Always [`CAPABILITY_FORMAT`].
    pub format: String,
    /// The diagnosis rules' version the failure classes are by.
    pub rules: String,
    /// The directories read.
    pub sources: Vec<Source>,
    /// The labels directories given, by `practice.json`'s sha256.
    pub labels: BTreeMap<PathBuf, String>,
    /// Where the trials' capability ids came from.
    pub labelled: Labelled,
    /// Per policy.
    pub policies: Vec<PolicyCompetence>,
    /// What could not be read.
    pub refused: Vec<Refused>,
}

/// A directory or trial that could not be read, and why.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Refused {
    /// The directory.
    pub dir: PathBuf,
    /// The trial, when one is named.
    pub trial: Option<String>,
    /// Why.
    pub reason: String,
}

impl From<DiagnosisRefused> for Refused {
    fn from(r: DiagnosisRefused) -> Self {
        Self {
            dir: r.export,
            trial: r.trial,
            reason: r.reason,
        }
    }
}

/// A capability report as the store keeps it, with its address: the
/// competence of each policy, recorded once under the digest of everything
/// it counted and where it came from.
#[derive(Clone, Debug, Serialize)]
pub struct StoredCapabilities {
    /// The report's address in the store ([`CAPABILITY_DOCUMENT`]).
    pub document: Digest,
    /// The report.
    pub report: CapabilityReport,
}

impl StoredCapabilities {
    /// Keeps `report` in `ctx`'s store; keeping the same report again
    /// writes nothing.
    ///
    /// # Errors
    /// The store's.
    pub fn keep(ctx: &Context, report: CapabilityReport) -> Result<Self, OrchestratorError> {
        let document = ctx.workspace().put_document(CAPABILITY_DOCUMENT, &report)?;
        Ok(Self { document, report })
    }
}

/// The practice run report fields this reader uses.
#[derive(Deserialize)]
struct RunReport {
    policy: RunPolicy,
    provenance: RunProvenance,
}

#[derive(Deserialize)]
struct RunPolicy {
    model: String,
    #[serde(default)]
    adapter: Option<String>,
}

#[derive(Deserialize)]
struct RunProvenance {
    dir: PathBuf,
    practice_sha256: String,
}

/// A practice run's policy and material, from its `report.json`.
struct Run {
    policy: String,
    material: Result<Practice, String>,
    practice_sha256: String,
}

fn read_run(dir: &Path) -> Result<Option<Run>, String> {
    let path = dir.join("report.json");
    if !path.is_file() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let report: RunReport =
        serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    let policy = match &report.policy.adapter {
        Some(adapter) => format!("{}+{adapter}", report.policy.model),
        None => report.policy.model.clone(),
    };
    let expected = report.provenance.practice_sha256;
    let material = Practice::read(&report.provenance.dir)
        .map_err(|e| e.to_string())
        .and_then(|practice| {
            if practice.sha256 == expected {
                Ok(practice)
            } else {
                Err(format!(
                    "{}: practice.json is {}, not the {expected} the run recorded",
                    report.provenance.dir.display(),
                    practice.sha256
                ))
            }
        });
    Ok(Some(Run {
        policy,
        material,
        practice_sha256: expected,
    }))
}

/// The capability ids of every task a labels directory stages, by source
/// digest; the directories' task facts are added to `facts`.
fn read_labels(
    dirs: &[PathBuf],
    refused: &mut Vec<Refused>,
    facts: &mut PracticeFacts,
) -> (BTreeMap<String, Vec<String>>, BTreeMap<PathBuf, String>) {
    let mut by_digest = BTreeMap::new();
    let mut read = BTreeMap::new();
    for dir in dirs {
        match Practice::read(dir) {
            Ok(practice) => {
                facts.tasks.extend(practice.task_facts());
                for task in &practice.manifest.tasks {
                    if !task.capabilities.is_empty() {
                        by_digest.insert(task.source_digest.clone(), task.capabilities.clone());
                    }
                }
                read.insert(dir.clone(), practice.sha256);
            }
            Err(e) => refused.push(Refused {
                dir: dir.clone(),
                trial: None,
                reason: e.to_string(),
            }),
        }
    }
    (by_digest, read)
}

/// Whether a trial counts as graded, and how: `Some(passed)`, or `None` when
/// its diagnosis blames the infrastructure or the task.
fn graded(trial: &DiagnosedTrial) -> Option<bool> {
    if trial.passed {
        return Some(true);
    }
    match trial.diagnosis.as_ref().and_then(|d| d.attribution) {
        Some(Attribution::Infrastructure | Attribution::Task) => None,
        _ => Some(false),
    }
}

/// What a capability report reads, and how it attributes it.
#[derive(Clone, Copy, Debug)]
pub struct CapabilityRequest<'a> {
    /// Export directories or practice run directories.
    pub dirs: &'a [PathBuf],
    /// Practice directories whose `practice.json` labels tasks, matched by
    /// source digest.
    pub labels: &'a [PathBuf],
    /// The policy trials of a directory that names none count under.
    pub policy: Option<&'a str>,
}

/// The competence per capability of every policy whose trials `request`
/// names.
#[must_use]
pub fn capability_report(request: &CapabilityRequest<'_>) -> CapabilityReport {
    let mut refused = Vec::new();
    let mut facts = PracticeFacts::default();
    let (by_digest, labels) = read_labels(request.labels, &mut refused, &mut facts);
    let mut signals: BTreeMap<String, Vec<SignalTrial>> = BTreeMap::new();
    let mut sources = Vec::new();
    let mut labelled = Labelled::default();
    let mut by_policy: BTreeMap<String, Vec<Attempt>> = BTreeMap::new();
    let mut rules = splinter_eval::diagnosis::RULES_VERSION.to_string();
    for dir in request.dirs {
        let run = match read_run(dir) {
            Ok(run) => run,
            Err(reason) => {
                refused.push(Refused {
                    dir: dir.clone(),
                    trial: None,
                    reason,
                });
                continue;
            }
        };
        let diagnosed = diagnose_dirs(std::slice::from_ref(dir));
        rules = diagnosed.rules.to_string();
        refused.extend(diagnosed.refused.into_iter().map(Refused::from));
        let export = if run.is_some() {
            dir.join("export")
        } else {
            dir.clone()
        };
        let policy = run.as_ref().map_or_else(
            || request.policy.unwrap_or(UNKNOWN_POLICY).to_string(),
            |r| r.policy.clone(),
        );
        let material = run.as_ref().and_then(|r| match &r.material {
            Ok(practice) => Some(practice),
            Err(reason) => {
                refused.push(Refused {
                    dir: dir.clone(),
                    trial: None,
                    reason: format!("its material is not labelled from: {reason}"),
                });
                None
            }
        });
        let staged: BTreeMap<&str, (&str, &[String])> = material
            .map(|p| {
                p.manifest
                    .tasks
                    .iter()
                    .map(|t| {
                        (
                            t.name.as_str(),
                            (t.source_digest.as_str(), t.capabilities.as_slice()),
                        )
                    })
                    .collect()
            })
            .unwrap_or_default();
        sources.push(Source {
            dir: dir.clone(),
            index_sha256: std::fs::read(export.join("index.jsonl"))
                .ok()
                .map(|bytes| sha256_hex(&bytes)),
            policy: policy.clone(),
            practice_sha256: run.as_ref().map(|r| r.practice_sha256.clone()),
        });
        if let Some(practice) = material {
            facts.tasks.extend(practice.task_facts());
        }
        let measured = signals.entry(policy.clone()).or_default();
        let attempts = by_policy.entry(policy).or_default();
        for trial in &diagnosed.attempts {
            let task = trial
                .staged_task
                .clone()
                .unwrap_or_else(|| trial.task.clone());
            let own = staged.get(task.as_str());
            let capabilities = if let Some(ids) = trial.capabilities.clone() {
                labelled.export += 1;
                ids
            } else if let Some((_, ids)) = own.filter(|(_, ids)| !ids.is_empty()) {
                labelled.material += 1;
                ids.to_vec()
            } else if let Some(ids) = own.and_then(|(digest, _)| by_digest.get(*digest)) {
                labelled.labels += 1;
                ids.clone()
            } else {
                labelled.none += 1;
                Vec::new()
            };
            if let Some(passed) = graded(trial) {
                measured.push(SignalTrial {
                    task: task.clone(),
                    digest: trial
                        .task_digest
                        .clone()
                        .or_else(|| own.map(|(digest, _)| (*digest).to_string())),
                    passed,
                    correctness: trial
                        .metrics
                        .as_ref()
                        .and_then(|m| m.get(PARTIAL_METRIC))
                        .and_then(serde_json::Value::as_f64),
                    usage: trial.usage,
                });
            }
            attempts.push(Attempt {
                task,
                capabilities,
                passed: graded(trial),
                class: trial.diagnosis.as_ref().map(|d| d.class),
            });
        }
    }
    let policies = by_policy
        .into_iter()
        .map(|(policy, attempts)| PolicyCompetence {
            trials: attempts.len(),
            capabilities: competence(&attempts, &rates_of(&attempts)),
            signal: signals
                .get(&policy)
                .filter(|trials| !trials.is_empty())
                .map(|trials| policy_signal(trials, &facts)),
            policy,
        })
        .collect();
    CapabilityReport {
        format: CAPABILITY_FORMAT.to_string(),
        rules,
        sources,
        labels,
        labelled,
        policies,
        refused,
    }
}

/// A graded attempt, as the learning signal reads it.
struct SignalTrial {
    task: String,
    digest: Option<String>,
    passed: bool,
    correctness: Option<f64>,
    usage: Option<TokenUsage>,
}

/// The learning signal of a policy's graded `trials`, by task, under both
/// rewards; `facts` gives each task's floor and lineage by content digest.
fn policy_signal(trials: &[SignalTrial], facts: &PracticeFacts) -> PolicySignal {
    let credit = PartialCredit {
        metric: PARTIAL_METRIC.to_string(),
        credit: DEFAULT_FAIL_CEILING,
        floors: facts.naive_floors(),
        sources: Vec::new(),
        groups_without_floor: 0,
    };
    let mut by_task: BTreeMap<&str, Vec<&SignalTrial>> = BTreeMap::new();
    for trial in trials {
        by_task.entry(&trial.task).or_default().push(trial);
    }
    let mut unmeasured_failures = 0;
    let (mut verdict, mut shaped) = (Vec::new(), Vec::new());
    for (task, attempts) in by_task {
        let digest = attempts.iter().find_map(|t| t.digest.as_deref());
        let family = facts.lineage(digest).unwrap_or(task);
        let attempt = |t: &SignalTrial, reward: f64| SignalAttempt {
            reward,
            cost: t.usage.map(|u| u.cost()),
            completion: t.usage.map(|u| u.completion),
        };
        let strict: Vec<SignalAttempt> = attempts
            .iter()
            .map(|t| attempt(t, if t.passed { 1.0 } else { 0.0 }))
            .collect();
        let floor = credit.floor(digest);
        let mut partial = Vec::new();
        for t in &attempts {
            match (t.passed, t.correctness) {
                (true, _) => partial.push(attempt(t, 1.0)),
                (false, Some(f)) => partial.push(attempt(t, credit.earned(f, floor))),
                (false, None) => unmeasured_failures += 1,
            }
        }
        verdict.push(TaskSignal::of(task, family, &strict));
        shaped.push(TaskSignal::of(task, family, &partial));
    }
    PolicySignal {
        verdict: SignalReport::of(verdict),
        shaped_partial: SignalReport::of(shaped),
        unmeasured_failures,
        floored_tasks: credit.floors.len(),
    }
}

/// Each task's pass rate over `attempts`, by the campaign's own bands
/// (retired at [`splinter_eval::campaign::pass_rate::RETIRE_AT`] over
/// [`splinter_eval::campaign::pass_rate::CALIBRATION_K`] graded attempts).
fn rates_of(attempts: &[Attempt]) -> PassRates {
    let mut rates = PassRates::default();
    let mut by_task: BTreeMap<&str, Vec<Option<bool>>> = BTreeMap::new();
    for attempt in attempts {
        by_task
            .entry(&attempt.task)
            .or_default()
            .push(attempt.passed);
    }
    for (task, outcomes) in by_task {
        rates.observe(task, 0, &outcomes);
    }
    rates
}
