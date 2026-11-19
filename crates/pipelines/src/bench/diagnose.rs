// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements causal diagnosis of why an agent failed, from
// the evidence its run left, for its clients. If your team needs expertise in
// telling model failures from infrastructure failures before you train on
// them, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Why each failed trial of an examiner's export failed.
//!
//! Every trial of an export (`bench export`) is read with its digests
//! verified ([`super::artifact::read_trial`]) and every trial the export
//! skipped is read from `skipped.jsonl`, whose rows carry the score record
//! but no trajectory. A failed trial's evidence - the score status and cause,
//! the verifier's reason, whether a worktree change was exported, the
//! trajectory - goes to the deterministic diagnosis
//! ([`splinter_eval::diagnosis`]). A passed trial has no diagnosis. Only the
//! verifier's reason is read of what the verifier knows: the export carries
//! nothing of the task's hidden material, and a sealed export is refused as
//! the intake refuses it.
//!
//! [`DiagnosisReport`] counts the classes per domain (the score record's
//! collection) and per attribution, so infrastructure failures are seen apart
//! from the model's.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use splinter_core::diagnosis::DiagnosticResult;
use splinter_core::experience::ExperienceId;
use splinter_core::usage::TokenUsage;
use splinter_eval::diagnosis::{diagnose, AttemptEvidence};

use super::artifact::{read_index, read_trial, Artifact, UnitKind};
use super::score::ScoreRecord;

/// The format tag of a [`DiagnosisReport`].
pub const DIAGNOSIS_FORMAT: &str = "splinter.bench-diagnosis.v1";

/// The file of an export that lists the trials it did not export.
const SKIPPED: &str = "skipped.jsonl";

/// The record the examiner writes beside an export of sealed exam material.
const SEALED_EXPORT: &str = "sealed-export.json";

/// The worktree change an export carries when the agent changed files.
pub const WORKTREE_PATCH: &str = "worktree.patch";

/// One trial, with its diagnosis when it failed.
#[derive(Clone, Debug, Serialize)]
pub struct DiagnosedTrial {
    /// The export it was read from.
    pub export: PathBuf,
    /// The trial's name.
    pub trial: String,
    /// The task's id.
    pub task: String,
    /// The domain: the score record's collection, else the task id's.
    pub domain: String,
    /// The examiner's status, when a score record says.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    /// Whether the trial passed.
    pub passed: bool,
    /// Whether the trial was exported (a skipped one has no trajectory).
    pub exported: bool,
    /// The experiences it became, when a practice run admitted it.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub experiences: Vec<ExperienceId>,
    /// The staged task folder it attempted, when a practice run's report
    /// names it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub staged_task: Option<String>,
    /// Why it failed; `None` for a passed trial.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diagnosis: Option<DiagnosticResult>,
    /// The capability ids the export names for its task; `None` when the
    /// export names none (made before the examiner labelled tasks).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub capabilities: Option<Vec<String>>,
    /// The task's content digest, for an exported trial.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_digest: Option<String>,
    /// What the verifier measured, for an exported trial.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metrics: Option<serde_json::Value>,
    /// What its model requests processed, when the export counts it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<TokenUsage>,
}

/// A trial or an export that could not be read, and why.
#[derive(Clone, Debug, Serialize)]
pub struct DiagnosisRefused {
    /// The export.
    pub export: PathBuf,
    /// The trial, when one is named.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trial: Option<String>,
    /// Why.
    pub reason: String,
}

/// What a diagnosis of exports found.
#[derive(Clone, Debug, Serialize)]
pub struct DiagnosisReport {
    /// Always [`DIAGNOSIS_FORMAT`].
    pub format: &'static str,
    /// The rules' version ([`splinter_eval::diagnosis::RULES_VERSION`]).
    pub rules: &'static str,
    /// Trials read.
    pub trials: usize,
    /// Of those, passed.
    pub passed: usize,
    /// Failed trials by domain, then class.
    pub by_domain: BTreeMap<String, BTreeMap<String, usize>>,
    /// Failed trials by attribution (`model`, `infrastructure`, `task`,
    /// `unattributed`).
    pub by_attribution: BTreeMap<String, usize>,
    /// Every trial read.
    pub attempts: Vec<DiagnosedTrial>,
    /// What could not be read.
    pub refused: Vec<DiagnosisRefused>,
}

/// A row of `skipped.jsonl`.
#[derive(Deserialize)]
struct SkippedRow {
    trial: String,
    task: String,
    #[serde(default)]
    score: Option<ScoreRecord>,
    #[serde(default)]
    capabilities: Option<Vec<String>>,
}

/// The evidence of an exported trial.
fn exported_evidence(artifact: &Artifact) -> (Option<DiagnosticResult>, bool) {
    let score = artifact.manifest.score.as_ref();
    let passed = artifact.manifest.grade.verdict == "pass";
    if passed {
        return (None, true);
    }
    let trajectory = match artifact.units.as_slice() {
        [unit] if matches!(unit.kind, UnitKind::Single(_)) => Some(&unit.trajectory),
        _ => None,
    };
    let evidence = AttemptEvidence {
        status: score.map(|s| s.status.as_str()),
        cause_code: score
            .and_then(|s| s.cause.as_ref())
            .and_then(|c| c.get("code"))
            .and_then(serde_json::Value::as_str),
        cause_message: score.and_then(ScoreRecord::cause_message),
        verification_reason: artifact.verifier_reason.as_deref(),
        workspace_changed: Some(artifact.manifest.files.contains_key(WORKTREE_PATCH)),
        gateway_statuses: &[],
        trajectory,
    };
    (Some(diagnose(&evidence)), false)
}

/// What an exported trial's requests processed: its score record's usage,
/// else its manifest's token counts.
fn trial_usage(artifact: &Artifact) -> Option<TokenUsage> {
    artifact
        .manifest
        .score
        .as_ref()
        .and_then(|s| s.usage.as_ref())
        .and_then(TokenUsage::from_score_usage)
        .or_else(|| {
            artifact
                .manifest
                .tokens
                .as_ref()
                .and_then(TokenUsage::from_trial_tokens)
        })
}

/// The domain of a trial: the score record's collection, else the part of
/// the task's name before its first dash.
fn domain_of(task: &str, score: Option<&ScoreRecord>) -> String {
    score
        .and_then(|s| s.identity.as_ref())
        .and_then(|i| i.collection.clone())
        .unwrap_or_else(|| {
            let name = task.rsplit('/').next().unwrap_or(task);
            name.split('-').next().unwrap_or(name).to_string()
        })
}

/// Diagnoses every trial of `exports`.
#[must_use]
pub fn diagnose_exports(exports: &[PathBuf]) -> DiagnosisReport {
    let mut attempts = Vec::new();
    let mut refused = Vec::new();
    for export in exports {
        diagnose_export(export, &mut attempts, &mut refused);
    }
    report(attempts, refused)
}

/// Diagnoses the trials under `dirs`: each an export, or a practice run's
/// directory (its `export/`, with the experiences its `report.json` says
/// each trial became).
#[must_use]
pub fn diagnose_dirs(dirs: &[PathBuf]) -> DiagnosisReport {
    let mut attempts = Vec::new();
    let mut refused = Vec::new();
    for dir in dirs {
        let run = dir.join("report.json");
        if !run.is_file() {
            diagnose_export(dir, &mut attempts, &mut refused);
            continue;
        }
        let export = dir.join("export");
        let from = attempts.len();
        diagnose_export(&export, &mut attempts, &mut refused);
        match practice_experiences(&run) {
            Ok(experiences) => {
                for attempt in &mut attempts[from..] {
                    if let Some((task, ids)) = experiences.get(&attempt.trial) {
                        attempt.staged_task = Some(task.clone());
                        attempt.experiences = ids.clone();
                    }
                }
            }
            Err(reason) => refused.push(DiagnosisRefused {
                export,
                trial: None,
                reason,
            }),
        }
    }
    report(attempts, refused)
}

/// The staged task folder and the experiences of each trial of a practice
/// run's report.
type PracticeTrials = BTreeMap<String, (String, Vec<ExperienceId>)>;

/// The trials of a practice run's report.
fn practice_experiences(report: &Path) -> Result<PracticeTrials, String> {
    #[derive(Deserialize)]
    struct Trial {
        trial: String,
        task: String,
        #[serde(default)]
        experiences: Vec<ExperienceId>,
    }
    #[derive(Deserialize)]
    struct Report {
        trials: Vec<Trial>,
    }
    let text = std::fs::read_to_string(report).map_err(|e| format!("{}: {e}", report.display()))?;
    let report: Report =
        serde_json::from_str(&text).map_err(|e| format!("{}: {e}", report.display()))?;
    Ok(report
        .trials
        .into_iter()
        .map(|t| (t.trial, (t.task, t.experiences)))
        .collect())
}

/// The report over `attempts`.
#[must_use]
pub fn report(attempts: Vec<DiagnosedTrial>, refused: Vec<DiagnosisRefused>) -> DiagnosisReport {
    let mut by_domain: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();
    let mut by_attribution: BTreeMap<String, usize> = BTreeMap::new();
    for diagnosis in attempts
        .iter()
        .filter_map(|a| a.diagnosis.as_ref().map(|d| (a, d)))
    {
        let (attempt, result) = diagnosis;
        *by_domain
            .entry(attempt.domain.clone())
            .or_default()
            .entry(result.class.as_str().to_string())
            .or_default() += 1;
        let attribution = result.attribution.map_or_else(
            || "unattributed".to_string(),
            |a| {
                serde_json::to_value(a)
                    .ok()
                    .and_then(|v| v.as_str().map(str::to_string))
                    .unwrap_or_default()
            },
        );
        *by_attribution.entry(attribution).or_default() += 1;
    }
    DiagnosisReport {
        format: DIAGNOSIS_FORMAT,
        rules: splinter_eval::diagnosis::RULES_VERSION,
        trials: attempts.len(),
        passed: attempts.iter().filter(|a| a.passed).count(),
        by_domain,
        by_attribution,
        attempts,
        refused,
    }
}

fn diagnose_export(
    export: &Path,
    attempts: &mut Vec<DiagnosedTrial>,
    refused: &mut Vec<DiagnosisRefused>,
) {
    let refuse = |trial: Option<String>, reason: String| DiagnosisRefused {
        export: export.to_path_buf(),
        trial,
        reason,
    };
    if export.join(SEALED_EXPORT).exists() {
        refused.push(refuse(
            None,
            format!("{SEALED_EXPORT} is beside it: exam material is never diagnosed for learning"),
        ));
        return;
    }
    match read_index(export) {
        Err(reason) => refused.push(refuse(None, reason)),
        Ok(rows) => {
            for row in rows {
                let row = match row {
                    Ok(row) => row,
                    Err(reason) => {
                        refused.push(refuse(None, reason));
                        continue;
                    }
                };
                match read_trial(export, &row) {
                    Err(reason) => refused.push(refuse(Some(row.trial.clone()), reason)),
                    Ok(artifact) => {
                        let (diagnosis, passed) = exported_evidence(&artifact);
                        let score = artifact.manifest.score.as_ref();
                        attempts.push(DiagnosedTrial {
                            export: export.to_path_buf(),
                            trial: row.trial.clone(),
                            task: row.task.clone(),
                            domain: domain_of(&row.task, score),
                            status: score.map(|s| s.status.clone()),
                            passed,
                            exported: true,
                            experiences: Vec::new(),
                            staged_task: None,
                            diagnosis,
                            capabilities: row
                                .capabilities
                                .clone()
                                .or_else(|| artifact.manifest.task.capabilities.clone()),
                            task_digest: Some(artifact.manifest.task.digest.clone()),
                            metrics: artifact.manifest.grade.metrics.clone(),
                            usage: trial_usage(&artifact),
                        });
                    }
                }
            }
        }
    }
    let Ok(text) = std::fs::read_to_string(export.join(SKIPPED)) else {
        return;
    };
    for (number, line) in text
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
    {
        let row: SkippedRow = match serde_json::from_str(line) {
            Ok(row) => row,
            Err(e) => {
                refused.push(refuse(None, format!("{SKIPPED} line {}: {e}", number + 1)));
                continue;
            }
        };
        let score = row.score.as_ref();
        let passed = score.is_some_and(|s| s.pass == Some(true));
        let diagnosis = (!passed).then(|| {
            diagnose(&AttemptEvidence {
                status: score.map(|s| s.status.as_str()),
                cause_code: score
                    .and_then(|s| s.cause.as_ref())
                    .and_then(|c| c.get("code"))
                    .and_then(serde_json::Value::as_str),
                cause_message: score.and_then(ScoreRecord::cause_message),
                ..AttemptEvidence::default()
            })
        });
        attempts.push(DiagnosedTrial {
            export: export.to_path_buf(),
            domain: domain_of(&row.task, score),
            status: score.map(|s| s.status.clone()),
            trial: row.trial,
            task: row.task,
            passed,
            exported: false,
            experiences: Vec::new(),
            staged_task: None,
            diagnosis,
            capabilities: row.capabilities,
            task_digest: None,
            metrics: None,
            usage: score
                .and_then(|s| s.usage.as_ref())
                .and_then(TokenUsage::from_score_usage),
        });
    }
}
