// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that turn an examiner's practice
// material into graded agent experience for its clients. If your team needs
// expertise in on-policy experience collection for agent training you can
// procure our services by sending an email to info@swedishembedded.com.

//! The campaign's rollouts over the examiner's staged practice material:
//! each generation's champion runs the material exactly as
//! `tasks harbor practice` runs it ([`crate::bench::practice::run_practice`]),
//! and the graded export is linked into the rollout stage's directory for
//! the ingest stage to admit. Every trial carries its task's capability ids
//! (the examiner's labels) and, when it failed, its diagnosed class
//! ([`crate::bench::diagnose`]), for the capability curriculum.
//!
//! A rollout asked for transfer exercises first has the examiner draw them
//! from the collection's generator (`bench task practice --count --seed
//! --capability`, into the stage's `transfer/`, kept there across a resume),
//! practises them the same way, and adds their trials to the same export.

use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use splinter_agent::CancelToken;
use splinter_orchestrator::context::Context;

use super::types::CampaignError;
use super::work::{Rolled, RolloutRequest, Rollouts, TransferRequest, Trial};
use crate::bench::diagnose::diagnose_dirs;
use crate::bench::practice::{
    run_practice, BenchCheckout, Practice, PracticeDraw, PracticeReport, PracticeRequest,
    PracticeSettings, TaskFacts,
};

/// Where and how the champion practises, as a campaign file's `[practice]`
/// names it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PracticeSpec {
    /// The staged practice directory (`bench task practice`, holding
    /// practice.json): the train side only.
    pub dir: PathBuf,
    /// The examiner's checkout.
    pub bench: PathBuf,
    /// The `harbor` program; the checkout's own when absent.
    #[serde(default)]
    pub harbor: Option<PathBuf>,
    /// The sven build the examiner runs the agent from.
    #[serde(default)]
    pub sven_build: Option<PathBuf>,
    /// The profile the policy's is derived from; the staging's when absent.
    #[serde(default)]
    pub profile: Option<PathBuf>,
    /// Trials run at once.
    #[serde(default = "one")]
    pub jobs: usize,
    /// The rung of the generator's difficulty ladder the transfer exercises
    /// are drawn at (`bench task practice --rung`); the generator's own
    /// when absent.
    #[serde(default)]
    pub rung: Option<i64>,
    /// The families the transfer exercises are drawn from (`--family`); every
    /// family that responds to the rung when empty.
    #[serde(default)]
    pub families: Vec<String>,
}

fn one() -> usize {
    1
}

/// The champion practising the staged material.
pub struct PracticeRollouts {
    practice: Practice,
    checkout: BenchCheckout,
    profile: Option<PathBuf>,
    profile_sha256: Option<String>,
    jobs: usize,
    transfer_rung: Option<i64>,
    transfer_families: Vec<String>,
}

impl PracticeRollouts {
    /// The rollouts `spec` names.
    ///
    /// # Errors
    /// The material cannot be read.
    pub fn new(spec: &PracticeSpec) -> Result<Self, CampaignError> {
        let practice = Practice::read(&spec.dir)
            .map_err(|e| CampaignError::Config(format!("{}: {e}", spec.dir.display())))?;
        let mut checkout = BenchCheckout::new(spec.bench.clone());
        if let Some(harbor) = &spec.harbor {
            checkout.harbor.clone_from(harbor);
        }
        checkout.sven_build.clone_from(&spec.sven_build);
        let profile_sha256 = spec
            .profile
            .as_ref()
            .map(|p| {
                std::fs::read(p)
                    .map(|bytes| crate::bench::artifact::sha256_hex(&bytes))
                    .map_err(|e| CampaignError::Config(format!("{}: {e}", p.display())))
            })
            .transpose()?;
        Ok(Self {
            practice,
            checkout,
            profile: spec.profile.clone(),
            profile_sha256,
            jobs: spec.jobs.max(1),
            transfer_rung: spec.rung,
            transfer_families: spec.families.clone(),
        })
    }
}

/// The export's listings, which a second export into the same directory
/// appends to instead of replacing.
const LISTINGS: [&str; 2] = ["index.jsonl", "skipped.jsonl"];

/// `from`'s export linked into `to` beside what `to` holds: its trial
/// directories linked, its listings appended.
fn merge_export(from: &Path, to: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(to)?;
    for entry in std::fs::read_dir(from)? {
        let entry = entry?;
        let name = entry.file_name();
        if LISTINGS.iter().any(|l| name == *l) {
            let lines = std::fs::read(entry.path())?;
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(to.join(&name))?
                .write_all(&lines)?;
        } else if entry.file_type()?.is_dir() {
            link_tree(&entry.path(), &to.join(&name))?;
        }
    }
    Ok(())
}

/// `from`'s files linked (copied where a link cannot be made) under `to`.
fn link_tree(from: &Path, to: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(to)?;
    for entry in std::fs::read_dir(from)? {
        let entry = entry?;
        let (source, target) = (entry.path(), to.join(entry.file_name()));
        if entry.file_type()?.is_dir() {
            link_tree(&source, &target)?;
        } else {
            let _ = std::fs::remove_file(&target);
            if std::fs::hard_link(&source, &target).is_err() {
                std::fs::copy(&source, &target)?;
            }
        }
    }
    Ok(())
}

impl Rollouts for PracticeRollouts {
    fn identity(&self) -> Value {
        let manifest = &self.practice.manifest;
        json!({
            "practice_sha256": self.practice.sha256,
            "collection": manifest.collection,
            "manifest_sha256": manifest.manifest_sha256,
            "content_sha256": manifest.content_sha256,
            "set": manifest.set,
            "profile_sha256": self.profile_sha256,
        })
    }

    fn tasks(&self) -> Result<Vec<String>, CampaignError> {
        Ok(self
            .practice
            .manifest
            .tasks
            .iter()
            .map(|t| t.name.clone())
            .collect())
    }

    fn capabilities(&self) -> Result<BTreeMap<String, Vec<String>>, CampaignError> {
        Ok(self.practice.capabilities())
    }

    fn facts(&self) -> Result<BTreeMap<String, TaskFacts>, CampaignError> {
        Ok(self
            .practice
            .task_facts()
            .into_values()
            .map(|facts| (facts.name.clone(), facts))
            .collect())
    }

    fn material(&self) -> Option<&Path> {
        Some(&self.practice.dir)
    }

    fn rollout(
        &mut self,
        ctx: &Context,
        request: &RolloutRequest<'_>,
        cancel: &CancelToken,
    ) -> Result<Rolled, CampaignError> {
        let export = request.dir.join("export");
        // A rollout starts over: what an interrupted attempt linked is replaced.
        let _ = std::fs::remove_dir_all(&export);
        let mut runs =
            vec![self.practise(ctx, request, &self.practice, Some(request.tasks), cancel)?];
        if let Some(transfer) = &request.transfer {
            let drawn = self.draw(transfer, request.dir)?;
            runs.push(self.practise(ctx, request, &drawn, None, cancel)?);
        }
        let mut trials = Vec::new();
        let mut counted = Vec::new();
        for (report, labels) in &runs {
            merge_export(&report.dir.join("export"), &export).map_err(|e| {
                CampaignError::Collector(format!(
                    "linking the export into {}: {e}",
                    export.display()
                ))
            })?;
            let classes: BTreeMap<String, _> = diagnose_dirs(std::slice::from_ref(&report.dir))
                .attempts
                .into_iter()
                .filter_map(|a| a.diagnosis.map(|d| (a.trial, d.class)))
                .collect();
            trials.extend(report.trials.iter().map(|t| Trial {
                task: t.task.clone(),
                reward: t.reward,
                passed: t.reward.map(|r| r >= 1.0),
                class: classes.get(&t.trial).copied(),
                capabilities: labels.get(&t.task).cloned().unwrap_or_default(),
                correctness: t.correctness,
                cost: t.usage.map(|u| u.cost()),
                completion_tokens: t.usage.map(|u| u.completion),
            }));
            counted.extend(report.trials.iter().filter_map(|t| t.tokens));
        }
        // The budget's tokens: every trial's usage as its score record counts
        // it; unmeasured when no trial's record counts any.
        Ok(Rolled {
            export,
            trials,
            tokens: (!counted.is_empty()).then(|| counted.iter().sum()),
        })
    }
}

impl PracticeRollouts {
    /// The champion of `request` on `tasks` of `practice` (all of them when
    /// `None`), with the material's capability labels.
    fn practise(
        &self,
        ctx: &Context,
        request: &RolloutRequest<'_>,
        practice: &Practice,
        tasks: Option<&[String]>,
        cancel: &CancelToken,
    ) -> Result<(PracticeReport, BTreeMap<String, Vec<String>>), CampaignError> {
        let policy = request.champion.reference()?;
        let report = run_practice(
            ctx,
            &PracticeRequest {
                practice,
                checkout: &self.checkout,
                profile: self.profile.as_deref(),
                knowledge: None,
                policy: &policy,
                adapter: None,
                settings: PracticeSettings {
                    attempts: request.attempts,
                    seed: request.seed,
                    jobs: self.jobs,
                },
                tasks,
                cancel: cancel.clone(),
            },
        )
        .map_err(|e| match e {
            crate::episodes::EpisodeError::Cancelled => CampaignError::Cancelled,
            other => CampaignError::Collector(other.to_string()),
        })?;
        Ok((report, practice.capabilities()))
    }

    /// The transfer exercises of `transfer`, drawn from the material's
    /// collection into `<dir>/transfer` (the draw already there when a
    /// resumed rollout made it).
    fn draw(&self, transfer: &TransferRequest<'_>, dir: &Path) -> Result<Practice, CampaignError> {
        let out = dir.join("transfer");
        let drawn = out.join(self.practice.staged_name());
        if let Ok(practice) = Practice::read(&drawn) {
            return Ok(practice);
        }
        let _ = std::fs::remove_dir_all(&out);
        let manifest = &self.practice.manifest;
        let written = self
            .checkout
            .draw_practice(
                &PracticeDraw {
                    collection: &manifest.collection,
                    environment: &manifest.environment,
                    count: transfer.count,
                    seed: transfer.seed,
                    rung: self.transfer_rung,
                    families: &self.transfer_families,
                    capabilities: transfer.capabilities,
                },
                &out,
                &dir.join("transfer.log"),
            )
            .map_err(CampaignError::Collector)?;
        Practice::read(&written).map_err(|e| CampaignError::Collector(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::PracticeSpec;

    #[test]
    fn the_practice_section_may_name_a_rung_and_families_for_its_transfer_draws() {
        let plain: PracticeSpec = toml::from_str("dir = \"p\"\nbench = \"b\"\n").unwrap();
        assert_eq!((plain.rung, plain.families), (None, Vec::new()));
        let narrowed: PracticeSpec = toml::from_str(
            "dir = \"p\"\nbench = \"b\"\nrung = 2\nfamilies = [\"procurement\", \"deadline\"]\n",
        )
        .unwrap();
        assert_eq!(narrowed.rung, Some(2));
        let negative: PracticeSpec =
            toml::from_str("dir = \"p\"\nbench = \"b\"\nrung = -1\n").unwrap();
        assert_eq!(negative.rung, Some(-1));
        assert_eq!(narrowed.families, ["procurement", "deadline"]);
    }
}
