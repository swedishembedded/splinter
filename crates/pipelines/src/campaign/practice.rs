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
//! the ingest stage to admit.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use splinter_agent::CancelToken;
use splinter_orchestrator::context::Context;

use super::types::CampaignError;
use super::work::{Rolled, RolloutRequest, Rollouts, Trial};
use crate::bench::practice::{
    run_practice, BenchCheckout, Practice, PracticeRequest, PracticeSettings,
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
        })
    }
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

    fn rollout(
        &mut self,
        ctx: &Context,
        request: &RolloutRequest<'_>,
        cancel: &CancelToken,
    ) -> Result<Rolled, CampaignError> {
        let policy = request.champion.reference()?;
        let report = run_practice(
            ctx,
            &PracticeRequest {
                practice: &self.practice,
                checkout: &self.checkout,
                profile: self.profile.as_deref(),
                policy: &policy,
                adapter: None,
                settings: PracticeSettings {
                    attempts: request.attempts,
                    seed: request.seed,
                    jobs: self.jobs,
                },
                tasks: Some(request.tasks),
                cancel: cancel.clone(),
            },
        )
        .map_err(|e| match e {
            crate::episodes::EpisodeError::Cancelled => CampaignError::Cancelled,
            other => CampaignError::Collector(other.to_string()),
        })?;
        let export = request.dir.join("export");
        link_tree(&report.dir.join("export"), &export).map_err(|e| {
            CampaignError::Collector(format!("linking the export into {}: {e}", export.display()))
        })?;
        let trials = report
            .trials
            .iter()
            .map(|t| Trial {
                task: t.task.clone(),
                reward: t.reward,
                passed: t.reward.map(|r| r >= 1.0),
            })
            .collect();
        Ok(Rolled {
            export,
            trials,
            tokens: None,
        })
    }
}
