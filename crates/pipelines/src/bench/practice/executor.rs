// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that turn an examiner's practice
// material into graded agent experience for its clients. If your team needs
// expertise in on-policy experience collection for agent training you can
// procure our services by sending an email to info@swedishembedded.com.

//! Practice material as the campaign's [`EpisodeExecutor`]: the policy a
//! campaign serves runs the examiner's staged tasks, and each graded trial
//! comes back as an [`Episode`].

use std::path::PathBuf;

use serde_json::{json, Value};
use splinter_agent::CancelToken;
use splinter_core::digest::Digest;
use splinter_core::model_ref::ModelRef;
use splinter_orchestrator::context::Context;

use super::{run_practice, BenchCheckout, Practice, PracticeRequest, PracticeSettings};
use crate::campaign::endpoint::{Episode, EpisodeExecutor};
use crate::campaign::{CampaignError, PolicyRef};
use crate::episodes::episode_of;

/// The examiner's practice material run for a campaign.
pub struct BenchPractice<'c> {
    ctx: &'c Context,
    practice: Practice,
    checkout: BenchCheckout,
    profile: Option<PathBuf>,
    /// The policy as configured: the base an adapter is served on.
    base: ModelRef,
    /// The policy the campaign serves now.
    serving: ModelRef,
    adapter: Option<Digest>,
    jobs: usize,
    cancel: CancelToken,
}

impl<'c> BenchPractice<'c> {
    /// An executor running `practice` through `checkout` under the policy
    /// `reference`, `jobs` trials at once.
    #[must_use]
    pub fn new(
        ctx: &'c Context,
        practice: Practice,
        checkout: BenchCheckout,
        reference: ModelRef,
        jobs: usize,
    ) -> Self {
        Self {
            ctx,
            practice,
            checkout,
            profile: None,
            serving: reference.clone(),
            base: reference,
            adapter: None,
            jobs,
            cancel: CancelToken::new(),
        }
    }

    /// This executor deriving the policy's profile from `profile` instead of
    /// the one the material was staged under.
    #[must_use]
    pub fn with_profile(mut self, profile: PathBuf) -> Self {
        self.profile = Some(profile);
        self
    }

    /// This executor, stopped when `cancel` fires.
    #[must_use]
    pub fn with_cancel(mut self, cancel: CancelToken) -> Self {
        self.cancel = cancel;
        self
    }
}

fn collector(why: impl std::fmt::Display) -> CampaignError {
    CampaignError::Collector(why.to_string())
}

impl EpisodeExecutor for BenchPractice<'_> {
    fn describe(&self) -> Value {
        json!({
            "executor": "bench practice",
            "policy": self.base.to_string(),
            "practice": {
                "dir": self.practice.dir,
                "sha256": self.practice.sha256,
                "collection": self.practice.manifest.collection,
                "draw": self.practice.manifest.draw,
            },
            "profile": self.profile,
            "jobs": self.jobs,
            "tasks": self.practice.manifest.tasks,
        })
    }

    fn tasks(&self) -> Vec<String> {
        self.practice
            .manifest
            .tasks
            .iter()
            .map(|t| t.name.clone())
            .collect()
    }

    fn serve(&mut self, policy: &PolicyRef) -> Result<(), CampaignError> {
        let Some(adapter) = &policy.adapter else {
            self.serving = self.base.clone();
            self.adapter = None;
            return Ok(());
        };
        let ModelRef::Local {
            checkpoint,
            context_tokens,
            ..
        } = &self.base
        else {
            return Err(CampaignError::Config(format!(
                "an adapter is served on a local base; the policy is {}",
                self.base
            )));
        };
        self.serving = ModelRef::Local {
            checkpoint: checkpoint.clone(),
            adapter: Some(adapter.to_string_lossy().into_owned()),
            context_tokens: *context_tokens,
        };
        self.adapter = policy.digest.clone();
        Ok(())
    }

    fn run(
        &mut self,
        tasks: &[String],
        attempts: usize,
        seed: u64,
    ) -> Result<Vec<Episode>, CampaignError> {
        let all = self.tasks();
        let chosen =
            (tasks.len() != all.len() || tasks.iter().any(|t| !all.contains(t))).then_some(tasks);
        let report = run_practice(
            self.ctx,
            &PracticeRequest {
                practice: &self.practice,
                checkout: &self.checkout,
                profile: self.profile.as_deref(),
                policy: &self.serving,
                adapter: self.adapter.clone(),
                settings: PracticeSettings {
                    attempts: u32::try_from(attempts).map_err(collector)?,
                    seed,
                    jobs: self.jobs,
                },
                tasks: chosen,
                cancel: self.cancel.clone(),
            },
        )
        .map_err(collector)?;
        let store = self.ctx.experiences();
        let mut episodes = Vec::with_capacity(report.trials.len());
        for trial in report.trials {
            // A harness trial's conversations share its grade; its episode
            // is its first conversation's.
            let experience = trial
                .experiences
                .first()
                .map(|id| store.get(id))
                .transpose()
                .map_err(collector)?;
            episodes.push(episode_of(trial.task, &trial.outcome, experience.as_ref()));
        }
        Ok(episodes)
    }
}
