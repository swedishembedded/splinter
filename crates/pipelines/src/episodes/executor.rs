// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that turn practice task folders
// into graded, replayable agent experience for its clients. If your team
// needs expertise in on-policy experience collection for agent training you
// can procure our services by sending an email to info@swedishembedded.com.

//! Episodes as the campaign's [`EpisodeExecutor`]: the policy a campaign
//! serves runs the practice tasks through sven, and each graded attempt
//! comes back as an [`Episode`] with the calls it sampled.

use serde_json::{json, Value};
use splinter_agent::CancelToken;
use splinter_core::chat::WireMessage;
use splinter_core::digest::Digest;
use splinter_core::model_ref::ModelRef;
use splinter_orchestrator::context::Context;

use super::grade::rollout_trace;
use super::{run_with_adapter, EpisodeOutcome, EpisodeRequest, EpisodeSettings, PracticeTask};
use crate::campaign::endpoint::{Episode, EpisodeExecutor};
use crate::campaign::{CampaignError, PolicyRef};

/// Practice tasks run through sven for a campaign.
pub struct SvenEpisodes<'c> {
    ctx: &'c Context,
    tasks: Vec<PracticeTask>,
    /// The policy as configured: the base an adapter is served on.
    base: ModelRef,
    /// The policy the campaign serves now.
    serving: ModelRef,
    adapter: Option<Digest>,
    settings: EpisodeSettings,
    cancel: CancelToken,
}

impl<'c> SvenEpisodes<'c> {
    /// An executor running `tasks` under the policy `reference`, each
    /// attempt bounded by `settings` (whose attempts and seed each run's
    /// request replaces).
    #[must_use]
    pub fn new(
        ctx: &'c Context,
        tasks: Vec<PracticeTask>,
        reference: ModelRef,
        settings: EpisodeSettings,
    ) -> Self {
        Self {
            ctx,
            tasks,
            serving: reference.clone(),
            base: reference,
            adapter: None,
            settings,
            cancel: CancelToken::new(),
        }
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

impl EpisodeExecutor for SvenEpisodes<'_> {
    fn describe(&self) -> Value {
        json!({
            "executor": "sven episodes",
            "policy": self.base.to_string(),
            "settings": {
                "jobs": self.settings.jobs,
                "deadline_secs": self.settings.deadline.as_secs_f64(),
                "max_tool_calls": self.settings.max_tool_calls,
                "max_output_tokens": self.settings.max_output_tokens,
            },
            "tasks": self.tasks.iter().map(|t| json!({
                "name": t.name, "task": t.task, "source": t.source,
            })).collect::<Vec<_>>(),
        })
    }

    fn tasks(&self) -> Vec<String> {
        self.tasks.iter().map(|t| t.task.to_string()).collect()
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
        let chosen = tasks
            .iter()
            .map(|name| {
                self.tasks
                    .iter()
                    .find(|t| t.task.to_string() == *name)
                    .cloned()
                    .ok_or_else(|| collector(format!("{name} is not a practice task of this run")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let settings = EpisodeSettings {
            attempts: u32::try_from(attempts).map_err(collector)?,
            seed,
            ..self.settings.clone()
        };
        let report = run_with_adapter(
            self.ctx,
            &EpisodeRequest {
                tasks: &chosen,
                policy: &self.serving,
                settings,
                cancel: self.cancel.clone(),
            },
            self.adapter.clone(),
        )
        .map_err(collector)?;
        let store = self.ctx.experiences();
        let mut episodes = Vec::with_capacity(report.records.len());
        for record in report.records {
            let experience = record
                .experience
                .as_ref()
                .map(|id| store.get(id))
                .transpose()
                .map_err(collector)?;
            episodes.push(Episode {
                task: record.task.to_string(),
                reward: record.reward,
                passed: record.outcome == EpisodeOutcome::Passed,
                prompt: experience
                    .iter()
                    .map(|e| WireMessage {
                        role: "user".into(),
                        content: e.instruction.clone(),
                        images: Vec::new(),
                        tool_calls: Vec::new(),
                        tool_call_id: None,
                        reasoning_content: None,
                        train: false,
                    })
                    .collect(),
                completion: experience
                    .as_ref()
                    .and_then(|e| e.final_output.clone())
                    .unwrap_or_default(),
                feedback: None,
                privileged: None,
                trace: experience.as_ref().map(|e| rollout_trace(&e.trajectory)),
            });
        }
        Ok(episodes)
    }
}
