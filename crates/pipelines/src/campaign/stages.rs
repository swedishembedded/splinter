// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that run model learning as one
// continuous, resumable campaign with dynamic sampling, replay and a
// validation gate with rollback, for its clients. If your team needs expertise
// in on-policy training loops for language-model agents you can procure our
// services by sending an email to info@swedishembedded.com.

//! The collecting stages of an arm: calibration, the over-provisioned batch,
//! and the records the batch becomes.

use splinter_eval::campaign::sampling::{draw_prompts, AllFailStrategy, BatchBuilder};

use super::config::{AdvantageMode, LearningStrategy};
use super::controller::{CampaignController, Collected};
use super::records::{passing_records, BatchRecords};
use super::types::{CampaignError, CollectRequest, RolloutGroup};

impl CampaignController<'_> {
    pub(super) fn passing(&self, collected: &Collected) -> Vec<splinter_data::Record> {
        let groups: Vec<&RolloutGroup> = collected
            .informative
            .iter()
            .map(|&i| &collected.groups[i])
            .collect();
        passing_records(&groups)
    }

    /// Attempts every task nobody has measured k times under the champion and
    /// counts the outcomes; saves after each request.
    pub(super) fn calibrate(&mut self, generation: u32) -> Result<(), CampaignError> {
        let pool = self.pool();
        let todo: Vec<String> = self
            .loaded
            .state
            .rates
            .uncalibrated(&pool)
            .into_iter()
            .cloned()
            .collect();
        let mut rng = self.rng(generation, 2);
        for chunk in todo.chunks(self.config.calibration_batch) {
            let groups = self.parts.collector.collect(&CollectRequest {
                tasks: chunk.to_vec(),
                attempts: self.config.calibration_k,
                policy: self.loaded.state.policy.clone(),
                seed: rng.next_u64(),
            })?;
            let version = self.loaded.state.policy.version;
            for group in &groups {
                let outcomes: Vec<Option<bool>> =
                    group.rollouts.iter().map(|r| Some(r.passed)).collect();
                self.loaded
                    .state
                    .rates
                    .observe(&group.task, version, &outcomes);
            }
            self.save()?;
        }
        Ok(())
    }

    /// Builds the batch: over-provisioned requests, refilled until the
    /// batch is full or the rounds are spent.
    pub(super) fn collect(&mut self, generation: u32) -> Result<Collected, CampaignError> {
        let mut builder = BatchBuilder::new(self.config.batch)
            .map_err(|e| CampaignError::Config(e.to_string()))?;
        let mut rng = self.rng(generation, 3);
        let mut groups: Vec<RolloutGroup> = Vec::new();
        let pool = self.pool();
        while let Some(wanted) = builder.next_request() {
            let tasks = draw_prompts(&pool, &self.loaded.state.rates, wanted, &mut rng);
            if tasks.is_empty() {
                break;
            }
            let got = self.parts.collector.collect(&CollectRequest {
                tasks,
                attempts: self.config.attempts_per_group,
                policy: self.loaded.state.policy.clone(),
                seed: rng.next_u64(),
            })?;
            let version = self.loaded.state.policy.version;
            let first = groups.len() as u64;
            let summaries = got
                .iter()
                .enumerate()
                .map(|(i, g)| g.summary(first + i as u64))
                .collect();
            for group in &got {
                let outcomes: Vec<Option<bool>> =
                    group.rollouts.iter().map(|r| Some(r.passed)).collect();
                self.loaded
                    .state
                    .rates
                    .observe(&group.task, version, &outcomes);
            }
            groups.extend(got);
            builder.offer(summaries);
        }
        let batch = builder.finish();
        Ok(Collected {
            informative_fraction: batch.informative_fraction(),
            informative: batch.informative.iter().map(|g| g.tag as usize).collect(),
            all_fail: batch.all_fail.iter().map(|g| g.tag as usize).collect(),
            rollouts: groups.iter().map(|g| g.rollouts.len() as u64).sum(),
            shortfall: batch.shortfall,
            groups,
        })
    }

    pub(super) fn records(
        &mut self,
        generation: u32,
        strategy: LearningStrategy,
        collected: &Collected,
    ) -> Result<BatchRecords, CampaignError> {
        let informative: Vec<&RolloutGroup> = collected
            .informative
            .iter()
            .map(|&i| &collected.groups[i])
            .collect();
        let all_fail: Vec<&RolloutGroup> = collected
            .all_fail
            .iter()
            .map(|&i| &collected.groups[i])
            .collect();
        let mut built = BatchRecords::default();
        let mode = match strategy {
            LearningStrategy::PositiveOnly => AdvantageMode::PositiveOnly,
            _ => AdvantageMode::TwoSided,
        };
        built.add_advantages(&informative, mode);
        if strategy == LearningStrategy::Repair {
            built.add_repairs(&informative, self.config.repairs_per_group);
        }
        let route = if strategy == LearningStrategy::Distill {
            AllFailStrategy::CliffDistill
        } else {
            self.config.all_fail
        };
        match route {
            AllFailStrategy::Skip => {}
            AllFailStrategy::CliffDistill => {
                built.add_distillation(&all_fail, self.config.distilled_per_group);
            }
            AllFailStrategy::Teacher => {
                if let Some(handler) = self.parts.handler.as_deref_mut() {
                    built.add_taught(handler.teach(&all_fail)?);
                }
            }
            AllFailStrategy::Decompose => {
                if let Some(handler) = self.parts.handler.as_deref_mut() {
                    let tasks = handler.decompose(&all_fail)?;
                    self.loaded.state.extra_tasks.extend(tasks);
                }
            }
        }
        let fresh = built.records.len();
        let wanted = (fresh as f64 * self.config.replay.per_batch_record).round() as usize;
        let mut rng = self.rng(generation, 4);
        built.add_replay(
            self.loaded
                .state
                .replay
                .sample(wanted, generation, &mut rng),
        );
        Ok(built)
    }
}
