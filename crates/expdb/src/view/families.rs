// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Episode families: every attempt at one task instance from one world.

use std::collections::{BTreeMap, HashMap};

use crate::error::{Error, Result};
use crate::id::{ContentId, RecordId};
use crate::manifest::Snapshot;
use crate::model::{family_key, Body, Outcome, PolicyRef, RecordKind};

/// One attempt of a family and how it ended.
#[derive(Debug, Clone, PartialEq)]
pub struct AttemptView {
    /// The attempt's record id.
    pub attempt: RecordId,
    /// The policy that attempted it.
    pub policy: PolicyRef,
    /// The random seed, when the run was seeded.
    pub seed: Option<u64>,
    /// How the environment said it ended, if it ended.
    pub outcome: Option<Outcome>,
    /// The reward the attempt stands at: the confidence-weighted score of
    /// its standing task-completion evaluations if it has any, else 1 for a
    /// pass and 0 for a fail. Absent for an attempt that was aborted or has
    /// not ended and was never evaluated.
    pub reward: Option<f64>,
}

/// All attempts that begin from one task instance in one world.
#[derive(Debug, Clone, PartialEq)]
pub struct FamilyView {
    /// The family's key.
    pub key: ContentId,
    /// The task instance.
    pub task_instance: ContentId,
    /// The world every attempt starts in.
    pub initial_state: ContentId,
    /// The attempts, in id order.
    pub attempts: Vec<AttemptView>,
}

impl FamilyView {
    fn rewards(&self) -> Vec<f64> {
        self.attempts.iter().filter_map(|a| a.reward).collect()
    }

    /// The share of measured attempts that passed. `None` if none was
    /// measured: unmeasured is not zero.
    pub fn pass_rate(&self) -> Option<f64> {
        let rewards = self.rewards();
        (!rewards.is_empty()).then(|| rewards.iter().sum::<f64>() / rewards.len() as f64)
    }

    /// The variance of the measured rewards: how much a group-relative
    /// update could learn from this family. `None` if none was measured.
    pub fn reward_variance(&self) -> Option<f64> {
        let mean = self.pass_rate()?;
        let rewards = self.rewards();
        Some(rewards.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / rewards.len() as f64)
    }

    /// Whether some attempts stand at a pass (reward of at least one half)
    /// and some at a fail.
    pub fn has_both_outcomes(&self) -> bool {
        let rewards = self.rewards();
        rewards.iter().any(|r| *r >= 0.5) && rewards.iter().any(|r| *r < 0.5)
    }
}

impl Snapshot {
    /// Every family with its attempts.
    pub fn families(&self) -> Result<Vec<FamilyView>> {
        let index = self.index()?;
        let mut outcomes: HashMap<RecordId, Outcome> = HashMap::new();
        for id in index.by_kind(RecordKind::AttemptEnd) {
            if let Some(Body::AttemptEnd { attempt, outcome }) = self.get(id)?.map(|r| r.body) {
                outcomes.insert(attempt, outcome);
            }
        }
        let evaluated = self.attempt_rewards()?;
        let mut attempts: BTreeMap<ContentId, Vec<AttemptView>> = BTreeMap::new();
        for id in index.by_kind(RecordKind::Attempt) {
            let Some(Body::Attempt(a)) = self.get(id)?.map(|r| r.body) else {
                continue;
            };
            let outcome = outcomes.get(&id).copied();
            let reward = match evaluated.get(&id) {
                Some(measured) => *measured,
                None => match outcome {
                    Some(Outcome::Pass) => Some(1.0),
                    Some(Outcome::Fail) => Some(0.0),
                    Some(Outcome::Aborted) | None => None,
                },
            };
            attempts.entry(a.family).or_default().push(AttemptView {
                attempt: id,
                policy: a.policy,
                seed: a.seed,
                outcome,
                reward,
            });
        }
        let mut meta: HashMap<ContentId, (ContentId, ContentId)> = HashMap::new();
        for id in index.by_kind(RecordKind::Family) {
            if let Some(Body::Family(f)) = self.get(id)?.map(|r| r.body) {
                meta.insert(
                    family_key(&f.task_instance, &f.initial_state),
                    (f.task_instance, f.initial_state),
                );
            }
        }
        attempts
            .into_iter()
            .map(|(key, attempts)| {
                let (task_instance, initial_state) = *meta.get(&key).ok_or_else(|| {
                    Error::corrupt(
                        format!("family {key}"),
                        "attempts exist but no family record",
                    )
                })?;
                Ok(FamilyView {
                    key,
                    task_instance,
                    initial_state,
                    attempts,
                })
            })
            .collect()
    }

    /// One family, if it has attempts.
    pub fn family(&self, key: &ContentId) -> Result<Option<FamilyView>> {
        Ok(self.families()?.into_iter().find(|f| f.key == *key))
    }

    /// Families where some attempts passed and some failed: the ones with
    /// the strongest contrast for preference and group-relative learning.
    pub fn mixed_outcome_families(&self) -> Result<Vec<FamilyView>> {
        Ok(self
            .families()?
            .into_iter()
            .filter(FamilyView::has_both_outcomes)
            .collect())
    }
}
