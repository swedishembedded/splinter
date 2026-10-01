// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Group-relative learning: the attempts of one family compared against each
//! other, where the family is neither always solved nor never solved.

use super::plan::{DataRef, Rollout, Sample, SampleBody};
use super::recipe::Recipe;
use super::rng::Rng;
use crate::error::Result;
use crate::id::ContentId;
use crate::manifest::Snapshot;

impl Snapshot {
    pub(super) fn compile_grpo(&self, recipe: &Recipe) -> Result<Vec<Sample>> {
        let mut by_family: std::collections::BTreeMap<ContentId, Vec<_>> =
            std::collections::BTreeMap::new();
        for attempt in self.selected_attempts(recipe)? {
            if attempt.reward.is_some() {
                by_family.entry(attempt.family).or_default().push(attempt);
            }
        }
        let mut samples = Vec::new();
        for (family, mut attempts) in by_family {
            if recipe.group_size == 0 || attempts.len() < recipe.group_size {
                continue;
            }
            if attempts.len() > recipe.group_size {
                Rng::new(recipe.seed ^ family.prefix_u64()).shuffle(&mut attempts);
                attempts.truncate(recipe.group_size);
            }
            attempts.sort_by_key(|a| a.attempt);
            let rewards: Vec<f64> = attempts.iter().filter_map(|a| a.reward).collect();
            let n = rewards.len() as f64;
            let mean = rewards.iter().sum::<f64>() / n;
            let variance = rewards.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
            let mixed = rewards.iter().any(|r| *r >= 0.5) && rewards.iter().any(|r| *r < 0.5);
            if variance < recipe.min_variance || (recipe.require_mixed && !mixed) {
                continue;
            }
            let deviation = variance.sqrt();
            let rollouts = attempts
                .iter()
                .zip(&rewards)
                .map(|(a, r)| Rollout {
                    response: DataRef::Episode { attempt: a.attempt },
                    reward: *r,
                    advantage: if deviation > 0.0 {
                        (r - mean) / deviation
                    } else {
                        0.0
                    },
                })
                .collect();
            samples.push(Sample {
                provenance: attempts[0].attempt,
                family: Some(family),
                body: SampleBody::Group {
                    context: DataRef::Task {
                        instance: attempts[0].task_instance,
                    },
                    rollouts,
                },
            });
        }
        Ok(samples)
    }
}
