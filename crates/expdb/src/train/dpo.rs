// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Preference: of the decisions taken at one state in one context, the one
//! whose attempt did better is chosen over the one whose attempt did worse.

use std::collections::{BTreeMap, HashMap};

use super::plan::{DataRef, Sample, SampleBody};
use super::recipe::Recipe;
use crate::error::Result;
use crate::id::{ContentId, RecordId};
use crate::manifest::Snapshot;
use crate::model::{Body, RecordKind};

struct Candidate {
    decision: RecordId,
    family: ContentId,
    reward: f64,
}

impl Snapshot {
    /// The mean credit an algorithm gave each decision.
    fn credit_by_decision(&self, algorithm: &str) -> Result<HashMap<RecordId, f64>> {
        let index = self.index()?;
        let mut sums: HashMap<RecordId, (f64, usize)> = HashMap::new();
        for id in index.by_kind(RecordKind::Credit) {
            if let Some(Body::Credit(credit)) = self.get(id)?.map(|r| r.body) {
                if credit.algorithm == algorithm {
                    let entry = sums.entry(credit.target).or_default();
                    entry.0 += credit.value;
                    entry.1 += 1;
                }
            }
        }
        Ok(sums
            .into_iter()
            .map(|(id, (sum, n))| (id, sum / n as f64))
            .collect())
    }

    pub(super) fn compile_dpo(&self, recipe: &Recipe) -> Result<Vec<Sample>> {
        let index = self.index()?;
        let attempts: HashMap<RecordId, (ContentId, Option<f64>)> = self
            .selected_attempts(recipe)?
            .into_iter()
            .map(|a| (a.attempt, (a.family, a.reward)))
            .collect();
        let credits = recipe
            .credit
            .as_deref()
            .map(|a| self.credit_by_decision(a))
            .transpose()?;

        let mut groups: BTreeMap<ContentId, Vec<Candidate>> = BTreeMap::new();
        for id in index.by_kind(RecordKind::Decision) {
            let Some(record) = self.get(id)? else {
                continue;
            };
            let (Body::Decision(decision), Some(attempt)) = (&record.body, record.attempt) else {
                continue;
            };
            let Some((family, outcome_reward)) = attempts.get(&attempt) else {
                continue;
            };
            let reward = match &credits {
                Some(credits) => credits.get(&id).copied(),
                None => *outcome_reward,
            };
            let Some(reward) = reward else { continue };
            // Only decisions that saw the same world and the same context are
            // comparable: same prompt, different choice.
            let context = self.render(&DataRef::Context { decision: id })?;
            let mut key = decision.state.as_bytes().to_vec();
            key.extend_from_slice(context.as_bytes());
            groups
                .entry(ContentId::of(&key))
                .or_default()
                .push(Candidate {
                    decision: id,
                    family: *family,
                    reward,
                });
        }

        let mut samples = Vec::new();
        for mut group in groups.into_values() {
            if group.len() < 2 {
                continue;
            }
            group.sort_by(|a, b| {
                b.reward
                    .total_cmp(&a.reward)
                    .then(a.decision.cmp(&b.decision))
            });
            let (best, worst) = (&group[0], &group[group.len() - 1]);
            let gap = best.reward - worst.reward;
            if gap < recipe.min_gap {
                continue;
            }
            let chosen = DataRef::Action {
                decision: best.decision,
            };
            let rejected = DataRef::Action {
                decision: worst.decision,
            };
            if self.render(&chosen)? == self.render(&rejected)? {
                continue;
            }
            samples.push(Sample {
                provenance: best.decision,
                family: Some(best.family),
                body: SampleBody::Preference {
                    context: DataRef::Context {
                        decision: best.decision,
                    },
                    chosen,
                    rejected,
                    reward_gap: gap,
                },
            });
        }
        Ok(samples)
    }
}
