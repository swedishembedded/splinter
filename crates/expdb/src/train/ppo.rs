// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Single transitions, with what an on-policy trainer needs.

use std::collections::HashMap;

use super::plan::{DataRef, Sample, SampleBody};
use super::recipe::Recipe;
use crate::error::Result;
use crate::id::{ContentId, RecordId};
use crate::manifest::Snapshot;
use crate::model::{Body, RecordKind};

impl Snapshot {
    pub(super) fn compile_ppo(&self, recipe: &Recipe) -> Result<Vec<Sample>> {
        let index = self.index()?;
        let attempts: HashMap<RecordId, (ContentId, Option<f64>)> = self
            .selected_attempts(recipe)?
            .into_iter()
            .map(|a| (a.attempt, (a.family, a.reward)))
            .collect();
        let mut samples = Vec::new();
        for id in index.by_kind(RecordKind::Transition) {
            let Some(record) = self.get(id)? else {
                continue;
            };
            let (Body::Transition(t), Some(attempt)) = (&record.body, record.attempt) else {
                continue;
            };
            let Some((family, attempt_reward)) = attempts.get(&attempt) else {
                continue;
            };
            let Some(Body::Decision(decision)) = self.get(t.decision)?.map(|r| r.body) else {
                continue;
            };
            let done = index
                .children(id)
                .iter()
                .any(|c| index.kind_of(*c) == Some(RecordKind::AttemptEnd));
            let reward = match (t.reward, done && recipe.terminal_reward) {
                (None, true) => *attempt_reward,
                (given, _) => given,
            };
            samples.push(Sample {
                provenance: id,
                family: Some(*family),
                body: SampleBody::Transition {
                    context: DataRef::Context {
                        decision: t.decision,
                    },
                    action: DataRef::Action {
                        decision: t.decision,
                    },
                    state: t.from,
                    next_state: t.to,
                    old_logprob: decision.old_logprob,
                    value_estimate: decision.value_estimate,
                    reward,
                    done,
                },
            });
        }
        Ok(samples)
    }
}
