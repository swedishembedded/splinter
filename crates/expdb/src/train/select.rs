// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Choosing the attempts a recipe may learn from.

use std::collections::HashMap;

use super::recipe::Recipe;
use crate::analyze::{EvalFilter, TASK_COMPLETION};
use crate::error::Result;
use crate::id::{ContentId, RecordId};
use crate::manifest::Snapshot;
use crate::model::{Epistemic, RecordKind, Target};

/// An attempt that passed the recipe's filters.
#[derive(Debug, Clone)]
pub(crate) struct AttemptInfo {
    pub(crate) attempt: RecordId,
    pub(crate) family: ContentId,
    pub(crate) task_instance: ContentId,
    /// The reward it stands at, if one was measured.
    pub(crate) reward: Option<f64>,
}

impl Snapshot {
    /// The attempts the recipe's policy, confidence and verification
    /// filters allow, with their rewards. The reward may be absent.
    pub(crate) fn selected_attempts(&self, recipe: &Recipe) -> Result<Vec<AttemptInfo>> {
        let mut evidence: HashMap<RecordId, (f64, bool)> = HashMap::new();
        for view in self.evaluations(&EvalFilter::new().criterion(TASK_COMPLETION))? {
            if let Target::Record(attempt) = view.evaluation.target {
                let entry = evidence.entry(attempt).or_insert((1.0, false));
                entry.0 = entry.0.min(view.evaluation.confidence);
                entry.1 |= view.evaluation.epistemic == Epistemic::Fact;
            }
        }
        let mut selected = Vec::new();
        for family in self.families()? {
            for attempt in family.attempts {
                let (confidence, verified) = evidence
                    .get(&attempt.attempt)
                    .copied()
                    .unwrap_or((1.0, false));
                let policy_ok = recipe.policy.as_ref().is_none_or(|(name, version)| {
                    attempt.policy.name == *name && attempt.policy.version == *version
                });
                if policy_ok
                    && recipe.min_confidence.is_none_or(|c| confidence >= c)
                    && (!recipe.verified_only || verified)
                {
                    selected.push(AttemptInfo {
                        attempt: attempt.attempt,
                        family: family.key,
                        task_instance: family.task_instance,
                        reward: attempt.reward,
                    });
                }
            }
        }
        selected.sort_by_key(|a| a.attempt);
        Ok(selected)
    }

    /// The decisions of an attempt, in order.
    pub(crate) fn decisions_of(&self, attempt: RecordId) -> Result<Vec<RecordId>> {
        let index = self.index()?;
        Ok(index
            .by_attempt(attempt)
            .into_iter()
            .filter(|id| index.kind_of(*id) == Some(RecordKind::Decision))
            .collect())
    }
}
