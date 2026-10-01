// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Process rewards: a score for each step, from the evaluations that stand.

use super::plan::{DataRef, Sample, SampleBody};
use super::recipe::Recipe;
use crate::analyze::EvalFilter;
use crate::error::Result;
use crate::manifest::Snapshot;
use crate::model::{RecordKind, Target};

impl Snapshot {
    pub(super) fn compile_prm(&self, recipe: &Recipe) -> Result<Vec<Sample>> {
        let index = self.index()?;
        let attempts: std::collections::HashMap<_, _> = self
            .selected_attempts(recipe)?
            .into_iter()
            .map(|a| (a.attempt, a.family))
            .collect();
        let mut filter = EvalFilter::new().criterion(&recipe.criterion);
        if let Some(c) = recipe.min_confidence {
            filter = filter.min_confidence(c);
        }
        let mut samples = Vec::new();
        for view in self.evaluations(&filter)? {
            let Target::Record(decision) = view.evaluation.target else {
                continue;
            };
            if index.kind_of(decision) != Some(RecordKind::Decision) {
                continue;
            }
            let Some(record) = self.get(decision)? else {
                continue;
            };
            let family = record.attempt.and_then(|a| attempts.get(&a).copied());
            if record.attempt.is_some() && family.is_none() {
                continue;
            }
            samples.push(Sample {
                provenance: decision,
                family,
                body: SampleBody::StepLabel {
                    context: DataRef::Context { decision },
                    step: DataRef::Action { decision },
                    score: view.evaluation.score,
                    confidence: view.evaluation.confidence,
                },
            });
        }
        Ok(samples)
    }
}
