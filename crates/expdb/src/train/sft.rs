// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Imitation: the steps, or the whole episodes, of attempts that succeeded.

use super::plan::{DataRef, Sample, SampleBody};
use super::recipe::Recipe;
use crate::error::Result;
use crate::manifest::Snapshot;

impl Snapshot {
    pub(super) fn compile_sft(&self, recipe: &Recipe) -> Result<Vec<Sample>> {
        let mut samples = Vec::new();
        for attempt in self.selected_attempts(recipe)? {
            if attempt.reward.is_none_or(|r| r < recipe.min_reward) {
                continue;
            }
            if recipe.episodes {
                samples.push(Sample {
                    provenance: attempt.attempt,
                    family: Some(attempt.family),
                    body: SampleBody::Sft {
                        context: DataRef::Task {
                            instance: attempt.task_instance,
                        },
                        target: DataRef::Episode {
                            attempt: attempt.attempt,
                        },
                        weight: 1.0,
                    },
                });
                continue;
            }
            for decision in self.decisions_of(attempt.attempt)? {
                samples.push(Sample {
                    provenance: decision,
                    family: Some(attempt.family),
                    body: SampleBody::Sft {
                        context: DataRef::Context { decision },
                        target: DataRef::Action { decision },
                        weight: 1.0,
                    },
                });
            }
        }
        Ok(samples)
    }
}
