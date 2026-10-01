// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Action chunks: the actions that follow, given the recent past of the
//! observation streams and the last instruction.

use super::mm_select::{common_extent, grid};
use super::plan::{DataRef, Sample, SampleBody};
use super::recipe::Recipe;
use crate::error::Result;
use crate::manifest::Snapshot;
use crate::model::TimeRange;

impl Snapshot {
    pub(super) fn compile_action_chunk(&self, recipe: &Recipe) -> Result<Vec<Sample>> {
        let spec = &recipe.mm;
        let mut samples = Vec::new();
        for episode in self.episode_contexts(recipe)? {
            let observed = episode.observed(&spec.streams);
            let Some(extent) = common_extent(&observed) else {
                continue;
            };
            for t in grid(
                extent.start_ns + spec.past_ns,
                extent.end_ns - spec.horizon_ns,
                spec.stride_ns,
            ) {
                let actions: Vec<DataRef> = episode
                    .actions
                    .iter()
                    .filter(|(start, _)| *start >= t && *start < t + spec.horizon_ns)
                    .map(|(_, record)| DataRef::ActionSegment { record: *record })
                    .collect();
                if actions.is_empty() {
                    continue;
                }
                let interval = TimeRange::new(t - spec.past_ns, t)?;
                let instruction = episode
                    .instructions
                    .iter()
                    .rev()
                    .find(|(at, _)| *at <= t)
                    .map(|(_, record)| DataRef::EventPayload { record: *record });
                samples.push(Sample {
                    provenance: episode.id,
                    family: Some(episode.family),
                    body: SampleBody::ActionChunk {
                        observations: observed
                            .iter()
                            .map(|s| DataRef::Window {
                                stream: s.id,
                                interval,
                            })
                            .collect(),
                        instruction,
                        actions,
                    },
                });
            }
        }
        Ok(samples)
    }
}
