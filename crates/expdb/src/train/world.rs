// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! World models: what the streams do next, given what they did and what was
//! done. Anchored on actions when there are any, on a grid when there are
//! none, so passive observation trains too.

use super::mm_select::{common_extent, grid};
use super::plan::{DataRef, Sample, SampleBody};
use super::recipe::Recipe;
use crate::error::Result;
use crate::manifest::Snapshot;
use crate::model::TimeRange;

impl Snapshot {
    pub(super) fn compile_world_model(&self, recipe: &Recipe) -> Result<Vec<Sample>> {
        let spec = &recipe.mm;
        let mut samples = Vec::new();
        for episode in self.episode_contexts(recipe)? {
            let observed = episode.observed(&spec.streams);
            let Some(extent) = common_extent(&observed) else {
                continue;
            };
            let anchors: Vec<i64> = if episode.actions.is_empty() {
                grid(
                    extent.start_ns + spec.past_ns,
                    extent.end_ns - spec.future_ns,
                    spec.stride_ns,
                )
            } else {
                let mut starts: Vec<i64> = episode.actions.iter().map(|(t, _)| *t).collect();
                starts.dedup();
                starts
            };
            for t in anchors {
                if t - spec.past_ns < extent.start_ns || t + spec.future_ns > extent.end_ns {
                    continue;
                }
                let window = |from: i64, to: i64| -> Result<Vec<DataRef>> {
                    let interval = TimeRange::new(from, to)?;
                    Ok(observed
                        .iter()
                        .map(|s| DataRef::Window {
                            stream: s.id,
                            interval,
                        })
                        .collect())
                };
                let actions = episode
                    .actions
                    .iter()
                    .filter(|(start, _)| *start >= t && *start < t + spec.future_ns)
                    .map(|(_, record)| DataRef::ActionSegment { record: *record })
                    .collect();
                samples.push(Sample {
                    provenance: episode.id,
                    family: Some(episode.family),
                    body: SampleBody::WorldModel {
                        context: window(t - spec.past_ns, t)?,
                        actions,
                        target: window(t, t + spec.future_ns)?,
                    },
                });
            }
        }
        Ok(samples)
    }
}
