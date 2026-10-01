// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Masked prediction: hide a window of one stream and show everything
//! around it, in every stream, including the same stream either side.

use super::mm_select::{common_extent, grid};
use super::plan::{DataRef, Sample, SampleBody};
use super::recipe::Recipe;
use crate::error::{Error, Result};
use crate::manifest::Snapshot;
use crate::model::TimeRange;

impl Snapshot {
    pub(super) fn compile_masked(&self, recipe: &Recipe) -> Result<Vec<Sample>> {
        let spec = &recipe.mm;
        let target_name = spec
            .target_stream
            .as_ref()
            .ok_or_else(|| Error::invalid("recipe", "a masked recipe needs the stream to hide"))?;
        let (w, ctx) = (spec.window_ns, spec.context_ns);
        let mut samples = Vec::new();
        for episode in self.episode_contexts(recipe)? {
            let Some(target) = episode.stream(target_name) else {
                continue;
            };
            let others: Vec<_> = episode
                .observed(&spec.streams)
                .into_iter()
                .filter(|s| s.id != target.id)
                .collect();
            let mut all: Vec<&super::mm_select::StreamCtx> = others.clone();
            all.push(target);
            let Some(extent) = common_extent(&all) else {
                continue;
            };
            for t in grid(
                extent.start_ns + ctx,
                extent.end_ns - w - ctx,
                spec.stride_ns,
            ) {
                let span = |from: i64, to: i64| TimeRange {
                    start_ns: from,
                    end_ns: to,
                };
                let mut visible: Vec<DataRef> = others
                    .iter()
                    .map(|s| DataRef::Window {
                        stream: s.id,
                        interval: span(t - ctx, t + w + ctx),
                    })
                    .collect();
                visible.push(DataRef::Window {
                    stream: target.id,
                    interval: span(t - ctx, t),
                });
                visible.push(DataRef::Window {
                    stream: target.id,
                    interval: span(t + w, t + w + ctx),
                });
                samples.push(Sample {
                    provenance: episode.id,
                    family: Some(episode.family),
                    body: SampleBody::Masked {
                        visible,
                        masked: DataRef::Window {
                            stream: target.id,
                            interval: span(t, t + w),
                        },
                    },
                });
            }
        }
        Ok(samples)
    }
}
