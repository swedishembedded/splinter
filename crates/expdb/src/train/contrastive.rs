// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Cross-modal alignment: the same moment of two modalities is a positive;
//! a different moment of the same episode is a hard negative, a moment of
//! another episode an easy one.

use super::mm_select::{common_extent, grid, EpisodeCtx};
use super::plan::{DataRef, Negative, Sample, SampleBody};
use super::recipe::Recipe;
use super::rng::Rng;
use crate::error::{Error, Result};
use crate::manifest::Snapshot;
use crate::model::TimeRange;

const MILLISECOND: i64 = 1_000_000;

impl Snapshot {
    pub(super) fn compile_contrastive(&self, recipe: &Recipe) -> Result<Vec<Sample>> {
        let spec = &recipe.mm;
        let (a_name, b_name) = spec.pair.as_ref().ok_or_else(|| {
            Error::invalid(
                "recipe",
                "a contrastive recipe needs the two modalities to align",
            )
        })?;
        let w = spec.window_ns;
        let episodes = self.episode_contexts(recipe)?;
        let mut samples = Vec::new();
        for (e, episode) in episodes.iter().enumerate() {
            let (Some(a), Some(b)) = (episode.stream(a_name), episode.stream(b_name)) else {
                continue;
            };
            let Some(extent) = common_extent(&[a, b]) else {
                continue;
            };
            for t in grid(extent.start_ns, extent.end_ns - w, spec.stride_ns) {
                let at = |from: i64| TimeRange {
                    start_ns: from,
                    end_ns: from + w,
                };
                let mut negatives = Vec::new();
                if spec.negatives > 0 {
                    let shifted = [t + spec.hard_shift_ns, t - spec.hard_shift_ns]
                        .into_iter()
                        .find(|s| *s != t && *s >= extent.start_ns && *s + w <= extent.end_ns);
                    if let Some(s) = shifted {
                        negatives.push(Negative {
                            data: DataRef::Window {
                                stream: b.id,
                                interval: at(s),
                            },
                            hard: true,
                        });
                    }
                }
                let mut rng = Rng::new(recipe.seed ^ episode.id.hash64() ^ (t as u64));
                let others: Vec<&EpisodeCtx> = episodes
                    .iter()
                    .enumerate()
                    .filter(|(i, other)| {
                        *i != e && other.stream(b_name).and_then(|s| s.extent).is_some()
                    })
                    .map(|(_, other)| other)
                    .collect();
                while negatives.len() < spec.negatives && !others.is_empty() {
                    let other = others[rng.below(others.len())];
                    let Some(stream) = other.stream(b_name) else {
                        break;
                    };
                    let Some(span) = stream.extent.filter(|s| s.len_ns() >= w) else {
                        break;
                    };
                    let room = ((span.len_ns() - w) / MILLISECOND).max(0) as usize;
                    let from = span.start_ns + rng.below(room + 1) as i64 * MILLISECOND;
                    negatives.push(Negative {
                        data: DataRef::Window {
                            stream: stream.id,
                            interval: at(from),
                        },
                        hard: false,
                    });
                }
                samples.push(Sample {
                    provenance: episode.id,
                    family: Some(episode.family),
                    body: SampleBody::Contrastive {
                        anchor: DataRef::Window {
                            stream: a.id,
                            interval: at(t),
                        },
                        positive: DataRef::Window {
                            stream: b.id,
                            interval: at(t),
                        },
                        negatives,
                    },
                });
            }
        }
        Ok(samples)
    }
}
