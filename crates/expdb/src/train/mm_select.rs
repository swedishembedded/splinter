// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! What the multimodal objectives share: the streams of each episode, the
//! time they cover on the episode's clock, and where its actions and
//! instructions fall.

use super::recipe::Recipe;
use crate::error::Result;
use crate::id::{ContentId, RecordId};
use crate::manifest::Snapshot;
use crate::model::{StreamOrigin, TimeRange};

/// One stream of an episode, placed on the episode's clock.
pub(crate) struct StreamCtx {
    pub(crate) id: RecordId,
    pub(crate) name: String,
    /// Where the stream has data, on the episode's clock.
    pub(crate) extent: Option<TimeRange>,
    pub(crate) raw: bool,
}

/// One episode, ready to be windowed.
pub(crate) struct EpisodeCtx {
    pub(crate) id: RecordId,
    pub(crate) family: ContentId,
    pub(crate) streams: Vec<StreamCtx>,
    /// Actions in start order, with their start on the episode's clock.
    pub(crate) actions: Vec<(i64, RecordId)>,
    /// Instruction events in time order.
    pub(crate) instructions: Vec<(i64, RecordId)>,
}

impl EpisodeCtx {
    pub(crate) fn stream(&self, name: &str) -> Option<&StreamCtx> {
        self.streams.iter().find(|s| s.name == name)
    }

    /// The streams a recipe observes: those named, or every recorded one.
    pub(crate) fn observed(&self, names: &[String]) -> Vec<&StreamCtx> {
        self.streams
            .iter()
            .filter(|s| {
                s.extent.is_some()
                    && if names.is_empty() {
                        s.raw
                    } else {
                        names.contains(&s.name)
                    }
            })
            .collect()
    }
}

/// The time every one of `streams` has data for.
pub(crate) fn common_extent(streams: &[&StreamCtx]) -> Option<TimeRange> {
    let mut lo = i64::MIN;
    let mut hi = i64::MAX;
    for s in streams {
        let e = s.extent?;
        lo = lo.max(e.start_ns);
        hi = hi.min(e.end_ns);
    }
    (!streams.is_empty() && lo < hi).then_some(TimeRange {
        start_ns: lo,
        end_ns: hi,
    })
}

/// Anchors `from, from + stride, ...` not past `to`.
pub(crate) fn grid(from: i64, to: i64, stride: i64) -> Vec<i64> {
    if stride <= 0 {
        return Vec::new();
    }
    let mut anchors = Vec::new();
    let mut t = from;
    while t <= to {
        anchors.push(t);
        t += stride;
    }
    anchors
}

impl Snapshot {
    pub(crate) fn episode_contexts(&self, recipe: &Recipe) -> Result<Vec<EpisodeCtx>> {
        let data = self.timeline()?;
        let mut contexts = Vec::new();
        for (episode_id, episode) in &data.episodes {
            let mut streams = Vec::new();
            for (stream_id, stream) in data
                .streams
                .iter()
                .filter(|(_, s)| s.episode == *episode_id)
            {
                let transform = self.resolve_clock(stream.clock, episode.clock)?;
                let extent = transform.and_then(|t| {
                    let chunks = data.chunks.get(stream_id)?;
                    let start = chunks
                        .iter()
                        .map(|c| t.apply(c.chunk.interval.start_ns))
                        .min()?;
                    let end = chunks
                        .iter()
                        .map(|c| t.apply(c.chunk.interval.end_ns))
                        .max()?;
                    Some(TimeRange {
                        start_ns: start,
                        end_ns: end,
                    })
                });
                streams.push(StreamCtx {
                    id: *stream_id,
                    name: stream.name.clone(),
                    extent,
                    raw: stream.origin == StreamOrigin::Raw,
                });
            }
            let mut actions = Vec::new();
            for a in data.actions.iter().filter(|a| a.episode == *episode_id) {
                if let Some(t) = self.resolve_clock(a.action.clock, episode.clock)? {
                    actions.push((t.apply(a.action.interval.start_ns), a.id));
                }
            }
            actions.sort_unstable();
            let mut instructions = Vec::new();
            for e in data
                .events
                .iter()
                .filter(|e| e.episode == *episode_id && e.event.name == recipe.mm.instruction_event)
            {
                if let Some(t) = self.resolve_clock(e.event.clock, episode.clock)? {
                    instructions.push((t.apply(e.event.at.start_ns), e.id));
                }
            }
            instructions.sort_unstable();
            contexts.push(EpisodeCtx {
                id: *episode_id,
                family: ContentId::of(episode_id.to_string().as_bytes()),
                streams,
                actions,
                instructions,
            });
        }
        Ok(contexts)
    }
}
