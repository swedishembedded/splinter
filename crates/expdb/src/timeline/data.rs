// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Everything about episodes in a snapshot, parsed once.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use super::views::{ActionView, ChunkView, CorrespondenceView, EventView};
use crate::error::Result;
use crate::id::{ContentId, RecordId};
use crate::index::locked;
use crate::manifest::Snapshot;
use crate::model::{Body, ClockMapping, Episode, ModalitySchema, RecordKind, Stream};

/// The episode-level records of a snapshot.
#[derive(Default)]
pub(crate) struct TimelineData {
    pub(crate) episodes: BTreeMap<RecordId, Episode>,
    pub(crate) streams: BTreeMap<RecordId, Stream>,
    pub(crate) chunks: HashMap<RecordId, Vec<ChunkView>>,
    pub(crate) events: Vec<EventView>,
    pub(crate) actions: Vec<ActionView>,
    pub(crate) mappings: Vec<ClockMapping>,
    pub(crate) correspondences: Vec<CorrespondenceView>,
    pub(crate) schemas: HashMap<ContentId, ModalitySchema>,
    /// Where each episode's streams, events and actions are, so that reading
    /// one episode costs its own records and not the snapshot's: a cohort of
    /// n episodes is read back in n steps, not n squared.
    pub(crate) by_episode: HashMap<RecordId, EpisodeParts>,
}

/// The records of one episode: its streams, and the positions of its events
/// and actions in [`TimelineData::events`] and [`TimelineData::actions`].
#[derive(Default)]
pub(crate) struct EpisodeParts {
    /// Stream ids, ascending.
    pub(crate) streams: Vec<RecordId>,
    pub(crate) events: Vec<usize>,
    pub(crate) actions: Vec<usize>,
}

impl Snapshot {
    pub(crate) fn timeline(&self) -> Result<Arc<TimelineData>> {
        if let Some(found) = locked(&self.cache().timeline).as_ref() {
            return Ok(Arc::clone(found));
        }
        let index = self.index()?;
        let mut data = TimelineData::default();
        for kind in [
            RecordKind::Episode,
            RecordKind::Stream,
            RecordKind::StreamChunk,
            RecordKind::Event,
            RecordKind::ActionSegment,
            RecordKind::ClockMapping,
            RecordKind::Correspondence,
            RecordKind::ModalitySchema,
        ] {
            for id in index.by_kind(kind) {
                let Some(record) = self.get(id)? else {
                    continue;
                };
                let episode = record.attempt;
                match record.body {
                    Body::Episode(e) => {
                        data.episodes.insert(id, e);
                    }
                    Body::Stream(s) => {
                        data.streams.insert(id, s);
                    }
                    Body::StreamChunk(chunk) => data
                        .chunks
                        .entry(chunk.stream)
                        .or_default()
                        .push(ChunkView { id, chunk }),
                    Body::Event(event) => {
                        if let Some(episode) = episode {
                            data.events.push(EventView { id, episode, event });
                        }
                    }
                    Body::ActionSegment(action) => {
                        if let Some(episode) = episode {
                            data.actions.push(ActionView {
                                id,
                                episode,
                                action,
                            });
                        }
                    }
                    Body::ClockMapping(m) => data.mappings.push(m),
                    Body::Correspondence(correspondence) => {
                        data.correspondences
                            .push(CorrespondenceView { id, correspondence });
                    }
                    Body::ModalitySchema(schema) => {
                        data.schemas.insert(schema.id()?, schema);
                    }
                    _ => {}
                }
            }
        }
        for (id, stream) in &data.streams {
            data.by_episode
                .entry(stream.episode)
                .or_default()
                .streams
                .push(*id);
        }
        for (at, event) in data.events.iter().enumerate() {
            data.by_episode
                .entry(event.episode)
                .or_default()
                .events
                .push(at);
        }
        for (at, action) in data.actions.iter().enumerate() {
            data.by_episode
                .entry(action.episode)
                .or_default()
                .actions
                .push(at);
        }
        for chunks in data.chunks.values_mut() {
            chunks.sort_by_key(|c| (c.chunk.interval.start_ns, c.id));
        }
        let data = Arc::new(data);
        *locked(&self.cache().timeline) = Some(Arc::clone(&data));
        Ok(data)
    }
}
