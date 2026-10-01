// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Episodes as a reader sees them.

use crate::error::{Error, Result};
use crate::id::RecordId;
use crate::manifest::Snapshot;
use crate::model::{
    ActionSegment, Correspondence, Episode, EpisodeKind, Event, ModalitySchema, Stream,
    StreamChunk, TimeRange,
};

/// A chunk and the record that holds it.
#[derive(Debug, Clone, PartialEq)]
pub struct ChunkView {
    /// The chunk record.
    pub id: RecordId,
    /// What it says.
    pub chunk: StreamChunk,
}

/// A stream with its schema and chunks in time order.
#[derive(Debug, Clone, PartialEq)]
pub struct StreamView {
    /// The stream record.
    pub id: RecordId,
    /// The stream.
    pub stream: Stream,
    /// The modality it is a stream of.
    pub schema: ModalitySchema,
    /// Its chunks, by start time on the stream's clock.
    pub chunks: Vec<ChunkView>,
}

/// An event and the episode it belongs to.
#[derive(Debug, Clone, PartialEq)]
pub struct EventView {
    /// The event record.
    pub id: RecordId,
    /// The episode.
    pub episode: RecordId,
    /// What happened.
    pub event: Event,
}

/// An action and the episode it belongs to.
#[derive(Debug, Clone, PartialEq)]
pub struct ActionView {
    /// The action record.
    pub id: RecordId,
    /// The episode.
    pub episode: RecordId,
    /// What was done.
    pub action: ActionSegment,
}

/// A claimed relationship between spans, and its record.
#[derive(Debug, Clone, PartialEq)]
pub struct CorrespondenceView {
    /// The correspondence record.
    pub id: RecordId,
    /// The claim.
    pub correspondence: Correspondence,
}

/// One episode with everything that belongs to it.
#[derive(Debug, Clone, PartialEq)]
pub struct EpisodeView {
    /// The episode record.
    pub id: RecordId,
    /// The episode.
    pub episode: Episode,
    /// Its streams, recorded and derived.
    pub streams: Vec<StreamView>,
    /// Its events, in id order.
    pub events: Vec<EventView>,
    /// Its actions, in id order.
    pub actions: Vec<ActionView>,
}

impl Snapshot {
    /// An episode with its streams, events and actions.
    pub fn episode(&self, id: RecordId) -> Result<Option<EpisodeView>> {
        let data = self.timeline()?;
        let Some(episode) = data.episodes.get(&id) else {
            return Ok(None);
        };
        let mut streams = Vec::new();
        for (stream_id, stream) in data.streams.iter().filter(|(_, s)| s.episode == id) {
            let schema = data.schemas.get(&stream.modality).cloned().ok_or_else(|| {
                Error::corrupt(
                    format!("stream {stream_id}"),
                    "its modality schema is not in the snapshot",
                )
            })?;
            streams.push(StreamView {
                id: *stream_id,
                stream: stream.clone(),
                schema,
                chunks: data.chunks.get(stream_id).cloned().unwrap_or_default(),
            });
        }
        let mut events: Vec<_> = data
            .events
            .iter()
            .filter(|e| e.episode == id)
            .cloned()
            .collect();
        events.sort_by_key(|e| e.id);
        let mut actions: Vec<_> = data
            .actions
            .iter()
            .filter(|a| a.episode == id)
            .cloned()
            .collect();
        actions.sort_by_key(|a| a.id);
        Ok(Some(EpisodeView {
            id,
            episode: episode.clone(),
            streams,
            events,
            actions,
        }))
    }

    /// The episodes of one kind.
    pub fn episodes_of_kind(&self, kind: EpisodeKind) -> Result<Vec<RecordId>> {
        Ok(self
            .timeline()?
            .episodes
            .iter()
            .filter(|(_, e)| e.kind == kind)
            .map(|(id, _)| *id)
            .collect())
    }

    /// The bytes of a chunk.
    pub fn read_chunk(&self, chunk: &ChunkView) -> Result<Vec<u8>> {
        self.blob_store()?.get(&chunk.chunk.content)
    }

    /// The claims relating spans of `stream` that overlap `interval` (on the
    /// stream's clock), the most confident first.
    pub fn correspondences(
        &self,
        stream: RecordId,
        interval: TimeRange,
    ) -> Result<Vec<CorrespondenceView>> {
        let data = self.timeline()?;
        let mut found: Vec<_> = data
            .correspondences
            .iter()
            .filter(|c| {
                let k = &c.correspondence;
                [k.a, k.b]
                    .iter()
                    .any(|span| span.stream == stream && span.interval.overlaps(&interval))
            })
            .cloned()
            .collect();
        found.sort_by(|a, b| {
            b.correspondence
                .confidence
                .total_cmp(&a.correspondence.confidence)
                .then(a.id.cmp(&b.id))
        });
        Ok(found)
    }
}
