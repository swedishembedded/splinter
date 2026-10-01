// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Recording modality-neutral experience: episodes of streams, events and
//! actions on explicit clocks.

use super::collector::Collector;
use crate::blob::BlobStats;
use crate::error::{Error, Result};
use crate::id::{ContentId, RecordId};
use crate::model::{
    ActionSegment, Body, ClockDomain, ClockMapping, Correspondence, Derivation, Edge, Episode,
    EpisodeKind, Event, Rel, Stream, StreamChunk, StreamOrigin, StreamSpec, TimeRange,
};

impl Collector {
    /// What the blob store has written, to see that repeated content cost
    /// nothing.
    pub fn blob_stats(&self) -> BlobStats {
        self.writer.blob_stats()
    }

    /// Records a clock domain once and returns the id streams refer to.
    pub fn register_clock(&mut self, clock: &ClockDomain) -> Result<ContentId> {
        self.entity(Body::ClockDomain(clock.clone()))?;
        clock.id()
    }

    /// Records how two clocks relate. More than one mapping may be recorded
    /// for a pair; readers use the one with the least uncertainty.
    pub fn map_clock(&mut self, mapping: ClockMapping) -> Result<RecordId> {
        self.record(Body::ClockMapping(mapping))
    }

    /// Records a record that belongs to an episode.
    fn in_episode(&mut self, episode: RecordId, body: Body) -> Result<RecordId> {
        let mut record = self.make(body);
        record.attempt = Some(episode);
        self.writer.push(record)
    }

    /// Starts an episode: a stretch of experience whose streams, events and
    /// actions belong together. Its clock is the one queries are made on.
    pub fn start_episode(
        &mut self,
        kind: EpisodeKind,
        label: &str,
        clock: ContentId,
    ) -> Result<RecordId> {
        let id = self.writer.next_id();
        let mut record = crate::model::Record::new(
            id,
            self.writer.database().clock().now_ns(),
            Body::Episode(Episode {
                kind,
                label: label.into(),
                clock,
            }),
        );
        record.attempt = Some(id);
        self.writer.push(record)
    }

    fn stream_body(
        &mut self,
        episode: RecordId,
        spec: &StreamSpec,
        origin: StreamOrigin,
    ) -> Result<Body> {
        self.entity(Body::ModalitySchema(spec.schema.clone()))?;
        let calibration = spec
            .calibration
            .as_deref()
            .map(|bytes| self.writer.blobs().put(bytes))
            .transpose()?;
        Ok(Body::Stream(Stream {
            episode,
            name: spec.name.clone(),
            modality: spec.schema.id()?,
            clock: spec.clock,
            rate_hz: spec.rate_hz,
            origin,
            calibration,
        }))
    }

    /// Adds a recorded stream to an episode.
    pub fn add_stream(&mut self, episode: RecordId, spec: &StreamSpec) -> Result<RecordId> {
        let body = self.stream_body(episode, spec, StreamOrigin::Raw)?;
        self.in_episode(episode, body)
    }

    /// Adds samples to a stream: `samples` of them covering `interval` of the
    /// stream's clock. Identical bytes already stored are not stored again.
    pub fn add_chunk(
        &mut self,
        episode: RecordId,
        stream: RecordId,
        interval: TimeRange,
        samples: u64,
        bytes: &[u8],
    ) -> Result<RecordId> {
        let content = self.writer.blobs().put(bytes)?;
        self.in_episode(
            episode,
            Body::StreamChunk(StreamChunk {
                stream,
                interval,
                samples,
                content,
            }),
        )
    }

    /// Records something that happened.
    pub fn add_event(&mut self, episode: RecordId, event: Event) -> Result<RecordId> {
        self.in_episode(episode, Body::Event(event))
    }

    /// Records an action over an interval.
    pub fn add_action(&mut self, episode: RecordId, action: ActionSegment) -> Result<RecordId> {
        self.in_episode(episode, Body::ActionSegment(action))
    }

    /// Records a claimed relationship between two spans. A better claim is
    /// another correspondence; neither stream is touched.
    pub fn correspond(
        &mut self,
        episode: RecordId,
        correspondence: Correspondence,
    ) -> Result<RecordId> {
        if !(0.0..=1.0).contains(&correspondence.confidence) {
            return Err(Error::invalid(
                "correspondence",
                "confidence must be between 0 and 1",
            ));
        }
        self.in_episode(episode, Body::Correspondence(correspondence))
    }

    /// Records a stream computed from other streams (a transcript, an
    /// embedding, a spectrogram) with the derivation that made it. The
    /// sources are not changed and stay canonical.
    pub fn derived_stream(
        &mut self,
        episode: RecordId,
        derivation: Derivation,
        spec: &StreamSpec,
        chunks: &[(TimeRange, u64, &[u8])],
    ) -> Result<RecordId> {
        let inputs = derivation.inputs.clone();
        let derivation_id = self.record(Body::Derivation(derivation))?;
        let body = self.stream_body(episode, spec, StreamOrigin::Derived)?;
        let stream = self.in_episode(episode, body)?;
        self.writer
            .link(Edge::new(stream, Rel::ProducedBy, derivation_id))?;
        for input in inputs {
            self.writer
                .link(Edge::new(stream, Rel::DerivedFrom, input))?;
        }
        for (interval, samples, bytes) in chunks {
            self.add_chunk(episode, stream, *interval, *samples, bytes)?;
        }
        Ok(stream)
    }
}
