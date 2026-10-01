// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Windows of every stream over an interval, and questions around events.

use super::views::{ActionView, ChunkView};
use crate::error::{Error, Result};
use crate::id::RecordId;
use crate::manifest::Snapshot;
use crate::model::{EpisodeKind, ModalitySchema, Stream, TimeRange};

/// Which streams of an episode are wanted.
#[derive(Debug, Clone, Default)]
pub struct StreamFilter {
    names: Option<Vec<String>>,
    modalities: Option<Vec<String>>,
}

impl StreamFilter {
    /// Every stream.
    pub fn all() -> Self {
        Self::default()
    }

    /// Only streams with these names.
    pub fn names(mut self, names: &[&str]) -> Self {
        self.names = Some(names.iter().map(|n| (*n).to_owned()).collect());
        self
    }

    /// Only streams of these modalities.
    pub fn modalities(mut self, modalities: &[&str]) -> Self {
        self.modalities = Some(modalities.iter().map(|m| (*m).to_owned()).collect());
        self
    }

    pub(crate) fn matches(&self, stream: &Stream, schema: &ModalitySchema) -> bool {
        self.names.as_ref().is_none_or(|n| n.contains(&stream.name))
            && self
                .modalities
                .as_ref()
                .is_none_or(|m| m.contains(&schema.name))
    }
}

/// The chunks of one stream that overlap a window.
#[derive(Debug, Clone, PartialEq)]
pub struct StreamWindow {
    /// The stream.
    pub stream: RecordId,
    /// Its name.
    pub name: String,
    /// Its modality's name.
    pub modality: String,
    /// The overlapping chunks, in time order. Their intervals are on the
    /// stream's own clock.
    pub chunks: Vec<ChunkView>,
}

/// Which events to open a window around, and how wide.
#[derive(Debug, Clone)]
pub struct EventQuery {
    name: String,
    before_ns: i64,
    after_ns: i64,
    min_value: Option<f64>,
    streams: StreamFilter,
    kinds: Option<Vec<EpisodeKind>>,
}

impl EventQuery {
    /// Events with this name.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.into(),
            before_ns: 0,
            after_ns: 0,
            min_value: None,
            streams: StreamFilter::all(),
            kinds: None,
        }
    }

    /// How much time before the event to include.
    pub fn before_ns(mut self, ns: i64) -> Self {
        self.before_ns = ns;
        self
    }

    /// How much time after the event to include.
    pub fn after_ns(mut self, ns: i64) -> Self {
        self.after_ns = ns;
        self
    }

    /// Only events whose value is at least this.
    pub fn min_value(mut self, value: f64) -> Self {
        self.min_value = Some(value);
        self
    }

    /// Only these streams.
    pub fn streams(mut self, filter: StreamFilter) -> Self {
        self.streams = filter;
        self
    }

    /// Only events in these kinds of episode.
    pub fn kinds(mut self, kinds: &[EpisodeKind]) -> Self {
        self.kinds = Some(kinds.to_vec());
        self
    }
}

/// Every wanted stream around one event.
#[derive(Debug, Clone, PartialEq)]
pub struct EventWindow {
    /// The episode.
    pub episode: RecordId,
    /// The event.
    pub event: super::views::EventView,
    /// The window, on the episode's clock.
    pub interval: TimeRange,
    /// The streams in it.
    pub streams: Vec<StreamWindow>,
}

impl Snapshot {
    /// The chunks of each wanted stream of an episode that overlap
    /// `interval`, which is on the episode's clock. Streams on other clocks
    /// are found where they really happened.
    pub fn window(
        &self,
        episode: RecordId,
        interval: TimeRange,
        filter: &StreamFilter,
    ) -> Result<Vec<StreamWindow>> {
        let data = self.timeline()?;
        let episode_clock = data
            .episodes
            .get(&episode)
            .ok_or_else(|| Error::NotFound {
                what: format!("episode {episode}"),
            })?
            .clock;
        let mut windows = Vec::new();
        for (id, stream) in data.streams.iter().filter(|(_, s)| s.episode == episode) {
            let schema = data.schemas.get(&stream.modality).ok_or_else(|| {
                Error::corrupt(
                    format!("stream {id}"),
                    "its modality schema is not in the snapshot",
                )
            })?;
            if !filter.matches(stream, schema) {
                continue;
            }
            let transform = self
                .resolve_clock(stream.clock, episode_clock)?
                .ok_or_else(|| Error::NotFound {
                    what: format!(
                        "a mapping from the clock of stream {id} to the clock of episode {episode}"
                    ),
                })?;
            let chunks = data
                .chunks
                .get(id)
                .into_iter()
                .flatten()
                .filter(|c| {
                    let span = TimeRange {
                        start_ns: transform.apply(c.chunk.interval.start_ns),
                        end_ns: transform.apply(c.chunk.interval.end_ns),
                    };
                    span.overlaps(&interval)
                })
                .cloned()
                .collect();
            windows.push(StreamWindow {
                stream: *id,
                name: stream.name.clone(),
                modality: schema.name.clone(),
                chunks,
            });
        }
        Ok(windows)
    }

    /// The time of an event on its episode's clock.
    fn event_time(&self, event: &super::views::EventView) -> Result<i64> {
        let data = self.timeline()?;
        let clock = data
            .episodes
            .get(&event.episode)
            .ok_or_else(|| Error::NotFound {
                what: format!("episode {}", event.episode),
            })?
            .clock;
        let transform = self
            .resolve_clock(event.event.clock, clock)?
            .ok_or_else(|| Error::NotFound {
                what: format!(
                    "a mapping from the clock of event {} to its episode's",
                    event.id
                ),
            })?;
        Ok(transform.apply(event.event.at.start_ns))
    }

    /// Every wanted stream around each matching event, in episode and time
    /// order: "all the sensory streams two seconds either side of a
    /// successful contact" is one call.
    pub fn windows_around(&self, query: &EventQuery) -> Result<Vec<EventWindow>> {
        let data = self.timeline()?;
        let mut found = Vec::new();
        for event in data.events.iter().filter(|e| e.event.name == query.name) {
            if query
                .min_value
                .is_some_and(|min| event.event.value.is_none_or(|v| v < min))
            {
                continue;
            }
            let Some(episode) = data.episodes.get(&event.episode) else {
                continue;
            };
            if query
                .kinds
                .as_ref()
                .is_some_and(|k| !k.contains(&episode.kind))
            {
                continue;
            }
            let at = self.event_time(event)?;
            let interval = if query.before_ns + query.after_ns == 0 {
                TimeRange::instant(at)
            } else {
                TimeRange::new(at - query.before_ns, at + query.after_ns)?
            };
            let streams = self.window(event.episode, interval, &query.streams)?;
            found.push((
                event.episode,
                at,
                EventWindow {
                    episode: event.episode,
                    event: event.clone(),
                    interval,
                    streams,
                },
            ));
        }
        found.sort_by_key(|(episode, at, w)| (*episode, *at, w.event.id));
        Ok(found.into_iter().map(|(_, _, w)| w).collect())
    }

    /// The actions that began no more than `within_ns` before an event of
    /// this name in their episode: what was done shortly before something
    /// happened.
    pub fn actions_followed_by(&self, event_name: &str, within_ns: i64) -> Result<Vec<ActionView>> {
        let data = self.timeline()?;
        let mut found: Vec<(RecordId, i64, ActionView)> = Vec::new();
        for event in data.events.iter().filter(|e| e.event.name == event_name) {
            let at = self.event_time(event)?;
            let Some(episode) = data.episodes.get(&event.episode) else {
                continue;
            };
            for action in data.actions.iter().filter(|a| a.episode == event.episode) {
                let transform = self
                    .resolve_clock(action.action.clock, episode.clock)?
                    .ok_or_else(|| Error::NotFound {
                        what: format!(
                            "a mapping from the clock of action {} to its episode's",
                            action.id
                        ),
                    })?;
                let start = transform.apply(action.action.interval.start_ns);
                if at >= start
                    && at - start <= within_ns
                    && !found.iter().any(|(_, _, a)| a.id == action.id)
                {
                    found.push((action.episode, start, action.clone()));
                }
            }
        }
        found.sort_by_key(|(episode, start, a)| (*episode, *start, a.id));
        Ok(found.into_iter().map(|(_, _, a)| a).collect())
    }
}
