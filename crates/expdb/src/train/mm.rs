// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Reading the windows and actions a multimodal sample points at.

use serde::{Serialize, Serializer};

use super::plan::DataRef;
use crate::error::{Error, Result};
use crate::manifest::Snapshot;
use crate::model::{Content, TimeRange};

fn hex<S: Serializer>(bytes: &[u8], serializer: S) -> std::result::Result<S::Ok, S::Error> {
    let mut text = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        text.push_str(&format!("{byte:02x}"));
    }
    serializer.serialize_str(&text)
}

/// A window of a stream, an action or an event, read.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MaterializedSpan {
    /// The stream, action or event name.
    pub label: String,
    /// The modality of the bytes.
    pub modality: String,
    /// The time covered, on the episode's clock. For a window of a stream that
    /// cannot be cut to a sample it is the whole of the chunks that overlap.
    pub interval: TimeRange,
    /// How many samples the bytes hold.
    pub count: u64,
    /// Whether the interval was cut to the sample (raw streams) rather than
    /// to whole chunks (compressed streams).
    pub exact: bool,
    /// The bytes, as hex in an export.
    #[serde(serialize_with = "hex")]
    pub bytes: Vec<u8>,
}

/// A negative example, read.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MaterializedNegative {
    /// What is not a match.
    pub span: MaterializedSpan,
    /// Whether it was taken from nearby in time.
    pub hard: bool,
}

impl Snapshot {
    /// Reads what a window, action or event reference points at.
    pub fn read_span(&self, data: &DataRef) -> Result<MaterializedSpan> {
        let timeline = self.timeline()?;
        match data {
            DataRef::Window { stream, interval } => {
                let s = timeline
                    .streams
                    .get(stream)
                    .ok_or_else(|| Error::NotFound {
                        what: format!("stream {stream}"),
                    })?;
                let schema = timeline.schemas.get(&s.modality).ok_or_else(|| {
                    Error::corrupt(
                        format!("stream {stream}"),
                        "its modality schema is not in the snapshot",
                    )
                })?;
                if schema.sample_bytes().is_some() && s.rate_hz.is_some() {
                    let cut = self.read_samples(*stream, *interval)?;
                    return Ok(MaterializedSpan {
                        label: s.name.clone(),
                        modality: schema.name.clone(),
                        interval: cut.interval,
                        count: cut.count,
                        exact: true,
                        bytes: cut.bytes,
                    });
                }
                let episode = timeline
                    .episodes
                    .get(&s.episode)
                    .ok_or_else(|| Error::NotFound {
                        what: format!("episode {}", s.episode),
                    })?;
                let to_episode =
                    self.resolve_clock(s.clock, episode.clock)?
                        .ok_or_else(|| Error::NotFound {
                            what: format!(
                                "a mapping from the clock of stream {stream} to its episode's"
                            ),
                        })?;
                let (mut bytes, mut count, mut covered) = (Vec::new(), 0u64, None::<(i64, i64)>);
                for chunk in timeline.chunks.get(stream).into_iter().flatten() {
                    let span = TimeRange {
                        start_ns: to_episode.apply(chunk.chunk.interval.start_ns),
                        end_ns: to_episode.apply(chunk.chunk.interval.end_ns),
                    };
                    if span.overlaps(interval) {
                        bytes.extend(self.read_chunk(chunk)?);
                        count += chunk.chunk.samples;
                        covered = Some(covered.map_or((span.start_ns, span.end_ns), |(a, b)| {
                            (a.min(span.start_ns), b.max(span.end_ns))
                        }));
                    }
                }
                let (from, to) = covered.unwrap_or((interval.start_ns, interval.start_ns));
                Ok(MaterializedSpan {
                    label: s.name.clone(),
                    modality: schema.name.clone(),
                    interval: TimeRange {
                        start_ns: from,
                        end_ns: to,
                    },
                    count,
                    exact: false,
                    bytes,
                })
            }
            DataRef::ActionSegment { record } => {
                let action = timeline
                    .actions
                    .iter()
                    .find(|a| a.id == *record)
                    .ok_or_else(|| Error::NotFound {
                        what: format!("action {record}"),
                    })?;
                let bytes = match &action.action.payload {
                    Content::Text { text } => text.clone().into_bytes(),
                    Content::Blob { blob } => self.blob_store()?.get(blob)?,
                };
                let modality = action
                    .action
                    .modality
                    .and_then(|m| timeline.schemas.get(&m))
                    .map_or_else(|| "action".to_owned(), |m| m.name.clone());
                Ok(MaterializedSpan {
                    label: action.action.name.clone(),
                    modality,
                    interval: action.action.interval,
                    count: 1,
                    exact: true,
                    bytes,
                })
            }
            DataRef::EventPayload { record } => {
                let event = timeline
                    .events
                    .iter()
                    .find(|e| e.id == *record)
                    .ok_or_else(|| Error::NotFound {
                        what: format!("event {record}"),
                    })?;
                let text = match event.event.payload.get("text").and_then(|t| t.as_str()) {
                    Some(text) => text.to_owned(),
                    None => event.event.payload.to_string(),
                };
                Ok(MaterializedSpan {
                    label: event.event.name.clone(),
                    modality: "text/utf8".into(),
                    interval: event.event.at,
                    count: 1,
                    exact: true,
                    bytes: text.into_bytes(),
                })
            }
            other => Err(Error::invalid(
                "data reference",
                format!("{other:?} is text, not a span"),
            )),
        }
    }
}
