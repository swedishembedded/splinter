// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Modality-neutral experience: streams of any kind of sample on explicit
//! clocks, grouped into episodes with events, actions and correspondences.
//!
//! Nothing here assumes text. A modality is a registered schema, time is an
//! interval on a clock domain, and a stream's samples live in the blob store
//! as chunks. Raw streams are canonical; a transcript, an embedding or a
//! label is another stream (or record) that names the derivation that made it.

use serde::{Deserialize, Serialize};

use super::entities::Content;
use super::interpret::Epistemic;
use crate::blob::BlobRef;
use crate::error::{Error, Result};
use crate::id::{ContentId, RecordId};

fn content_id_of<T: Serialize>(what: &'static str, value: &T) -> Result<ContentId> {
    let bytes = serde_json::to_vec(value).map_err(|source| Error::Encode { what, source })?;
    Ok(ContentId::of(&bytes))
}

/// A half-open span `[start_ns, end_ns)` of time on some clock, in
/// nanoseconds. An instant is a span of length zero, so a still image, a
/// single reading and a video are all the same kind of thing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TimeRange {
    /// The first nanosecond.
    pub start_ns: i64,
    /// The first nanosecond after the span.
    pub end_ns: i64,
}

impl TimeRange {
    /// A span; the end may not precede the start.
    pub fn new(start_ns: i64, end_ns: i64) -> Result<Self> {
        if end_ns < start_ns {
            return Err(Error::invalid(
                "time range",
                format!("it ends ({end_ns}) before it starts ({start_ns})"),
            ));
        }
        Ok(Self { start_ns, end_ns })
    }

    /// A single moment.
    pub fn instant(at_ns: i64) -> Self {
        Self {
            start_ns: at_ns,
            end_ns: at_ns,
        }
    }

    /// The length in nanoseconds.
    pub fn len_ns(&self) -> i64 {
        self.end_ns - self.start_ns
    }

    /// Whether a moment falls inside the span.
    pub fn contains(&self, at_ns: i64) -> bool {
        self.start_ns <= at_ns && at_ns < self.end_ns
    }

    /// Whether two spans share any time. An instant overlaps a span that
    /// contains it.
    pub fn overlaps(&self, other: &TimeRange) -> bool {
        match (self.len_ns() == 0, other.len_ns() == 0) {
            (true, true) => self.start_ns == other.start_ns,
            (true, false) => other.contains(self.start_ns),
            (false, true) => self.contains(other.start_ns),
            (false, false) => self.start_ns < other.end_ns && other.start_ns < self.end_ns,
        }
    }

    /// The span moved by `delta_ns`.
    pub fn shift(&self, delta_ns: i64) -> Self {
        Self {
            start_ns: self.start_ns + delta_ns,
            end_ns: self.end_ns + delta_ns,
        }
    }
}

/// How samples of a modality are spread over time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimeSemantics {
    /// Single readings at their own moments.
    Instant,
    /// Spans, such as an utterance.
    Interval,
    /// Evenly spaced at the stream's rate.
    Regular,
}

/// A registered kind of sample: what one sample is, how it is encoded and how
/// it sits in time. The store never matches on a modality, so a new sensor
/// needs a new schema and nothing else.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModalitySchema {
    /// The semantic name, such as `audio/pcm`.
    pub name: String,
    /// The element type of a sample: `f32`, `i16`, `u8` and so on.
    pub physical_type: String,
    /// The shape of one sample; empty for a scalar.
    pub shape: Vec<u64>,
    /// How samples sit in time.
    pub time: TimeSemantics,
    /// How chunks are encoded: `raw` for fixed-size samples, else a codec.
    pub codec: String,
    /// Anything else that describes the modality.
    pub metadata: serde_json::Value,
}

impl ModalitySchema {
    /// A schema of your own.
    pub fn custom(name: &str, physical_type: &str, shape: &[u64], codec: &str) -> Self {
        Self {
            name: name.into(),
            physical_type: physical_type.into(),
            shape: shape.to_vec(),
            time: TimeSemantics::Regular,
            codec: codec.into(),
            metadata: serde_json::Value::Null,
        }
    }

    /// The same modality with another sample shape, such as four microphone
    /// channels instead of one.
    pub fn with_shape(mut self, shape: &[u64]) -> Self {
        self.shape = shape.to_vec();
        self
    }

    /// A schema that ships with the crate.
    pub fn builtin(name: &str) -> Option<Self> {
        let regular = |name: &str, ty: &str, shape: &[u64]| Self::custom(name, ty, shape, "raw");
        Some(match name {
            "text/utf8" => Self {
                time: TimeSemantics::Interval,
                ..Self::custom(name, "u8", &[], "utf8")
            },
            "audio/pcm" => regular(name, "f32", &[]),
            "video/rgb" => Self::custom(name, "u8", &[], "h264"),
            "image/rgb" => Self {
                time: TimeSemantics::Instant,
                ..Self::custom(name, "u8", &[], "png")
            },
            "sensor/imu" => regular(name, "f32", &[6]),
            "sensor/depth" => regular(name, "f32", &[]),
            "robot/joint_state" => regular(name, "f32", &[7]),
            "robot/action" => regular(name, "f32", &[7]),
            "geometry/pointcloud" => Self {
                time: TimeSemantics::Instant,
                ..Self::custom(name, "f32", &[3], "raw")
            },
            _ => return None,
        })
    }

    /// The content id every stream of this modality refers to.
    pub fn id(&self) -> Result<ContentId> {
        content_id_of("modality schema", self)
    }

    /// Bytes in one sample, when samples are fixed-size and stored raw.
    pub fn sample_bytes(&self) -> Option<usize> {
        let element = match self.physical_type.as_str() {
            "u8" | "i8" => 1,
            "u16" | "i16" => 2,
            "u32" | "i32" | "f32" => 4,
            "u64" | "i64" | "f64" => 8,
            _ => return None,
        };
        if self.codec != "raw" {
            return None;
        }
        self.shape
            .iter()
            .try_fold(element as u64, |acc, dim| acc.checked_mul(*dim))
            .and_then(|n| usize::try_from(n).ok())
    }
}

/// `to_ref + slope * (at - from_ref)`, in nanoseconds.
///
/// The part of the conversion that is exact (the offset, and the delta
/// itself) is done in 128-bit integers; only the small correction for a slope
/// that is not one goes through a float, so epoch-scale timestamps keep every
/// nanosecond. Results beyond the range of `i64` saturate instead of wrapping.
pub(crate) fn convert_ns(from_ref: i64, to_ref: i64, slope: f64, at: i64) -> i64 {
    let delta = i128::from(at) - i128::from(from_ref);
    let correction = if slope == 1.0 {
        0
    } else {
        ((slope - 1.0) * delta as f64).round() as i128
    };
    let value = i128::from(to_ref) + delta + correction;
    value.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64
}

/// A source of timestamps. Cameras, microphones, MCUs and hosts each keep
/// their own.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClockDomain {
    /// A short name, such as `imu-mcu`.
    pub name: String,
    /// What keeps this clock.
    pub description: String,
}

impl ClockDomain {
    /// The content id streams and mappings refer to.
    pub fn id(&self) -> Result<ContentId> {
        content_id_of("clock domain", self)
    }
}

/// How one clock reads against another:
/// `destination = dst_ref + slope * (source - src_ref)`, known to within an
/// uncertainty. Measured against a reference near the data, so a slope close
/// to one costs nothing in precision.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ClockMapping {
    /// The clock the mapping reads from.
    pub source: ContentId,
    /// The clock it maps to.
    pub destination: ContentId,
    /// A source time at which the clocks were compared.
    pub src_ref_ns: i64,
    /// The destination time at that moment.
    pub dst_ref_ns: i64,
    /// Destination seconds per source second.
    pub slope: f64,
    /// How far the mapping may be off, in nanoseconds.
    pub uncertainty_ns: u64,
}

impl ClockMapping {
    /// A source time on the destination clock.
    pub fn to_destination(&self, source_ns: i64) -> i64 {
        convert_ns(self.src_ref_ns, self.dst_ref_ns, self.slope, source_ns)
    }

    /// A destination time on the source clock.
    pub fn to_source(&self, destination_ns: i64) -> i64 {
        convert_ns(
            self.dst_ref_ns,
            self.src_ref_ns,
            1.0 / self.slope,
            destination_ns,
        )
    }
}

/// What kind of experience an episode is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EpisodeKind {
    /// Watching the world: no actions needed.
    Observational,
    /// Acting in it and seeing the result.
    Interactive,
    /// Intervening deliberately to answer a question.
    Experimental,
}

/// One synchronised stretch of experience: all of its streams, actions and
/// events belong together, and their synchronisation is part of the data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Episode {
    /// What kind of experience it is.
    pub kind: EpisodeKind,
    /// A short description.
    pub label: String,
    /// The clock its times are expressed on, and queries are made on.
    pub clock: ContentId,
}

/// Whether a stream was recorded or computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamOrigin {
    /// Recorded from the world.
    Raw,
    /// Computed from other streams by a derivation.
    Derived,
}

/// A sequence of samples of one modality on one clock.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Stream {
    /// The episode it belongs to.
    pub episode: RecordId,
    /// A name, such as `camera-left`.
    pub name: String,
    /// The modality schema's content id.
    pub modality: ContentId,
    /// The clock its chunk times are on.
    pub clock: ContentId,
    /// Samples per second, for regular streams.
    pub rate_hz: Option<f64>,
    /// Recorded or derived.
    pub origin: StreamOrigin,
    /// Calibration, such as a microphone array's geometry.
    pub calibration: Option<BlobRef>,
}

/// Describes a stream to be created.
#[derive(Debug, Clone)]
pub struct StreamSpec {
    pub(crate) name: String,
    pub(crate) schema: ModalitySchema,
    pub(crate) clock: ContentId,
    pub(crate) rate_hz: Option<f64>,
    pub(crate) calibration: Option<Vec<u8>>,
}

impl StreamSpec {
    /// A stream called `name` of a modality, on a clock.
    pub fn new(name: &str, schema: &ModalitySchema, clock: ContentId) -> Self {
        Self {
            name: name.into(),
            schema: schema.clone(),
            clock,
            rate_hz: None,
            calibration: None,
        }
    }

    /// Its sample rate.
    pub fn rate(mut self, hz: f64) -> Self {
        self.rate_hz = Some(hz);
        self
    }

    /// Calibration bytes, stored once in the blob store.
    pub fn calibration(mut self, bytes: &[u8]) -> Self {
        self.calibration = Some(bytes.to_vec());
        self
    }
}

/// A piece of a stream: samples over an interval of the stream's clock.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StreamChunk {
    /// The stream.
    pub stream: RecordId,
    /// The time covered, on the stream's clock.
    pub interval: TimeRange,
    /// How many samples it holds.
    pub samples: u64,
    /// Where the samples are.
    pub content: BlobRef,
}

/// Something that happened at a moment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Event {
    /// What happened: `contact`, `instruction`, `reward`.
    pub name: String,
    /// When, on `clock`.
    pub at: TimeRange,
    /// The clock the time is on.
    pub clock: ContentId,
    /// A magnitude, such as a reward or a success flag, if there is one.
    pub value: Option<f64>,
    /// Anything else about it.
    pub payload: serde_json::Value,
    /// Observed, detected by a model, or judged.
    pub epistemic: Epistemic,
}

impl Event {
    /// An observed event at one moment.
    pub fn at(name: &str, at_ns: i64, clock: ContentId) -> Self {
        Self {
            name: name.into(),
            at: TimeRange::instant(at_ns),
            clock,
            value: None,
            payload: serde_json::Value::Null,
            epistemic: Epistemic::Fact,
        }
    }

    /// Its magnitude.
    pub fn value(mut self, value: f64) -> Self {
        self.value = Some(value);
        self
    }

    /// Its details.
    pub fn payload(mut self, payload: serde_json::Value) -> Self {
        self.payload = payload;
        self
    }

    /// How far it may be believed.
    pub fn epistemic(mut self, epistemic: Epistemic) -> Self {
        self.epistemic = epistemic;
        self
    }
}

/// What sort of thing an action is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionKind {
    /// A choice among a few: click, answer, call.
    Discrete,
    /// A continuous signal: torques, steering, a mouse path.
    Continuous,
    /// A structured edit: a query, a patch, a CAD operation.
    Structured,
    /// Communication: speech, text, gesture.
    Communicative,
    /// A change made to the world: heat a sample, inject a chemical.
    Environmental,
}

/// An action over an interval, with whatever payload describes it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionSegment {
    /// What sort of action.
    pub kind: ActionKind,
    /// Its name.
    pub name: String,
    /// The modality of its payload, if it has one.
    pub modality: Option<ContentId>,
    /// When it was carried out, on `clock`.
    pub interval: TimeRange,
    /// The clock the interval is on.
    pub clock: ContentId,
    /// What it was: parameters, a trajectory, text.
    pub payload: Content,
}

impl ActionSegment {
    /// An action with no payload yet.
    pub fn new(kind: ActionKind, name: &str, interval: TimeRange, clock: ContentId) -> Self {
        Self {
            kind,
            name: name.into(),
            modality: None,
            interval,
            clock,
            payload: Content::text(""),
        }
    }

    /// Its payload.
    pub fn payload(mut self, payload: Content) -> Self {
        self.payload = payload;
        self
    }

    /// The modality of its payload.
    pub fn modality(mut self, modality: ContentId) -> Self {
        self.modality = Some(modality);
        self
    }
}

/// How two spans of streams relate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CorrespondenceRelation {
    /// They happened together.
    SynchronizedWith,
    /// The first describes the second.
    Describes,
    /// The first was caused by the second.
    CausedBy,
    /// The first is evidence for the second.
    EvidenceFor,
    /// They are the same event seen two ways.
    SameEventAs,
}

/// A span of one stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SpanRef {
    /// The stream.
    pub stream: RecordId,
    /// The span, on the stream's clock.
    pub interval: TimeRange,
}

/// A claimed relationship between two spans. It carries its confidence and,
/// when a model proposed it, the derivation, so a better alignment is another
/// correspondence and the streams are never overwritten.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Correspondence {
    /// The first span.
    pub a: SpanRef,
    /// The second span.
    pub b: SpanRef,
    /// How they relate.
    pub relation: CorrespondenceRelation,
    /// How sure the claim is, from 0 to 1.
    pub confidence: f64,
    /// How the claim was made, if a procedure made it.
    pub derivation: Option<RecordId>,
}
