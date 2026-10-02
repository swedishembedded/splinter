// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Cutting exact samples out of raw fixed-size streams.

use crate::error::{Error, Result};
use crate::id::RecordId;
use crate::manifest::Snapshot;
use crate::model::TimeRange;

/// Samples cut from a stream.
#[derive(Debug, Clone, PartialEq)]
pub struct Samples {
    /// The span the samples actually cover, on the episode's clock. It is the
    /// requested span rounded outwards to whole samples.
    pub interval: TimeRange,
    /// How many samples.
    pub count: u64,
    /// Bytes in each sample.
    pub sample_bytes: usize,
    /// The samples, back to back.
    pub bytes: Vec<u8>,
}

/// `x`, rounded up unless it is within rounding noise of a whole number.
fn ceil_stable(x: f64) -> i64 {
    let r = x.round();
    if (x - r).abs() < 1e-6 {
        r as i64
    } else {
        x.ceil() as i64
    }
}

impl Snapshot {
    /// The samples of a regular, raw stream between two times on its
    /// episode's clock, cut exactly even across chunk boundaries. A
    /// compressed stream cannot be cut to a sample and is refused; its chunks
    /// are still available through [`Snapshot::window`].
    pub fn read_samples(&self, stream: RecordId, interval: TimeRange) -> Result<Samples> {
        let data = self.timeline()?;
        let s = data.streams.get(&stream).ok_or_else(|| Error::NotFound {
            what: format!("stream {stream}"),
        })?;
        let schema = data.schemas.get(&s.modality).ok_or_else(|| {
            Error::corrupt(
                format!("stream {stream}"),
                "its modality schema is not in the snapshot",
            )
        })?;
        let sample_bytes = schema.sample_bytes().ok_or_else(|| {
            Error::invalid(
                "stream",
                format!(
                    "`{}` is stored as {} and cannot be cut to samples",
                    s.name, schema.codec
                ),
            )
        })?;
        let rate = s.rate_hz.filter(|r| *r > 0.0).ok_or_else(|| {
            Error::invalid(
                "stream",
                format!("`{}` has no sample rate to cut by", s.name),
            )
        })?;
        let episode_clock = data
            .episodes
            .get(&s.episode)
            .ok_or_else(|| Error::NotFound {
                what: format!("episode {}", s.episode),
            })?
            .clock;
        let to_episode =
            self.resolve_clock(s.clock, episode_clock)?
                .ok_or_else(|| Error::NotFound {
                    what: format!("a mapping from the clock of stream {stream} to its episode's"),
                })?;
        let to_stream = to_episode.invert();
        let (a, b) = (
            to_stream.apply(interval.start_ns),
            to_stream.apply(interval.end_ns),
        );
        let period_ns = 1e9 / rate;

        let blobs = self.blob_store()?;
        let (mut bytes, mut count) = (Vec::new(), 0u64);
        let mut covered: Option<(i64, i64)> = None;
        for chunk in data.chunks.get(&stream).into_iter().flatten() {
            let start = chunk.chunk.interval.start_ns;
            let n = i64::try_from(chunk.chunk.samples).map_err(|_| {
                Error::corrupt(
                    format!("chunk {}", chunk.id),
                    "its sample count is beyond any real stream",
                )
            })?;
            let first = ceil_stable(a.saturating_sub(start) as f64 / period_ns).clamp(0, n);
            let last = ceil_stable(b.saturating_sub(start) as f64 / period_ns).clamp(0, n);
            if last <= first {
                continue;
            }
            let content = blobs.get(&chunk.chunk.content)?;
            if Some(content.len())
                != usize::try_from(chunk.chunk.samples)
                    .ok()
                    .and_then(|n| n.checked_mul(sample_bytes))
            {
                return Err(Error::corrupt(
                    format!("chunk {}", chunk.id),
                    format!(
                        "{} bytes do not hold {} samples of {sample_bytes} bytes",
                        content.len(),
                        n
                    ),
                ));
            }
            bytes.extend_from_slice(
                &content[first as usize * sample_bytes..last as usize * sample_bytes],
            );
            count += (last - first) as u64;
            let from = start + (first as f64 * period_ns).round() as i64;
            let to = start + (last as f64 * period_ns).round() as i64;
            covered = Some(covered.map_or((from, to), |(f, t)| (f.min(from), t.max(to))));
        }
        let (from, to) = covered.unwrap_or((a, a));
        let actual = TimeRange {
            start_ns: to_episode.apply(from),
            end_ns: to_episode.apply(to),
        };
        Ok(Samples {
            interval: actual,
            count,
            sample_bytes,
            bytes,
        })
    }
}
