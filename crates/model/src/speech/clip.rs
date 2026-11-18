// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements spoken dialogue systems that exchange audio
// with speech services in plain, portable containers, for its clients. If
// your team needs expertise in audio handling around recognition and
// synthesis services, you can procure our services by sending an email to
// info@swedishembedded.com.

//! A clip of speech, and the WAV container it travels in.
//!
//! A clip is mono `f32` samples in `[-1, 1]` at a sample rate. It is written as
//! a 16-bit PCM mono WAV file, the container brain's speech routes take and
//! give. Reading accepts what recorders commonly write: integer PCM of 8, 16,
//! 24 or 32 bits and 32-bit float, plain or in the extensible header, any
//! number of channels (averaged to mono), at the file's own rate. Anything
//! else is refused by name rather than read as noise.

use std::path::Path;

use crate::error::PolicyError;

/// Mono speech: samples in `[-1, 1]` at a sample rate.
#[derive(Clone, Debug, PartialEq)]
pub struct Clip {
    samples: Vec<f32>,
    sample_rate: u32,
}

/// Why bytes could not be read as a WAV file.
#[derive(Debug, thiserror::Error)]
#[error("not a WAV file this reader takes: {0}")]
pub struct WavError(String);

const PCM: u16 = 1;
const FLOAT: u16 = 3;
const EXTENSIBLE: u16 = 0xFFFE;

impl Clip {
    /// `samples` at `sample_rate` Hz.
    #[must_use]
    pub fn new(samples: Vec<f32>, sample_rate: u32) -> Self {
        Self {
            samples,
            sample_rate,
        }
    }

    /// The samples.
    #[must_use]
    pub fn samples(&self) -> &[f32] {
        &self.samples
    }

    /// Samples per second.
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// The length in seconds; zero for a clip with no rate.
    #[must_use]
    pub fn seconds(&self) -> f64 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        self.samples.len() as f64 / f64::from(self.sample_rate)
    }

    /// The clip as a 16-bit PCM mono WAV file. A sample outside `[-1, 1]` is
    /// clipped to it.
    #[must_use]
    pub fn to_wav(&self) -> Vec<u8> {
        let data_len = u32::try_from(self.samples.len() * 2).unwrap_or(u32::MAX);
        let mut out = Vec::with_capacity(44 + self.samples.len() * 2);
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(36u32.saturating_add(data_len)).to_le_bytes());
        out.extend_from_slice(b"WAVE");
        out.extend_from_slice(b"fmt ");
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&PCM.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&self.sample_rate.to_le_bytes());
        out.extend_from_slice(&(self.sample_rate.saturating_mul(2)).to_le_bytes());
        out.extend_from_slice(&2u16.to_le_bytes());
        out.extend_from_slice(&16u16.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&data_len.to_le_bytes());
        for sample in &self.samples {
            let value = (sample.clamp(-1.0, 1.0) * 32767.0).round() as i16;
            out.extend_from_slice(&value.to_le_bytes());
        }
        out
    }

    /// Writes [`Self::to_wav`] to `path`.
    ///
    /// # Errors
    /// The file cannot be written.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), PolicyError> {
        let path = path.as_ref();
        std::fs::write(path, self.to_wav()).map_err(|source| PolicyError::Io {
            path: path.to_path_buf(),
            source,
        })
    }

    /// The clip a WAV file's bytes hold, its channels averaged to mono.
    ///
    /// # Errors
    /// The bytes are not a RIFF/WAVE file, lack a format or data chunk, or
    /// hold a sample format this reader does not take.
    pub fn from_wav(bytes: &[u8]) -> Result<Self, WavError> {
        let refused = |why: &str| WavError(why.to_string());
        if bytes.len() < 12 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
            return Err(refused("no RIFF/WAVE header"));
        }
        let mut format: Option<Format> = None;
        let mut at = 12;
        while at + 8 <= bytes.len() {
            let id = &bytes[at..at + 4];
            let declared = u32_at(bytes, at + 4) as usize;
            let body_start = at + 8;
            // A streaming writer may leave the length unset: the chunk is the rest.
            let body_end = body_start.saturating_add(declared).min(bytes.len());
            let body = &bytes[body_start..body_end];
            match id {
                b"fmt " => format = Some(Format::read(body)?),
                b"data" => {
                    let format =
                        format.ok_or_else(|| refused("the data comes before its format"))?;
                    return Ok(format.decode(body));
                }
                _ => {}
            }
            at = body_start
                .saturating_add(declared)
                .saturating_add(declared % 2);
        }
        Err(refused(if format.is_some() {
            "no data chunk"
        } else {
            "no format chunk"
        }))
    }
}

/// What a `fmt ` chunk says about the samples.
#[derive(Clone, Copy)]
struct Format {
    float: bool,
    channels: usize,
    sample_rate: u32,
    bytes_per_sample: usize,
}

impl Format {
    fn read(body: &[u8]) -> Result<Self, WavError> {
        if body.len() < 16 {
            return Err(WavError("a format chunk shorter than 16 bytes".into()));
        }
        let mut tag = u16_at(body, 0);
        if tag == EXTENSIBLE {
            if body.len() < 26 {
                return Err(WavError(
                    "an extensible format chunk without its subformat".into(),
                ));
            }
            tag = u16_at(body, 24);
        }
        let channels = usize::from(u16_at(body, 2));
        let sample_rate = u32_at(body, 4);
        let bits = u16_at(body, 14);
        let float = match (tag, bits) {
            (PCM, 8 | 16 | 24 | 32) => false,
            (FLOAT, 32) => true,
            _ => {
                return Err(WavError(format!(
                    "format tag {tag} with {bits} bits per sample (integer PCM of 8, 16, 24 or 32 bits, or 32-bit float, is read)"
                )))
            }
        };
        if channels == 0 || sample_rate == 0 {
            return Err(WavError("no channels or no sample rate".into()));
        }
        Ok(Self {
            float,
            channels,
            sample_rate,
            bytes_per_sample: usize::from(bits / 8),
        })
    }

    fn decode(self, data: &[u8]) -> Clip {
        let frame = self.bytes_per_sample * self.channels;
        let samples = data
            .chunks_exact(frame)
            .map(|frame| {
                let sum: f32 = frame
                    .chunks_exact(self.bytes_per_sample)
                    .map(|sample| self.sample(sample))
                    .sum();
                sum / self.channels as f32
            })
            .collect();
        Clip::new(samples, self.sample_rate)
    }

    fn sample(self, bytes: &[u8]) -> f32 {
        match (self.float, bytes.len()) {
            (true, _) => f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            // 8-bit PCM is unsigned around 128.
            (false, 1) => (f32::from(bytes[0]) - 128.0) / 128.0,
            (false, 2) => f32::from(i16::from_le_bytes([bytes[0], bytes[1]])) / 32768.0,
            (false, 3) => {
                let value = i32::from_le_bytes([0, bytes[0], bytes[1], bytes[2]]) >> 8;
                value as f32 / 8_388_608.0
            }
            (false, _) => {
                i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f32
                    / 2_147_483_648.0
            }
        }
    }
}

fn u16_at(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([bytes[at], bytes[at + 1]])
}

fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A WAV header and data with the given format fields.
    fn wav(tag: u16, channels: u16, rate: u32, bits: u16, data: &[u8]) -> Vec<u8> {
        let mut out = b"RIFF\0\0\0\0WAVE".to_vec();
        out.extend_from_slice(b"LIST");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(b"abc\0");
        out.extend_from_slice(b"fmt ");
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&tag.to_le_bytes());
        out.extend_from_slice(&channels.to_le_bytes());
        out.extend_from_slice(&rate.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&bits.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(data);
        out
    }

    #[test]
    fn a_clip_survives_its_own_wav_file_to_within_one_16_bit_step() {
        let clip = Clip::new(vec![0.0, 0.5, -0.5, 1.0, -1.0, 0.123_45], 24_000);
        let back = Clip::from_wav(&clip.to_wav()).unwrap();
        assert_eq!(back.sample_rate(), 24_000);
        assert_eq!(back.samples().len(), clip.samples().len());
        for (a, b) in clip.samples().iter().zip(back.samples()) {
            assert!((a - b).abs() <= 1.0 / 32767.0, "{a} came back as {b}");
        }
    }

    #[test]
    fn a_sample_out_of_range_is_clipped_not_wrapped() {
        let back = Clip::from_wav(&Clip::new(vec![3.0, -3.0], 8_000).to_wav()).unwrap();
        assert!(back.samples()[0] > 0.99 && back.samples()[1] < -0.99);
    }

    #[test]
    fn channels_are_averaged_and_unknown_chunks_skipped() {
        let mut data = Vec::new();
        for (left, right) in [(16_384i16, 0i16), (-16_384, -16_384)] {
            data.extend_from_slice(&left.to_le_bytes());
            data.extend_from_slice(&right.to_le_bytes());
        }
        let clip = Clip::from_wav(&wav(PCM, 2, 16_000, 16, &data)).unwrap();
        assert_eq!(clip.samples(), &[0.25, -0.5]);
        assert_eq!(clip.sample_rate(), 16_000);
    }

    #[test]
    fn float_and_wide_integer_samples_are_read_at_their_scale() {
        let float = wav(FLOAT, 1, 16_000, 32, &0.75f32.to_le_bytes());
        assert_eq!(Clip::from_wav(&float).unwrap().samples(), &[0.75]);
        let wide = wav(PCM, 1, 16_000, 24, &[0, 0, 0xC0]);
        assert_eq!(Clip::from_wav(&wide).unwrap().samples(), &[-0.5]);
        let narrow = wav(PCM, 1, 16_000, 8, &[192]);
        assert_eq!(Clip::from_wav(&narrow).unwrap().samples(), &[0.5]);
    }

    #[test]
    fn what_is_not_a_readable_wav_is_refused_by_name() {
        assert!(Clip::from_wav(b"OggS not a wav").is_err());
        let compressed = wav(0x55, 1, 16_000, 16, &[0, 0]);
        assert!(Clip::from_wav(&compressed)
            .unwrap_err()
            .to_string()
            .contains("format tag 85"));
        let mut no_data = wav(PCM, 1, 16_000, 16, &[]);
        no_data.truncate(no_data.len() - 8);
        assert!(Clip::from_wav(&no_data)
            .unwrap_err()
            .to_string()
            .contains("no data chunk"));
    }
}
